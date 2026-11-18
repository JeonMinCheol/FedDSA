import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientfml import clientFML
from utils.augment import augment_batch


class SwitchNorm1d(nn.Module):
    """Switchable Normalization (Luo et al., ICLR 2019) for (N, C) inputs, as in the authors' 1-D variant: the
    normalizing mean and variance are learned softmax mixtures of layer-norm (per sample) and batch-norm (per feature)
    statistics; batch statistics use running averages at eval. Running stats start like BatchNorm (mean 0, var 1)
    and use BatchNorm's momentum, because the original 0.997 barely moves within the few local steps of a round."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.ones(1, num_features))
        self.eps, self.momentum = eps, momentum

    def forward(self, x):
        mean_ln, var_ln = x.mean(1, keepdim=True), x.var(1, keepdim=True)
        if self.training:
            mean_bn, var_bn = x.mean(0, keepdim=True), x.var(0, keepdim=True)
            with torch.no_grad():
                self.running_mean.lerp_(mean_bn, self.momentum)
                self.running_var.lerp_(var_bn, self.momentum)
        else:
            mean_bn, var_bn = self.running_mean, self.running_var
        mw, vw = self.mean_weight.softmax(0), self.var_weight.softmax(0)
        mean = mw[0] * mean_ln + mw[1] * mean_bn
        var = vw[0] * var_ln + vw[1] * var_bn
        return (x - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class GateNet(nn.Module):
    """pFedMoE gating network (Yi et al. 2024, arXiv 2402.01350, Sec. 4 / Eq. 4): the flattened sample -> switch norm ->
    Linear -> BN -> Sigmoid -> Linear(2) -> BN -> Softmax = per-sample weights of the (global, local) experts.
    The paper does not report the hidden width; 128 here (--pfedmoe_gate_hidden)."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), SwitchNorm1d(in_dim), nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                                 nn.Sigmoid(), nn.Linear(hidden, 2), nn.BatchNorm1d(2), nn.Softmax(dim=1))

    def forward(self, x):
        return self.net(x)


class clientPFedMoE(clientFML):
    """pFedMoE: shared homogeneous small feature extractor (global expert, FedAvg'd) + my heterogeneous large feature
    extractor (local expert); a local gate mixes their representations per sample (Eq. 5) and my own header predicts
    (Eq. 6). Everything trains end to end on CE (Eq. 7-8) with the same lr for both extractors; only the shared
    extractor is uploaded. Inference uses the same mixture."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.gate_lr = float(getattr(args, "pfedmoe_gate_lr", 0.01))
        self.gate_hidden = int(getattr(args, "pfedmoe_gate_hidden", 128))
        self.shared_lr_scale = float(getattr(args, "pfedmoe_shared_lr_scale", 1.0))  # fairness control (our twin uses x3)
        self.gate = None  # built on first use from the input size

    def _mix(self, model, shared, x):
        a = self.gate(x)
        rep = a[:, :1] * shared.base(x) + a[:, 1:] * model.base(x)
        return model.head(rep), a

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        shared = load_item('Server', 'global_model', self.save_folder_name)  # a copy of the received extractor
        if self.gate is None:
            x0, _ = next(iter(trainloader))
            self.gate = GateNet(x0[0].numel(), self.gate_hidden).to(self.device)
        params = list(model.parameters()) + list(shared.base.parameters())
        # paper: eta_theta = eta_omega for stability; --pfedmoe_shared_lr_scale != 1 is our fairness control only
        optimizer = torch.optim.SGD([{"params": list(model.parameters())},
                                     {"params": list(shared.base.parameters()), "lr": self.learning_rate * self.shared_lr_scale}],
                                    lr=self.learning_rate)
        optimizer_gate = torch.optim.SGD(self.gate.parameters(), lr=self.gate_lr)
        model.train()
        shared.train()
        self.gate.train()
        start_time = time.time()

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if self.augment:  # same standard augmentation as the other baselines (off in the main setting)
                    x = augment_batch(x)
                out, _ = self._mix(model, shared, x)
                loss = self.loss(out, y)
                optimizer.zero_grad()
                optimizer_gate.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10)  # same divergence guard as this repo's FML/FedAPEN
                optimizer.step()
                optimizer_gate.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(shared, self.role, 'global_model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    @torch.no_grad()
    def test_metrics(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        shared = load_item('Server', 'global_model', self.save_folder_name)
        if self.gate is None:  # round 0: evaluated before any training, an untrained gate
            x0, _ = next(iter(self.load_test_data()))
            self.gate = GateNet(x0[0].numel(), self.gate_hidden).to(self.device)
        model.eval()
        shared.eval()
        self.gate.eval()
        test_acc = test_num = 0
        w_global = 0.0
        y_prob, y_true = [], []
        for x, y in self.load_test_data():
            x, y = x.to(self.device), y.to(self.device)
            out, a = self._mix(model, shared, x)
            test_acc += int((out.argmax(1) == y).sum())
            test_num += y.shape[0]
            w_global += float(a[:, 0].sum())
            y_prob.append(out.cpu().numpy())
            y_true.append(label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes)))
        self.diag = (w_global, test_num)
        auc = metrics.roc_auc_score(np.concatenate(y_true), np.concatenate(y_prob), average='micro')
        return test_acc, test_num, auc

    @torch.no_grad()
    def train_metrics(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        shared = load_item('Server', 'global_model', self.save_folder_name)
        if self.gate is None:
            return 0.0, 1
        model.eval()
        shared.eval()
        self.gate.eval()
        losses = train_num = 0
        for x, y in self.load_train_data():
            x, y = x.to(self.device), y.to(self.device)
            out, _ = self._mix(model, shared, x)
            losses += float(self.loss(out, y)) * y.shape[0]
            train_num += y.shape[0]
        return losses, train_num
