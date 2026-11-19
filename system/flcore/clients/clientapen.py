import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from utils.augment import augment_batch
from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientfml import clientFML


class clientAPEN(clientFML):
    """FedAPEN (Qin et al., KDD'23), ported from github.com/zhenqincn/FedAPEN, algorithm 'learned_adaptive_training':
    private + shared model with mutual KL, plus an ensemble CE where the other model is detached; the ensemble weight is
    learned on a held-out 5% of the local training data after each aggregation and used for training and inference."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        data = self._read_client_data_cached(is_train=True)
        perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(self.id)).tolist()
        n_val = max(1, int(len(data) * 0.05))  # paper/code default ratio_train_adaptability = [0.95, 0.05]
        self.val_data = Subset(data, perm[:n_val])
        self.train_part = Subset(data, perm[n_val:])
        self.w = torch.tensor([0.5], device=self.device, requires_grad=True)  # weight of the private model
        self.w_opt = torch.optim.SGD([self.w], lr=1e-3)
        self.KLb = nn.KLDivLoss(reduction="batchmean")
        self._w_fresh = False  # weight already learned for the current aggregated shared model
        self.personal_head = bool(getattr(args, "apen_personal_head", False))
        self.shared_lr_scale = float(getattr(args, "apen_shared_lr_scale", 1.0))  # fairness control for our twin lr x2

    def shared_model(self):
        """The aggregated shared model; with --apen_personal_head the client keeps its own head (body FedAvg'd only)."""
        g = load_item('Server', 'global_model', self.save_folder_name)
        own = load_item(self.role, 'global_model', self.save_folder_name) if self.personal_head else None
        if own is not None:
            g.head.load_state_dict(own.head.state_dict())
        return g

    def load_train_data(self, batch_size=None, drop_last=True):
        return DataLoader(self.train_part, batch_size or self.batch_size, drop_last=drop_last, shuffle=True,
                          pin_memory=str(self.device).startswith("cuda"))

    def learn_weight(self, model, global_model):
        model.eval()
        global_model.eval()
        loader = DataLoader(self.val_data, self.batch_size, shuffle=False)
        for _ in range(10):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    p, s = model(x), global_model(x)
                self.w_opt.zero_grad()
                F.cross_entropy(self.w * p + (1 - self.w) * s, y).backward()
                self.w_opt.step()
                self.w.data.clamp_(0.0, 1.0)
        self._w_fresh = True

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = self.shared_model()
        if not self._w_fresh:
            self.learn_weight(model, global_model)
        self._w_fresh = False
        wp = float(self.w)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate * self.shared_lr_scale)
        model.train()
        global_model.train()
        start_time = time.time()

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if self.augment:  # same standard augmentation as the FedSA runs (crop + flip)
                    x = augment_batch(x)
                out, out_g = model(x), global_model(x)
                loss = self.loss(out, y) + self.KLb(F.log_softmax(out, 1), F.softmax(out_g.detach(), 1)) \
                    + self.loss(wp * out + (1 - wp) * out_g.detach(), y)
                loss_g = self.loss(out_g, y) + self.KLb(F.log_softmax(out_g, 1), F.softmax(out.detach(), 1)) \
                    + self.loss(wp * out.detach() + (1 - wp) * out_g, y)
                optimizer.zero_grad()
                optimizer_g.zero_grad()
                (loss + loss_g).backward()
                # same divergence guard as this repo's FML (not in the FedAPEN code)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        """Personalized accuracy of the learned-weight ensemble (FedAPEN inference)."""
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = self.shared_model()
        self.learn_weight(model, global_model)
        wp = float(self.w)
        test_acc = test_num = cp = cg = cgp = 0
        if getattr(self, "_log_prior", None) is None:  # local class frequencies from training data only (no test labels)
            ys = torch.as_tensor([int(y) for _, y in self.train_part])
            self._log_prior = torch.bincount(ys, minlength=self.num_classes).float().clamp_min(1).log().to(self.device)
        y_prob, y_true = [], []
        with torch.no_grad():
            for x, y in self.load_test_data():
                x, y = x.to(self.device), y.to(self.device)
                p, g = model(x), global_model(x)
                output = wp * p + (1 - wp) * g
                cp += int((p.argmax(1) == y).sum())  # diagnostics: private alone / shared alone
                cg += int((g.argmax(1) == y).sum())
                cgp += int(((g + self._log_prior).argmax(1) == y).sum())  # shared + log local prior: is the gain a prior fix?
                test_acc += int((output.argmax(1) == y).sum())
                test_num += y.shape[0]
                y_prob.append(output.cpu().numpy())
                y_true.append(label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes)))
        self.diag = (cp, cg, test_num, wp, cgp)
        auc = metrics.roc_auc_score(np.concatenate(y_true), np.concatenate(y_prob), average='micro')
        return test_acc, test_num, auc
