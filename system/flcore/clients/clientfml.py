from utils.augment import augment_batch
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item


class clientFML(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.alpha = args.alpha
        self.beta = args.beta

        self.KL = nn.KLDivLoss()
        # control (review 11-20): the shared model learns from CE only and the private KL target is detached, so the
        # shared model receives nothing from the private side; the private model's own loss is unchanged
        self.shared_ce_only = bool(getattr(args, "fml_shared_ce_only", False))
        self._log_prior = None

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        global_model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.augment:  # same standard augmentation as the FedSA runs (crop + flip)
                    x = augment_batch(x)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = model(x)
                output_g = global_model(x)
                if self.shared_ce_only:
                    loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g.detach(), dim=1)) * (1-self.alpha)
                    loss_g = self.loss(output_g, y)
                else:
                    loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)
                    loss_g = self.loss(output_g, y) * self.beta + self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) * (1-self.beta)

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()

        # model.cpu()
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        # model.to(self.device)
        model.eval()
        global_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                output_g = global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) * (1-self.alpha)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    def test_metrics(self):
        """FML inference is the private model (as before); diagnostics also score the shared model alone and shared +
        log local class frequency (training data only), with test log loss, to test whether the private model adds more
        than the local label prior (same analysis as FedAPEN)."""
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        model.eval()
        global_model.eval()
        if self._log_prior is None:
            ys = torch.as_tensor([int(y) for _, y in self._read_client_data_cached(is_train=True)])
            self._log_prior = torch.bincount(ys, minlength=self.num_classes).float().clamp_min(1).log().to(self.device)
        acc = {"private": 0, "shared": 0, "shared_prior": 0}
        nll = {k: 0.0 for k in acc}
        test_num = 0
        y_prob, y_true = [], []
        with torch.no_grad():
            for x, y in self.load_test_data():
                x = x[0].to(self.device) if type(x) == type([]) else x.to(self.device)
                y = y.to(self.device)
                p, g = model(x), global_model(x)
                for k, z in (("private", p), ("shared", g), ("shared_prior", g + self._log_prior)):
                    acc[k] += int((z.argmax(1) == y).sum())
                    nll[k] += float(F.cross_entropy(z.float(), y, reduction="sum"))
                test_num += y.shape[0]
                y_prob.append(p.detach().cpu().numpy())
                nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                lb = np.eye(nc)[y.detach().cpu().numpy()]
                y_true.append(lb[:, :2] if self.num_classes == 2 else lb)
        self.diag_fml = (acc, nll, test_num)
        from sklearn import metrics
        auc = metrics.roc_auc_score(np.concatenate(y_true), np.concatenate(y_prob), average='micro')
        return acc["private"], test_num, auc
