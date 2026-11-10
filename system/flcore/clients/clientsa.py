import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client, load_item, save_item


class clientSA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(self.random_seed)
        self.device = torch.device(args.device)
        self.keep_client_models_on_gpu = bool(getattr(args, "keep_client_models_on_gpu", True))

        self.loss_mse = nn.MSELoss()
        self.protos = None
        self.global_protos = {}

        self.semantic_anchors = None
        self.local_gap = 0.0
        self.global_gap = 0.0
        self.latest_eval_metrics = {}
        self.latest_train_metrics = {}

        self.lamda = float(args.lamda)
        self.mcl = float(args.margin_contrastive)
        self.cc = float(args.classifier_calibration)
        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")

    def release_model_from_device(self, model=None):
        if model is None:
            model = getattr(self, "model", None)
        if model is None:
            return
        if self.keep_client_models_on_gpu and self.device.type == "cuda":
            model.zero_grad(set_to_none=True)
            return
        super().release_model_from_device(model)

    def _move_input(self, x, y):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    @torch.no_grad()
    def _protos_from_sums(self, proto_sum, proto_count):
        if proto_sum is None or proto_count is None:
            return {}

        protos = {}
        for label in torch.nonzero(proto_count > 0, as_tuple=False).view(-1).tolist():
            protos[int(label)] = (proto_sum[label] / proto_count[label].clamp_min(1.0)).detach().cpu()
        return protos

    def train(self):
        start_time = time.time()
        trainloader = self.load_train_data(drop_last=True)
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        model.train()

        self.cal_local_gap()
        self.cal_global_gap()

        max_local_epochs = self.local_epochs
        if self.train_slow and max_local_epochs > 1:
            max_local_epochs = int(self.rng.randint(1, max(2, max_local_epochs // 2 + 1)))

        proto_sum = None
        proto_count = None
        metric_sums = defaultdict(float)
        metric_batches = 0

        try:
            for _ in range(max_local_epochs):
                for x, y in trainloader:
                    x, y = self._move_input(x, y)
                    if self.train_slow:
                        time.sleep(0.1 * abs(self.rng.rand()))

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep = model.base(x)
                        output = model.head(rep)
                        task_loss = self.loss(output, y)
                        anchors_loss, anchor_metrics = self.cal_anchors_loss(rep, model, y)
                        loss = task_loss + anchors_loss

                    with torch.no_grad():
                        rep_detached = rep.detach().float()
                        if proto_sum is None:
                            proto_sum = torch.zeros(
                                self.num_classes,
                                rep_detached.shape[1],
                                device=rep_detached.device,
                                dtype=torch.float32,
                            )
                            proto_count = torch.zeros(self.num_classes, device=rep_detached.device, dtype=torch.float32)
                        proto_sum.index_add_(0, y, rep_detached)
                        proto_count.index_add_(0, y, torch.ones_like(y, dtype=proto_count.dtype))

                    optimizer.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        continue

                    with torch.no_grad():
                        metric_sums["task_loss"] += float(task_loss.detach().float().item())
                        metric_sums["anchor_loss"] += float(anchors_loss.detach().float().item())
                        metric_sums["total_loss"] += float(loss.detach().float().item())
                        for metric_name, metric_value in anchor_metrics.items():
                            metric_sums[metric_name] += float(metric_value.detach().float().item())
                        metric_batches += 1

                    if self.amp_enabled:
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

            self.protos = self._protos_from_sums(proto_sum, proto_count)
            denom = max(metric_batches, 1)
            self.latest_train_metrics = {
                key: value / denom for key, value in metric_sums.items()
            }
            self.latest_train_metrics.update(
                {
                    "local_gap": float(self.local_gap),
                    "global_gap": float(self.global_gap),
                    "metric_batches": float(metric_batches),
                }
            )
            save_item(self.protos, self.role, "protos", self.save_folder_name)
            save_item(model, self.role, "model", self.save_folder_name)
        finally:
            self.release_model_from_device(model)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_semantic_anchors(self, semantic_anchors):
        if semantic_anchors is None:
            self.semantic_anchors = None
            return
        self.semantic_anchors = torch.as_tensor(
            semantic_anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def cal_local_gap(self):
        self.local_gap = 0.0
        if not self.protos:
            return

        anchors = []
        for anchor in self.protos.values():
            if anchor is None or isinstance(anchor, list):
                continue
            anchors.append(torch.as_tensor(anchor, device=self.device, dtype=torch.float32).view(-1))

        if len(anchors) < 2:
            return

        anchors_tensor = torch.stack(anchors, dim=0)
        dist_matrix = torch.cdist(anchors_tensor, anchors_tensor, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        margins = torch.min(dist_matrix, dim=1).values
        margins = margins[torch.isfinite(margins)]
        if margins.numel() > 0:
            self.local_gap = float(torch.mean(margins).item())

    def cal_global_gap(self):
        self.global_gap = 0.0
        if self.semantic_anchors is None or self.semantic_anchors.size(0) < 2:
            return

        with torch.no_grad():
            anchors = torch.nan_to_num(self.semantic_anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
            dist_matrix = torch.cdist(anchors, anchors, p=2)
            dist_matrix.fill_diagonal_(float("inf"))
            margins = torch.min(dist_matrix, dim=1).values
            margins = margins[torch.isfinite(margins)]
            if margins.numel() > 0:
                self.global_gap = float(torch.mean(margins).item())

    def cal_anchors_loss(self, feats, model, y):
        zero = feats.new_zeros(())
        metrics = {
            "fedsa_cc_loss": zero,
            "fedsa_mse_loss": zero,
            "fedsa_mcl_loss": zero,
        }
        if self.semantic_anchors is None:
            return zero, metrics

        anchors = self.semantic_anchors.to(device=feats.device, dtype=feats.dtype)
        if anchors.ndim != 2 or anchors.size(0) < self.num_classes:
            return zero, metrics

        y = y.long().view(-1)
        loss = zero

        if self.cc > 0:
            sa_lbl = torch.arange(self.num_classes, device=feats.device, dtype=torch.long)
            sa_logits = model.head(anchors[: self.num_classes])
            cc_loss = self.loss(sa_logits, sa_lbl) * self.cc
            loss = loss + cc_loss
            metrics["fedsa_cc_loss"] = cc_loss

        if self.lamda > 0:
            target_anchors = anchors.index_select(0, y)
            mse_loss = self.loss_mse(feats.float(), target_anchors.float()) * self.lamda
            loss = loss + mse_loss
            metrics["fedsa_mse_loss"] = mse_loss

        if self.mcl > 0:
            mcl_loss = self.cal_mcl_loss(feats, y, anchors[: self.num_classes]) * self.mcl
            loss = loss + mcl_loss
            metrics["fedsa_mcl_loss"] = mcl_loss

        return loss, metrics

    def cal_mcl_loss(self, feats, y, anchors=None):
        if anchors is None:
            anchors = self.semantic_anchors
        if anchors is None:
            return feats.new_zeros(())

        anchors = anchors.to(device=feats.device, dtype=feats.dtype)
        dist = torch.cdist(feats.float(), anchors.float(), p=2)
        gap = max(float(self.local_gap), float(self.global_gap), 0.0)
        if gap > 0:
            one_hot = F.one_hot(y.long(), self.num_classes).to(device=feats.device, dtype=dist.dtype)
            dist = dist + one_hot * gap
        return self.loss(-dist, y.long())

    def test_metrics(self):
        testloader = self.load_test_data()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model
        global_protos = getattr(self, "global_protos", None)
        if not global_protos:
            global_protos = load_item("Server", "global_protos", self.save_folder_name)
        if not global_protos:
            self.latest_eval_metrics = {}
            return 0, 1e-5, 0.0

        proto_labels = []
        proto_values = []
        proto_items = global_protos.items() if isinstance(global_protos, dict) else enumerate(global_protos)
        for label, proto in proto_items:
            if proto is None or isinstance(proto, list):
                continue
            label = int(label)
            if label < 0 or label >= self.num_classes:
                continue
            proto_values.append(torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1))
            proto_labels.append(label)

        if not proto_values:
            self.latest_eval_metrics = {}
            return 0, 1e-5, 0.0

        proto_matrix = torch.stack(proto_values, dim=0)
        proto_labels = torch.as_tensor(proto_labels, device=self.device, dtype=torch.long)

        model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        anchor_eval_num = 0
        anchor_same_dist_sum = 0.0
        anchor_neg_dist_sum = 0.0
        anchor_margin_sum = 0.0
        anchor_margins = []
        anchor_nn_correct = 0.0
        anchors = None
        if self.semantic_anchors is not None and self.semantic_anchors.size(0) >= self.num_classes:
            anchors = self.semantic_anchors[: self.num_classes].to(self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                for x, y in testloader:
                    x, y = self._move_input(x, y)
                    rep = model.base(x)

                    proto_score = (rep.detach().float().unsqueeze(1) - proto_matrix.unsqueeze(0)).pow(2).mean(dim=2)
                    pred = proto_labels.index_select(0, proto_score.argmin(dim=1))
                    test_acc += pred.eq(y).sum().item()
                    test_num += y.shape[0]

                    if anchors is not None:
                        dist = torch.cdist(rep.detach().float(), anchors.float(), p=2)
                        same_dist = dist[torch.arange(dist.size(0), device=dist.device), y]
                        neg_dist = dist.masked_fill(
                            F.one_hot(y, self.num_classes).to(dtype=torch.bool),
                            float("inf"),
                        ).min(dim=1).values
                        margin = neg_dist - same_dist
                        nearest_anchor = dist.argmin(dim=1)

                        count = int(y.numel())
                        anchor_eval_num += count
                        anchor_same_dist_sum += float(same_dist.sum().detach().item())
                        anchor_neg_dist_sum += float(neg_dist.sum().detach().item())
                        anchor_margin_sum += float(margin.sum().detach().item())
                        anchor_nn_correct += float(nearest_anchor.eq(y).float().sum().detach().item())
                        anchor_margins.append(margin.detach().cpu().float().numpy())
        finally:
            self.release_model_from_device(model)

        if test_num == 0:
            return 0, 0, 0.0

        self.latest_eval_metrics = {
            "anchor_eval_samples": float(anchor_eval_num),
            "anchor_same_dist": anchor_same_dist_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "anchor_neg_dist": anchor_neg_dist_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "anchor_margin": anchor_margin_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "anchor_margin_p10": float(np.percentile(np.concatenate(anchor_margins, axis=0), 10)) if anchor_margins else 0.0,
            "anchor_margin_values": np.concatenate(anchor_margins, axis=0) if anchor_margins else np.array([], dtype=np.float32),
            "nearest_anchor_acc": anchor_nn_correct / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
        }
        return test_acc, test_num, 0.0

    def train_metrics(self):
        trainloader = self.load_train_data(drop_last=False)
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model
        model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0.0

        try:
            with torch.no_grad():
                for x, y in trainloader:
                    x, y = self._move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        logits = model(x)
                        loss = self.loss(logits, y)
                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]
        finally:
            self.release_model_from_device(model)

        return losses, train_num


def agg_func(protos):
    agg_protos = {}
    for label, proto_list in protos.items():
        if len(proto_list) == 0:
            continue
        proto_tensor = torch.stack([proto.detach().float().cpu().view(-1) for proto in proto_list], dim=0)
        agg_protos[int(label)] = proto_tensor.mean(dim=0)
    return agg_protos
