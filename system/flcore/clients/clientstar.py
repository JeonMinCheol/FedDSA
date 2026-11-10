import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client, load_item, save_item


class clientstar(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(self.random_seed)
        self.device = torch.device(args.device)

        self.protos = None
        self.global_protos = {}

        self.semantic_anchors = None
        self.stable_semantic_anchors = None
        self.local_gap = 0.0
        self.global_gap = 0.0
        self.local_gap_mean = 0.0
        self.local_gap_median = 0.0
        self.global_gap_mean = 0.0
        self.global_gap_median = 0.0
        self.latest_eval_metrics = {}
        self.latest_train_metrics = {}
        models = list(getattr(args, "models", []) or [])
        self.model_key = str(models[self.id % len(models)]) if len(models) > 0 else "default"

        self.lamda = args.lamda
        self.mcl = args.margin_contrastive
        self.anchor_agreement_gate = str(getattr(args, "anchor_agreement_gate", "off")).lower()
        if self.anchor_agreement_gate not in {"off", "auto"}:
            self.anchor_agreement_gate = "off"
        self.current_round = 0
        self.stable_anchor_mode = str(args.stable_anchor_mode).lower()
        if self.stable_anchor_mode not in {"single", "dual"}:
            self.stable_anchor_mode = "dual"
        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")

    def move_input(self, x, y):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    @torch.no_grad()
    def protos_from_sums(self, proto_sum, proto_count):
        if proto_sum is None or proto_count is None:
            return {}

        protos = {}
        for label in torch.nonzero(proto_count > 0, as_tuple=False).view(-1).tolist():
            label = int(label)
            count = proto_count[label].clamp_min(1.0)
            proto = proto_sum[label] / count
            protos[label] = proto.detach().cpu()
        return protos

    def forward_rep_logits(self, model, x):
        if hasattr(model, "shared_representation") and hasattr(model, "classify"):
            rep = model.shared_representation(x)
            return rep, model.classify(rep)

        rep = model.base(x)
        if isinstance(rep, tuple):
            rep = rep[-1]
        if rep.ndim > 2:
            rep = torch.flatten(rep, 1)
        output = model.head(rep)
        return rep, output

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
                    x, y = self.move_input(x, y)
                    if self.train_slow:
                        time.sleep(0.1 * abs(self.rng.rand()))

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep, output = self.forward_rep_logits(model, x)
                        task_loss = self.loss(output, y)
                        anchors_loss, anchor_metrics = self.cal_anchors_loss(rep, model, y, output)
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

            self.collect_protos(model=model)
           
            denom = max(metric_batches, 1)
            self.latest_train_metrics = {
                key: value / denom for key, value in metric_sums.items()
            }
            self.latest_train_metrics.update(
                {
                    "local_gap": float(self.local_gap),
                    "global_gap": float(self.global_gap),
                    "local_gap_mean": float(self.local_gap_mean),
                    "local_gap_median": float(self.local_gap_median),
                    "global_gap_mean": float(self.global_gap_mean),
                    "global_gap_median": float(self.global_gap_median),
                    "metric_batches": float(metric_batches),
                }
            )
            save_item(self.protos, self.role, "protos", self.save_folder_name)
            save_item(model, self.role, "model", self.save_folder_name)
        finally:
            self.release_model_from_device(model)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    @torch.no_grad()
    def collect_protos(self, model=None):
        trainloader = self.load_train_data(drop_last=False)
        loaded_here = model is None
        if model is None:
            model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model
        model.to(self.device)
        model.eval()

        proto_sum = None
        proto_count = None
        try:
            for x, y in trainloader:
                x, y = self.move_input(x, y)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep, _ = self.forward_rep_logits(model, x)

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
        finally:
            if loaded_here:
                self.release_model_from_device(model)

        self.protos = self.protos_from_sums(
            proto_sum,
            proto_count,
        )
        return self.protos

    def set_semantic_anchors(self, semantic_anchors):
        if semantic_anchors is None:
            self.semantic_anchors = None
            return
        self.semantic_anchors = torch.as_tensor(
            semantic_anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def set_stable_semantic_anchors(self, semantic_anchors):
        if semantic_anchors is None:
            self.stable_semantic_anchors = None
            return
        self.stable_semantic_anchors = torch.as_tensor(
            semantic_anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def gap_summary_from_margins(self, margins):
        margins = margins[torch.isfinite(margins)]
        if margins.numel() <= 0:
            return 0.0, 0.0, 0.0
        mean_gap = float(torch.mean(margins).item())
        median_gap = float(torch.median(margins).item())
        return mean_gap, mean_gap, median_gap

    def cal_local_gap(self):
        self.local_gap = 0.0
        self.local_gap_mean = 0.0
        self.local_gap_median = 0.0
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
        self.local_gap, self.local_gap_mean, self.local_gap_median = self.gap_summary_from_margins(margins)

    def cal_global_gap(self):
        self.global_gap = 0.0
        self.global_gap_mean = 0.0
        self.global_gap_median = 0.0
        anchors = self.anchor_bank_for_mcl()
        if anchors is None or anchors.size(0) < 2:
            return

        with torch.no_grad():
            anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
            dist_matrix = torch.cdist(anchors, anchors, p=2)
            dist_matrix.fill_diagonal_(float("inf"))
            margins = torch.min(dist_matrix, dim=1).values
            self.global_gap, self.global_gap_mean, self.global_gap_median = self.gap_summary_from_margins(margins)

    def cal_anchors_loss(self, feats, model, y, logits=None):
        zero = feats.new_zeros(())
        metrics = {
            "anchor_mse_loss": zero,
            "mcl_loss": zero,
            "anchor_agree_head_acc": zero,
            "anchor_agree_both_wrong": zero,
            "anchor_agree_recovery_floor": feats.new_ones(()),
        }
        loss = zero

        if self.semantic_anchors is None:
            return loss, metrics

        mse_anchors = self.semantic_anchors
        mcl_anchors = self.anchor_bank_for_mcl()
        if (
            mse_anchors is None
            or mcl_anchors is None
            or mse_anchors.ndim != 2
            or mcl_anchors.ndim != 2
            or mse_anchors.size(0) < self.num_classes
            or mcl_anchors.size(0) < self.num_classes
        ):
            return loss, metrics

        y = y.long().view(-1)
        mse_anchors = mse_anchors.to(device=feats.device, dtype=feats.dtype)[: self.num_classes]
        mcl_anchors = mcl_anchors.to(device=feats.device, dtype=feats.dtype)[: self.num_classes]
        agreement_weight, agreement_metrics = self.cal_anchor_agreement_weights(
            feats,
            y,
            mcl_anchors,
            logits=logits,
        )
        metrics.update(agreement_metrics)
        anchor_weight = agreement_weight

        mcl_loss = zero
        mcl_metrics = {}
        if self.mcl > 0:
            mcl_loss, mcl_metrics = self.cal_mcl_loss(
                feats,
                y,
                mcl_anchors,
                return_metrics=True,
                sample_weight=anchor_weight,
            )
            metrics.update(mcl_metrics)

        if self.lamda > 0:
            mse_loss = self.cal_anchor_mse_loss(
                feats,
                y,
                mse_anchors,
                anchor_weight,
            ) * self.lamda
            loss = loss + mse_loss
            metrics["anchor_mse_loss"] = mse_loss

        if self.mcl > 0:
            mcl_loss = mcl_loss * self.mcl
            loss = loss + mcl_loss
            metrics["mcl_loss"] = mcl_loss

        return loss, metrics

    def cal_anchor_agreement_weights(self, feats, y, anchors, logits=None):
        one = feats.new_ones(feats.size(0), dtype=torch.float32)
        zero = feats.new_zeros(())
        metrics = {
            "anchor_agree_head_acc": zero,
            "anchor_agree_both_wrong": zero,
            "anchor_agree_recovery_floor": one.mean(),
        }
        if self.anchor_agreement_gate != "auto" or anchors is None:
            return one, metrics

        anchors = anchors.to(device=feats.device, dtype=feats.dtype)
        if anchors.ndim != 2 or anchors.size(0) < self.num_classes:
            return one, metrics
        if logits is None:
            return one, metrics

        with torch.no_grad():
            y = y.long().view(-1)
            dist = torch.cdist(feats.detach().float(), anchors.float(), p=2)
            anchor_pred = dist.argmin(dim=1)
            anchor_correct = anchor_pred.eq(y)

            head_pred = logits.detach().argmax(dim=1)
            head_correct = head_pred.eq(y)

            conflict = head_correct & (~anchor_correct)
            both_wrong = (~head_correct) & (~anchor_correct)

            anchor_acc = anchor_correct.float().mean()
            head_acc = head_correct.float().mean()
            both_wrong_rate = both_wrong.float().mean()
            reliability = (anchor_acc / head_acc.clamp_min(1e-6)).clamp(0.0, 1.0)
            recovery_floor = torch.sqrt((head_acc * (1.0 - both_wrong_rate)).clamp(0.0, 1.0))

            recovery_gap = (recovery_floor - reliability).clamp_min(0.0)
            recovery_room = (1.0 - reliability).clamp_min(1e-6)
            recovery_blend = ((recovery_gap / recovery_room) * recovery_floor).clamp(0.0, 1.0)
            recovered_reliability = (
                reliability + recovery_blend * (recovery_floor - reliability)
            ).clamp(0.0, 1.0)

            conflict_weight = recovered_reliability.to(device=feats.device, dtype=torch.float32)
            both_wrong_weight = torch.sqrt(recovered_reliability).to(device=feats.device, dtype=torch.float32)
            weight = torch.ones_like(y, dtype=torch.float32, device=feats.device) * conflict_weight
            weight = torch.where(both_wrong, both_wrong_weight.expand_as(weight), weight)

        metrics = {
            "anchor_agree_head_acc": head_correct.detach().float().mean(),
            "anchor_agree_both_wrong": both_wrong.detach().float().mean(),
            "anchor_agree_recovery_floor": recovery_floor.detach(),
        }
        return weight.detach().to(device=feats.device, dtype=torch.float32), metrics

    def cal_anchor_mse_loss(self, feats, y, anchors, sample_weight=None):
        feats = feats.float()
        anchors = anchors.to(device=feats.device, dtype=torch.float32)
        y = y.long().view(-1)
        if feats.numel() == 0 or anchors.ndim != 2 or anchors.size(0) < self.num_classes:
            return feats.new_zeros(())

        target_anchors = anchors.index_select(0, y).float()
        per_sample_loss = F.mse_loss(feats, target_anchors, reduction="none").mean(dim=1)
        if sample_weight is None:
            return per_sample_loss.mean()
        else:
            weights = sample_weight.to(device=feats.device, dtype=feats.dtype).view(-1)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
            return (per_sample_loss * weights).mean()

    def cal_mcl_loss(self, feats, y, anchors=None, return_metrics=False, sample_weight=None):
        if anchors is None:
            anchors = self.anchor_bank_for_mcl()
        if anchors is None:
            zero = feats.new_zeros(())
            return (zero, {}) if return_metrics else zero

        y = y.long().view(-1)
        anchors = anchors.to(device=feats.device, dtype=feats.dtype)
        dist = torch.cdist(feats.float(), anchors.float(), p=2)
        row_idx = torch.arange(dist.size(0), device=dist.device)
        same_dist = dist[row_idx, y]
        neg_dist = dist.masked_fill(
            F.one_hot(y, self.num_classes).to(device=dist.device, dtype=torch.bool),
            float("inf"),
        ).min(dim=1).values
        sample_margin = neg_dist - same_dist

        local_gap = max(float(self.local_gap), 0.0)
        global_gap = max(float(self.global_gap), 0.0)
        gap = min(local_gap, global_gap) if local_gap > 0 and global_gap > 0 else 0.0
        if sample_weight is None:
            effective_sample_weight = torch.ones_like(sample_margin.float())
        else:
            effective_sample_weight = sample_weight.to(device=sample_margin.device, dtype=torch.float32)

        base_metrics = {
            "mcl_tail_frac": (sample_margin.detach() < 0).float().mean(),
            "mcl_sample_margin": torch.nan_to_num(sample_margin.detach().float()).mean(),
        }

        if gap > 0:
            one_hot = F.one_hot(y, self.num_classes).to(device=feats.device, dtype=dist.dtype)
            dist = dist + one_hot * gap
        logits = -dist
        per_sample_loss = F.cross_entropy(logits, y, reduction="none")
        raw_loss = per_sample_loss.mean()

        loss = (per_sample_loss * effective_sample_weight).mean()
        metrics = dict(base_metrics)
        metrics.update(
            {
                "mcl_raw_loss": raw_loss.detach(),
            }
        )
        return (loss, metrics) if return_metrics else loss

    def anchor_bank_for_mcl(self):
        if self.stable_anchor_mode == "dual" and self.stable_semantic_anchors is not None:
            return self.stable_semantic_anchors
        return self.semantic_anchors

    def test_metrics(self):
        testloader = self.load_test_data()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model
        model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        anchor_eval_num = 0
        anchor_same_dist_sum = 0.0
        anchor_neg_dist_sum = 0.0
        anchor_margin_sum = 0.0
        anchor_nn_correct = 0.0
        anchors = None
        anchor_bank = self.anchor_bank_for_mcl()
        if anchor_bank is not None and anchor_bank.size(0) >= self.num_classes:
            anchors = anchor_bank[: self.num_classes].to(self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                for x, y in testloader:
                    x, y = self.move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep, logits = self.forward_rep_logits(model, x)

                    head_pred = torch.argmax(logits, dim=1)
                    head_hits = head_pred.eq(y)
                    test_acc += head_hits.sum().item()
                    test_num += y.shape[0]

                    y_prob.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                    nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                    labels = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        labels = labels[:, :2]
                    y_true.append(labels)

                    if anchors is not None:
                        dist = torch.cdist(rep.detach().float(), anchors.float(), p=2)
                        same_dist = dist[torch.arange(dist.size(0), device=dist.device), y]
                        neg_dist = dist.masked_fill(
                            F.one_hot(y, self.num_classes).to(dtype=torch.bool),
                            float("inf"),
                        ).min(dim=1).values
                        margin = neg_dist - same_dist
                        nearest_anchor = dist.argmin(dim=1)
                        finite = torch.isfinite(same_dist) & torch.isfinite(neg_dist) & torch.isfinite(margin)

                        count = int(finite.sum().detach().item())
                        if count > 0:
                            same_dist = same_dist[finite]
                            neg_dist = neg_dist[finite]
                            margin = margin[finite]
                            nearest_anchor = nearest_anchor[finite]
                            y_eval = y[finite]

                            anchor_eval_num += count
                            anchor_same_dist_sum += float(same_dist.sum().detach().item())
                            anchor_neg_dist_sum += float(neg_dist.sum().detach().item())
                            anchor_margin_sum += float(margin.sum().detach().item())
                            anchor_nn_correct += float(nearest_anchor.eq(y_eval).float().sum().detach().item())
        finally:
            self.release_model_from_device(model)

        if len(y_prob) == 0:
            return 0, 0, 0.0

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average="micro")
        except Exception:
            auc = 0.0
        self.latest_eval_metrics = {
            "anchor_eval_samples": float(anchor_eval_num),
            "anchor_same_dist": anchor_same_dist_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "anchor_neg_dist": anchor_neg_dist_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "anchor_margin": anchor_margin_sum / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
            "nearest_anchor_acc": anchor_nn_correct / float(anchor_eval_num) if anchor_eval_num > 0 else 0.0,
        }
        return test_acc, test_num, auc

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
                    x, y = self.move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep, logits = self.forward_rep_logits(model, x)
                        loss = self.loss(logits, y)
                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]
        finally:
            self.release_model_from_device(model)

        return losses, train_num
