import time
import hashlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client, load_item, save_item
from utils.feddsa_utils import configure_utility_module
from utils.distance_numerics import guarded_cdist, nearest_gap
from utils.mcl_gap import add_mcl_gap, sample_gap_metrics


class clientfeddsa(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        if float(args.lamda) != 1.0 or float(args.margin_contrastive) != 1.0:
            raise ValueError("FedDSA uses unit MSE and MCL coefficients; -lam and -mcl must both be 1.0")
        configure_utility_module(args)
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.candidate_gain_enabled = args.candidate_gain_enabled
        self.candidate_next_gap = args.candidate_next_gap
        self.mcl_gap_class_cap = args.mcl_gap_class_cap
        self.raar_diagnostics = bool(getattr(args, "raar_diagnostics", False))
        self.anchor_candidate_evidence = None
        torch.manual_seed(self.random_seed)
        self.device = torch.device(args.device)
        self.keep_client_models_on_gpu = bool(
            getattr(args, "keep_client_models_on_gpu", True)
        )

        self.protos = None
        self.meta_anchor_grad = None
        self.semantic_anchors = None
        self.adaptive_anchors = None
        self.stable_semantic_anchors = None
        self.latest_train_metrics = {}
        self.current_round = 0
        self.lamda = self.mcl = 1.0
        self.mcl_variant = str(getattr(args, "mcl_variant", "fedsa")).lower()
        if self.mcl_variant not in {"fedsa", "hard_negative"}:
            self.mcl_variant = "fedsa"
        self.mcl_gap_mode = str(getattr(args, "mcl_gap_mode", "min")).lower()
        if self.mcl_gap_mode not in {"min", "max"}:
            self.mcl_gap_mode = "min"
        self.mcl_hard_margin = max(
            float(getattr(args, "mcl_hard_margin", 0.5)), 0.0
        )
        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")

        self.local_gap = 0.0
        self.global_gap = 0.0
        self.meta_enabled = str(getattr(args, "semantic_gate_mode", "off")).lower() == "meta"
        self.meta_inner_lr = float(getattr(args, "meta_inner_lr", 0.01))
        self.new_anchor_mode = str(getattr(args, "new_anchor_mode", "none")).lower()
        self.anchor_mode = "single"
        self.separate_anchor_roles = bool(
            getattr(args, "separate_anchor_roles", True)
        )
        self.reliability_anchor_source = str(
            getattr(args, "reliability_anchor_source", "mcl")
        ).lower()
        if self.reliability_anchor_source not in {"mcl", "semantic"}:
            self.reliability_anchor_source = "mcl"
        self.scalar_reliability_state = bool(
            getattr(args, "scalar_reliability_state", False)
        )
        self.class_reliability = None
        self.prototype_utility = bool(
            getattr(args, "prototype_utility", False)
        )
        self.utility_inner_lr = max(
            float(getattr(args, "utility_inner_lr", 0.01)), 0.0
        )
        self.utility_temperature = max(
            float(getattr(args, "utility_temperature", 0.1)), 1e-6
        )
        self.utility_min_weight = min(
            max(float(getattr(args, "utility_min_weight", 0.25)), 0.0), 1.0
        )
        self.utility_max_weight = max(
            float(getattr(args, "utility_max_weight", 1.5)),
            self.utility_min_weight,
        )
        self.utility_local_gate = bool(getattr(args, "utility_local_gate", True))
        self.utility_weight_normalize = bool(
            getattr(args, "utility_weight_normalize", True)
        )
        self.utility_positive_only = bool(
            getattr(args, "utility_positive_only", False)
        )
        self.class_utility = None
        self.latest_utility_metrics = {}
        self.capture_utility_gate = False
        self.last_local_utility_gate = None
        self.agreement_weighting = bool(
            getattr(args, "agreement_weighting", True)
        )
        self.dual_disagreement_gate = bool(
            getattr(args, "dual_disagreement_gate", False)
        )
        self.dual_disagreement_temperature = max(
            float(getattr(args, "dual_disagreement_temperature", 0.05)), 1e-6
        )
        self.boundary_loss_weight = max(
            float(getattr(args, "boundary_loss_weight", 1.0)), 0.0
        )
        self.class_boundary_margin = None
        self.utility_validation_probe = bool(
            getattr(args, "utility_validation_probe", False)
        )
        self.validation_enabled = (
            self.new_anchor_mode == "observation_validation"
            or self.utility_validation_probe
        )
        self.validation_anchor_grad = None
        self.validation_anchor_gain = None
        self.validation_anchor_gain_count = None
        self.validation_anchor_before_loss = 0.0
        self.validation_anchor_observation_loss = 0.0

    def set_round(self, round_idx):
        self.current_round = int(round_idx)

    def load_train_data(self, batch_size=None, drop_last=True, *, phase="train"):
        train_data = self._read_client_data_cached(is_train=True)
        self.current_train_samples = len(train_data)
        # Each phase replays independently; optional CAU never advances training sampling.
        key = f"{self.random_seed}:{self.id}:{self.current_round}:{phase}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return DataLoader(
            train_data,
            self.batch_size if batch_size is None else batch_size,
            drop_last=drop_last,
            shuffle=bool(train_data),
            pin_memory=str(self.device).startswith("cuda"),
            generator=generator,
        )


    def move_input(self, x, y):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    def set_semantic_anchors(self, anchors):
        if anchors is None:
            self.semantic_anchors = None
            return
        self.semantic_anchors = torch.as_tensor(
            anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def set_stable_semantic_anchors(self, anchors):
        if anchors is None:
            self.stable_semantic_anchors = None
            return
        self.stable_semantic_anchors = torch.as_tensor(
            anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def set_adaptive_anchors(self, anchors):
        """Set the explicitly performance/adaptation-driven anchor state."""

        if anchors is None:
            self.adaptive_anchors = None
            return
        self.adaptive_anchors = torch.as_tensor(
            anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

    def set_class_reliability(self, reliability):
        if reliability is None:
            self.class_reliability = None
            return
        self.class_reliability = torch.as_tensor(
            reliability, device=self.device, dtype=torch.float32
        ).detach().clone()

    def set_class_utility(self, utility):
        if utility is None:
            self.class_utility = None
            return
        self.class_utility = torch.as_tensor(
            utility, device=self.device, dtype=torch.float32
        ).detach().clone()

    def set_class_boundary_margin(self, margin):
        if margin is None:
            self.class_boundary_margin = None
            return
        self.class_boundary_margin = torch.as_tensor(
            margin, device=self.device, dtype=torch.float32
        ).detach().clone()

    def forward_rep_logits(self, model, x):
        rep = model.shared_representation(x)
        return rep, model.classify(rep)

    def anchor_bank_for_mcl(self):
        if self.anchor_mode == "single":
            return self.semantic_anchors
        if self.new_anchor_mode == "class_boundary":
            # The second state is a scalar margin per class; MCL still uses
            # the semantic vector bank as its class center.
            return self.semantic_anchors
        if self.new_anchor_mode in {
            "semantic_discriminative",
            "observation_validation",
            "relation_memory",
            "meta_separated",
        } and self.adaptive_anchors is not None:
            adaptive = self.adaptive_anchors
            if (
                self.scalar_reliability_state
                and self.class_reliability is not None
                and self.semantic_anchors is not None
            ):
                reliability = self.class_reliability[: self.num_classes].to(
                    device=adaptive.device, dtype=adaptive.dtype
                ).view(-1, 1).clamp(0.0, 1.0)
                semantic = self.semantic_anchors[: self.num_classes].to(
                    device=adaptive.device, dtype=adaptive.dtype
                )
                return semantic + reliability * (adaptive[: self.num_classes] - semantic)
            return adaptive
        if self.stable_semantic_anchors is not None:
            return self.stable_semantic_anchors
        return self.semantic_anchors

    def compute_anchor_utility(self, model):
        """Keep CAU's two batches; cache a third for RAAR's separate decision."""
        from utils.head_counterfactual_utility import head_gains
        self.anchor_utility = self.anchor_utility_count = None
        self.validation_anchor_gain = self.validation_anchor_gain_count = None
        self.anchor_candidate_evidence = None
        self.latest_utility_metrics = {}
        candidate = bool(getattr(self, "candidate_gain_enabled", False))
        if not candidate and getattr(self, "current_round", 0) % getattr(self, "utility_interval_rounds", 1):
            return None
        if (not self.prototype_utility and not candidate) or self.semantic_anchors is None:
            return None
        third_batch = self.utility_validation_probe or candidate
        needed = 3 if third_batch else 2
        diagnostic = candidate and getattr(self, "raar_diagnostics", False)
        batches = []
        for x, y in self.load_train_data(drop_last=False, phase="utility"):
            batches.append(self.move_input(x, y))
            if len(batches) == (4 if diagnostic else needed):
                break
        if len(batches) < needed:
            return None
        result = head_gains(self, model, batches[0], batches[1],
                            batches[2] if third_batch else None,
                            batches[3] if diagnostic and len(batches) > 3 else None)
        self.anchor_utility, self.anchor_utility_count = result[0]
        if self.utility_validation_probe:
            self.validation_anchor_gain, self.validation_anchor_gain_count = result[1]
        valid = self.anchor_utility_count > 0
        self.latest_utility_metrics = {
            "utility_mean": float(self.anchor_utility[valid].mean()) if valid.any() else 0.,
            "utility_positive_rate": float((self.anchor_utility[valid] > 0).float().mean()) if valid.any() else 0.,
        }
        return self.anchor_utility

    @torch.no_grad()
    def protos_from_sums(self, proto_sum, proto_count):
        if proto_sum is None or proto_count is None:
            return {}

        protos = {}
        for label in torch.nonzero(proto_count > 0, as_tuple=False).view(-1).tolist():
            label = int(label)
            protos[label] = (proto_sum[label] / proto_count[label].clamp_min(1.0)).detach().cpu()
        return protos

    @torch.no_grad()
    def collect_protos(self, model=None):
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
            for x, y in self.load_train_data(drop_last=False, phase="prototype"):
                x, y = self.move_input(x, y)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep, _ = self.forward_rep_logits(model, x)
                rep = rep.detach().float()

                if proto_sum is None:
                    proto_sum = torch.zeros(self.num_classes, rep.size(1), device=rep.device)
                    proto_count = torch.zeros(self.num_classes, device=rep.device)
                proto_sum.index_add_(0, y, rep)
                proto_count.index_add_(0, y, torch.ones_like(y, dtype=proto_count.dtype))
        finally:
            if loaded_here:
                self.release_model_from_device(model)

        self.protos = self.protos_from_sums(proto_sum, proto_count)
        return self.protos

    def compute_meta_anchor_gradient(self, model):
        """Differentiate one projector/classifier update through a query CE loss."""

        self.meta_anchor_grad = None
        if not self.meta_enabled or self.semantic_anchors is None:
            return None
        # In the single-anchor control the adaptive role is not consumed by
        # local training.  Do not update an unused state through a virtual
        # loss, otherwise the control would no longer isolate the dual-role
        # contribution.
        if self.new_anchor_mode == "meta_separated" and self.anchor_mode == "single":
            return None
        if self.new_anchor_mode == "meta_separated":
            if self.adaptive_anchors is None:
                return None
        elif self.stable_semantic_anchors is None:
            return None
        projector = getattr(model, "shared_projector", None)
        head = getattr(model, "shared_head", None)
        if not isinstance(projector, torch.nn.Linear) or head is None:
            return None

        batches = []
        for x, y in self.load_train_data(drop_last=False, phase="meta"):
            if isinstance(x, (list, tuple)):
                x = x[0]
            batches.append((x, y.long().view(-1)))
            if len(batches) >= 2:
                break
        if len(batches) < 2:
            return None
        sx, sy = batches[0]
        qx, qy = batches[1]
        sx, sy = self.move_input(sx, sy)
        qx, qy = self.move_input(qx, qy)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            support_feat = model.extract_backbone(sx).detach().float()
            query_feat = model.extract_backbone(qx).detach().float()

        proj_weight = projector.weight
        proj_bias = projector.bias
        head_weight = head.weight
        head_bias = head.bias
        params = [proj_weight, proj_bias, head_weight]
        if head_bias is not None:
            params.append(head_bias)

        if self.new_anchor_mode == "meta_separated":
            # The semantic anchor is the MSE target; the adaptive anchor is
            # the discriminative MCL target whose position is meta-learned.
            adaptive = self.adaptive_anchors[: self.num_classes].float().detach().clone()
            mcl_target = adaptive
        else:
            # Preserve the historical meta mode for backwards-compatible
            # comparisons: it meta-updates the semantic state and uses the
            # stable state for MCL.
            adaptive = self.semantic_anchors[: self.num_classes].float().detach().clone()
            mcl_target = self.stable_semantic_anchors[: self.num_classes].float().detach()
        adaptive.requires_grad_(True)
        z = F.linear(support_feat, proj_weight, proj_bias)
        support_logits = F.linear(z, head_weight, head_bias)
        support_task = F.cross_entropy(support_logits, sy)
        semantic_target = self.semantic_anchors[: self.num_classes].float().detach()
        mse_target = semantic_target if self.new_anchor_mode == "meta_separated" else adaptive
        support_mse = F.mse_loss(z, mse_target.index_select(0, sy))
        support_mcl = F.cross_entropy(-torch.cdist(z.float(), mcl_target.float(), p=2), sy)
        support_loss = support_task + support_mse + support_mcl
        grads = torch.autograd.grad(support_loss, params, create_graph=True, allow_unused=False)
        updated = [param - self.meta_inner_lr * grad for param, grad in zip(params, grads)]
        up_w, up_b, up_head_w = updated[:3]
        up_head_b = updated[3] if len(updated) > 3 else None
        query_z = F.linear(query_feat, up_w, up_b)
        query_logits = F.linear(query_z, up_head_w, up_head_b)
        query_loss = F.cross_entropy(query_logits, qy)
        anchor_grad = torch.autograd.grad(query_loss, adaptive, allow_unused=False)[0]
        self.meta_anchor_grad = anchor_grad.detach().cpu()
        if was_training:
            model.train()
        return self.meta_anchor_grad

    def compute_validation_anchor_gradient(self, model):
        """Use a held-out train split as a validation signal for the anchor.

        The query batch is drawn from the client's training partition and is
        never read from ``load_test_data``.  The model representation is
        detached; only the validation anchor is differentiated.  This gives
        the server a common C x d signal that can be aggregated across
        heterogeneous client models.
        """

        if self.utility_validation_probe:
            # The utility probe already evaluated the with/without-anchor
            # branches on a disjoint validation batch.
            return self.validation_anchor_grad
        self.validation_anchor_grad = None
        self.validation_anchor_gain = None
        self.validation_anchor_gain_count = None
        self.validation_anchor_before_loss = 0.0
        self.validation_anchor_observation_loss = 0.0
        if not self.validation_enabled or self.semantic_anchors is None:
            return None

        batches = []
        for x, y in self.load_train_data(drop_last=False, phase="validation"):
            if isinstance(x, (list, tuple)):
                x = x[0]
            batches.append((x, y.long().view(-1)))
            if len(batches) >= (3 if self.utility_validation_probe else 2):
                break
        if not batches:
            return None

        if self.utility_validation_probe and len(batches) < 3:
            # Keep the validation target independent from the utility query;
            # do not silently reuse the same batch on small clients.
            return None

        # The probe uses a third batch so the validation query is disjoint
        # from the utility query; legacy validation keeps its prior fallback.
        qx, qy = batches[2] if self.utility_validation_probe else batches[-1]
        qx, qy = self.move_input(qx, qy)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            query_feat = model.shared_representation(qx).detach().float()

        if self.stable_semantic_anchors is not None:
            validation = self.stable_semantic_anchors[: self.num_classes].float().detach().clone()
            observation = self.semantic_anchors[: self.num_classes].float().detach()
        elif self.utility_validation_probe and self.protos:
            # Single-anchor diagnostic: compare the received global anchor
            # with the client's independently observed local prototype.
            validation = self.semantic_anchors[: self.num_classes].float().detach().clone()
            observation = validation.clone()
            for label, proto in self.protos.items():
                label = int(label)
                proto = torch.as_tensor(proto, device=observation.device, dtype=observation.dtype).view(-1)
                if 0 <= label < self.num_classes and proto.numel() == observation.size(1):
                    observation[label] = proto
        else:
            return None
        validation.requires_grad_(True)
        query_dist = torch.cdist(query_feat, validation, p=2).pow(2)
        old_per_sample = F.cross_entropy(-query_dist, qy, reduction="none")
        observation_dist = torch.cdist(query_feat, observation, p=2).pow(2)
        new_per_sample = F.cross_entropy(-observation_dist, qy, reduction="none")
        query_loss = old_per_sample.mean()
        observation_loss = new_per_sample.mean()

        gain = old_per_sample.detach() - new_per_sample.detach()
        gain_sum = torch.zeros(self.num_classes, device=query_feat.device)
        gain_count = torch.zeros(self.num_classes, device=query_feat.device)
        gain_sum.index_add_(0, qy, gain)
        gain_count.index_add_(0, qy, torch.ones_like(qy, dtype=gain_count.dtype))
        self.validation_anchor_gain = (
            gain_sum / gain_count.clamp_min(1.0)
        ).detach().cpu()
        self.validation_anchor_gain_count = gain_count.detach().cpu()

        if self.new_anchor_mode == "observation_validation":
            anchor_grad = torch.autograd.grad(
                query_loss, validation, allow_unused=False
            )[0]
            self.validation_anchor_grad = anchor_grad.detach().cpu()
        self.validation_anchor_before_loss = float(query_loss.detach().item())
        self.validation_anchor_observation_loss = float(observation_loss.detach().item())
        if was_training:
            model.train()
        return self.validation_anchor_grad

    def update_gap_stats(self):
        self.local_gap = self.nearest_gap_from_dict(self.protos)
        self.global_gap = self.nearest_gap_from_tensor(self.anchor_bank_for_mcl())

    def nearest_gap_from_dict(self, proto_dict):
        if not proto_dict:
            return 0.0
        values = [
            torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1)
            for proto in proto_dict.values()
            if proto is not None and not isinstance(proto, list)
        ]
        if len(values) < 2:
            return 0.0
        return self.nearest_gap_from_tensor(torch.stack(values, dim=0))

    def nearest_gap_from_tensor(self, anchors):
        if anchors is None or anchors.ndim != 2 or anchors.size(0) < 2:
            return 0.0
        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return nearest_gap(anchors,audit=getattr(self,"distance_numerics",None))

    def release_model_from_device(self, model=None):
        if model is None:
            model = getattr(self, "model", None)
        if model is None:
            return
        if self.keep_client_models_on_gpu and self.device.type == "cuda":
            # Idle clients keep their weights, but not the last batch's gradients.
            model.zero_grad(set_to_none=True)
            return
        super().release_model_from_device(model)

    def train(self):
        start_time = time.time()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, "model", self.save_folder_name)
            self.model = model

        model.to(self.device)
        from utils.isolated_dropout import prepare_dropout
        prepare_dropout(model, self.random_seed, self.id, self.current_round)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        metric_sums = defaultdict(float)
        metric_batches = 0

        self.update_gap_stats()
        local_epochs = self.local_epochs
        if self.train_slow and local_epochs > 1:
            local_epochs = int(self.rng.randint(1, max(2, local_epochs // 2 + 1)))

        try:
            for epoch in range(local_epochs):
                for x, y in self.load_train_data(drop_last=True, phase=f"train/{epoch}"):
                    x, y = self.move_input(x, y)
                    if self.train_slow:
                        time.sleep(0.1 * abs(self.rng.rand()))

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep, logits = self.forward_rep_logits(model, x)
                        task_loss = self.loss(logits, y)
                        # PCW judges the actual classifier prediction, including its bias.
                        anchor_loss, anchor_metrics = self.anchor_loss(rep, y, logits.detach())
                        loss = task_loss + anchor_loss

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"FedDSA nonfinite training loss: client={self.id}, "
                            f"round={self.current_round}; refusing to skip corrupted training"
                        )

                    optimizer.zero_grad(set_to_none=True)
                    if self.amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        metric_sums["training_classifier_acc"] += float(logits.detach().argmax(dim=1).eq(y).float().mean())
                        metric_sums["task_loss"] += float(task_loss.detach().float().item())
                        metric_sums["anchor_loss"] += float(anchor_loss.detach().float().item())
                        metric_sums["total_loss"] += float(loss.detach().float().item())
                        for key, value in anchor_metrics.items():
                            metric_sums[key] += float(value.detach().float().item())
                        metric_batches += 1

            self.collect_protos(model=model)
            if self.prototype_utility or getattr(self, "candidate_gain_enabled", False):
                utility = self.compute_anchor_utility(model)
            if self.meta_enabled:
                meta_grad = self.compute_meta_anchor_gradient(model)
                if meta_grad is not None:
                    metric_sums["meta_grad_norm"] = float(meta_grad.float().norm().item())
            if self.validation_enabled:
                validation_grad = self.compute_validation_anchor_gradient(model)
                if validation_grad is not None:
                    metric_sums["validation_grad_norm"] = float(
                        validation_grad.float().norm().item()
                    )
                    metric_sums["validation_before_loss"] = self.validation_anchor_before_loss
                    metric_sums["validation_observation_loss"] = (
                        self.validation_anchor_observation_loss
                    )
            denom = max(metric_batches, 1)
            self.latest_train_metrics = {key: value / denom for key, value in metric_sums.items()}
            self.latest_train_metrics.update(self.latest_utility_metrics)
            save_item(self.protos, self.role, "protos", self.save_folder_name)
            save_item(model, self.role, "model", self.save_folder_name)
        finally:
            self.release_model_from_device(model)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def anchor_loss(self, feats, y, logits):
        zero = feats.new_zeros(())
        metrics = {
            "anchor_mse_loss": zero,
            "mcl_loss": zero,
            "mcl_tail_frac": zero,
            "anchor_agree_head_acc": zero,
            "anchor_agree_both_wrong": zero,
            "dual_disagreement_rate": zero,
            "dual_uncertainty_mean": zero,
            "utility_weight_mean": zero,
            "utility_weight_std": zero,
            "boundary_loss": zero,
            "boundary_active_rate": zero,
            "boundary_margin": zero,
            "class_boundary_loss": zero,
            "class_boundary_active_rate": zero,
        }
        if self.semantic_anchors is None:
            return zero, metrics

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
            return zero, metrics

        y = y.long().view(-1)
        adaptive_anchors = None
        disagreement_mask = None
        disagreement_weight = None
        boundary_anchors = None
        if (
            self.new_anchor_mode == "boundary"
            and self.anchor_mode == "dual"
            and self.adaptive_anchors is not None
        ):
            boundary_anchors = self.adaptive_anchors[: self.num_classes].to(
                device=feats.device, dtype=feats.dtype
            )
            if self.stable_semantic_anchors is not None:
                mse_anchors = self.stable_semantic_anchors[: self.num_classes]
                mcl_anchors = self.stable_semantic_anchors[: self.num_classes]
        if (
            self.dual_disagreement_gate
            and self.anchor_mode == "dual"
            and self.new_anchor_mode == "none"
            and self.stable_semantic_anchors is not None
        ):
            stable = self.stable_semantic_anchors[: self.num_classes].to(
                device=feats.device, dtype=feats.dtype
            )
            adaptive_anchors = self.semantic_anchors[: self.num_classes].to(
                device=feats.device, dtype=feats.dtype
            )
            with torch.no_grad():
                stable_dist = torch.cdist(
                    feats.detach().float(), stable.float(), p=2
                )
                adaptive_dist = torch.cdist(
                    feats.detach().float(), adaptive_anchors.float(), p=2
                )
                row = torch.arange(y.numel(), device=feats.device)
                one_hot = F.one_hot(y, self.num_classes).bool()
                stable_same = stable_dist[row, y]
                adaptive_same = adaptive_dist[row, y]
                stable_neg = stable_dist.masked_fill(one_hot, float("inf")).min(dim=1).values
                adaptive_neg = adaptive_dist.masked_fill(one_hot, float("inf")).min(dim=1).values
                stable_margin = stable_neg - stable_same
                adaptive_margin = adaptive_neg - adaptive_same
                margin_delta = (stable_margin - adaptive_margin).abs()
                disagreement_weight = (
                    margin_delta
                    / (margin_delta + float(self.dual_disagreement_temperature))
                ).clamp(0.0, 1.0)
                stable_pred = stable_dist.argmin(dim=1)
                adaptive_pred = adaptive_dist.argmin(dim=1)
                disagreement_mask = stable_pred.ne(adaptive_pred)
            # Keep the semantic regression target stationary.  The adaptive
            # state is reserved for samples where the two states disagree.
            mse_anchors = stable
            mcl_anchors = stable
            metrics["dual_disagreement_rate"] = disagreement_mask.float().mean()
            metrics["dual_uncertainty_mean"] = disagreement_weight.mean()
        if (
            self.anchor_mode == "dual"
            and not self.separate_anchor_roles
            and self.new_anchor_mode in {
                "relation_memory",
                "semantic_discriminative",
                "observation_validation",
                "meta_separated",
            }
        ):
            # Shared-target control: keep the adaptive/discriminative target
            # for both losses while preserving the same server update path.
            mse_anchors = mcl_anchors
        mse_anchors = mse_anchors.to(device=feats.device, dtype=feats.dtype)[: self.num_classes]
        mcl_anchors = mcl_anchors.to(device=feats.device, dtype=feats.dtype)[: self.num_classes]

        reliability_anchors = mcl_anchors
        if (
            self.reliability_anchor_source == "semantic"
            and self.anchor_mode == "dual"
            and self.separate_anchor_roles
            and self.new_anchor_mode in {
                "relation_memory",
                "semantic_discriminative",
                "observation_validation",
                "meta_separated",
            }
        ):
            reliability_anchors = mse_anchors
        sample_weight, agree_metrics = self.anchor_agreement_weight(
            feats, y, reliability_anchors, logits
        )
        if (
            self.scalar_reliability_state
            and self.class_reliability is not None
            and self.class_reliability.numel() >= self.num_classes
        ):
            state = self.class_reliability[: self.num_classes].to(
                device=feats.device, dtype=sample_weight.dtype
            ).clamp_min(0.25)
            state_weight = (state / state.mean().clamp_min(1e-6)).clamp(0.5, 1.5)
            sample_weight = sample_weight * state_weight.index_select(0, y)
        if (
            self.prototype_utility
            and self.utility_local_gate
            and self.class_utility is not None
            and self.class_utility.numel() >= self.num_classes
        ):
            utility = self.class_utility[: self.num_classes].to(
                device=feats.device, dtype=sample_weight.dtype
            )
            if self.utility_positive_only:
                signal = utility.clamp_min(0.0)
                utility_weight = signal / (
                    signal + float(self.utility_temperature)
                )
                utility_weight = self.utility_min_weight + (
                    self.utility_max_weight - self.utility_min_weight
                ) * utility_weight
            else:
                utility_weight = torch.sigmoid(
                    utility / float(self.utility_temperature)
                )
                # Scale before clipping, including when mean normalization follows.
                utility_weight = 2.0 * utility_weight
            capture_gate = getattr(self, "capture_utility_gate", False)
            if capture_gate:
                before_clip = utility_weight.detach().clone()
            utility_weight = utility_weight.clamp(
                self.utility_min_weight, self.utility_max_weight
            )
            if capture_gate:
                before_normalization = utility_weight.detach().clone()
                normalized_before_clip = (
                    before_normalization
                    / before_normalization.mean().clamp_min(1e-6)
                    if self.utility_weight_normalize
                    else None
                )
            if self.utility_weight_normalize:
                utility_weight = (
                    utility_weight / utility_weight.mean().clamp_min(1e-6)
                ).clamp(self.utility_min_weight, self.utility_max_weight)
            if capture_gate:
                self.last_local_utility_gate = {
                    "utility_before_training": utility.detach().clone(),
                    "before_clip": before_clip,
                    "before_normalization": before_normalization,
                    "normalized_before_clip": normalized_before_clip,
                    "weight": utility_weight.detach().clone(),
                }
                self.capture_utility_gate = False
            sample_weight = sample_weight * utility_weight.index_select(0, y)
            metrics["utility_weight_mean"] = utility_weight.mean()
            metrics["utility_weight_std"] = utility_weight.std(unbiased=False)
        metrics.update(agree_metrics)
        metrics["final_anchor_weight_mean"] = sample_weight.detach().float().mean()

        mse = self.anchor_mse(feats, y, mse_anchors, sample_weight)
        mcl, mcl_metrics = self.mcl_loss(
            feats,
            y,
            mcl_anchors,
            sample_weight,
            alternate_anchors=adaptive_anchors,
            alternate_mask=disagreement_mask,
            alternate_weight=disagreement_weight,
        )
        loss = mse + mcl
        metrics["anchor_mse_loss"] = mse
        metrics.update(mcl_metrics)
        metrics["mcl_loss"] = mcl
        if boundary_anchors is not None and self.boundary_loss_weight > 0.0:
            boundary, boundary_metrics = self.boundary_loss(
                feats,
                y,
                mcl_anchors,
                boundary_anchors,
                sample_weight,
            )
            boundary = boundary * self.boundary_loss_weight
            loss = loss + boundary
            metrics.update(boundary_metrics)
            metrics["boundary_loss"] = boundary
        if (
            self.new_anchor_mode == "class_boundary"
            and self.class_boundary_margin is not None
            and self.boundary_loss_weight > 0.0
        ):
            class_boundary, class_boundary_metrics = self.class_boundary_loss(
                feats,
                y,
                mcl_anchors,
                self.class_boundary_margin[: self.num_classes].to(
                    device=feats.device, dtype=feats.dtype
                ),
                sample_weight,
            )
            class_boundary = class_boundary * self.boundary_loss_weight
            loss = loss + class_boundary
            metrics.update(class_boundary_metrics)
            metrics["class_boundary_loss"] = class_boundary
        return loss, metrics

    def anchor_agreement_weight(self, feats, y, anchors, logits):
        with torch.no_grad():
            dist = torch.cdist(feats.detach().float(), anchors.float(), p=2)
            anchor_correct = dist.argmin(dim=1).eq(y)
            head_correct = logits.detach().argmax(dim=1).eq(y)
            both_wrong = (~head_correct) & (~anchor_correct)

            anchor_acc = anchor_correct.float().mean()
            head_acc = head_correct.float().mean()
            both_wrong_rate = both_wrong.float().mean()
            reliability = (anchor_acc / head_acc.clamp_min(1e-6)).clamp(0.0, 1.0)
            recovery_floor = torch.sqrt((head_acc * (1.0 - both_wrong_rate)).clamp(0.0, 1.0))
            recovery_gap = (recovery_floor - reliability).clamp_min(0.0)
            recovery_room = (1.0 - reliability).clamp_min(1e-6)
            blend = ((recovery_gap / recovery_room) * recovery_floor).clamp(0.0, 1.0)
            reliability = (reliability + blend * (recovery_floor - reliability)).clamp(0.0, 1.0)

            weight = torch.full_like(y, float(reliability.item()), dtype=torch.float32, device=feats.device)
            weight = torch.where(both_wrong, torch.sqrt(reliability).expand_as(weight), weight)

            if not self.agreement_weighting:
                weight = torch.ones_like(weight)

        metrics = {
            "anchor_predictor_acc": anchor_correct.float().mean(),
            "pcw_batch_R": reliability.detach(),
            "pcw_weight_mean": weight.detach().float().mean(),
            "anchor_agree_head_acc": head_correct.float().mean(),
            "anchor_agree_both_wrong": both_wrong.float().mean(),
        }
        return weight.detach(), metrics

    def anchor_mse(self, feats, y, anchors, sample_weight):
        target = anchors.index_select(0, y).float()
        per_sample = F.mse_loss(feats.float(), target, reduction="none").mean(dim=1)
        weights = sample_weight.to(device=feats.device, dtype=per_sample.dtype).clamp_min(0.0)
        return (per_sample * weights).mean()

    def boundary_loss(self, feats, y, stable_anchors, boundary_anchors, sample_weight):
        positive = stable_anchors.index_select(0, y).float()
        rival = boundary_anchors.index_select(0, y).float()
        positive_dist = (feats.float() - positive).norm(dim=1)
        rival_dist = (feats.float() - rival).norm(dim=1)
        margin = rival_dist - positive_dist
        per_sample = F.relu(float(self.mcl_hard_margin) - margin)
        weights = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).clamp_min(0.0)
        return (per_sample * weights).mean(), {
            "boundary_active_rate": (
                margin.detach() < float(self.mcl_hard_margin)
            ).float().mean(),
            "boundary_margin": margin.detach().float().mean(),
        }

    def class_boundary_loss(
        self, feats, y, anchors, class_margin, sample_weight
    ):
        dist = torch.cdist(feats.float(), anchors.float(), p=2)
        row = torch.arange(dist.size(0), device=dist.device)
        same = dist[row, y]
        neg = dist.masked_fill(
            F.one_hot(y, self.num_classes).bool(), float("inf")
        ).min(dim=1).values
        sample_margin = neg - same
        target_margin = class_margin.index_select(0, y).clamp_min(0.0)
        per_sample = F.relu(target_margin - sample_margin)
        weights = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).clamp_min(0.0)
        return (per_sample * weights).mean(), {
            "class_boundary_active_rate": (
                sample_margin.detach() < target_margin.detach()
            ).float().mean(),
        }

    def mcl_loss(
        self,
        feats,
        y,
        anchors,
        sample_weight,
        alternate_anchors=None,
        alternate_mask=None,
        alternate_weight=None,
    ):
        if getattr(self, "mcl_gap_class_cap", False) and alternate_anchors is not None:
            raise ValueError('Class-capped MCL gap requires one actual anchor bank;alternate-bank mixing is unsupported')
        if getattr(self, "mcl_gap_class_cap", False) and self.mcl_variant != "fedsa":
            raise ValueError('Class-capped MCL gap currently requires the FedSA distance-softmax objective')
        distance_anchors = anchors.float()
        dist = guarded_cdist(feats.float(), distance_anchors,audit=getattr(self,"distance_numerics",None),context="local_mcl")
        if alternate_anchors is not None and alternate_weight is not None:
            alternate_dist = torch.cdist(
                feats.float(), alternate_anchors.float(), p=2
            )
            blend = alternate_weight.to(device=dist.device, dtype=dist.dtype).view(-1, 1)
            dist = dist + blend * (alternate_dist - dist)
        elif alternate_anchors is not None and alternate_mask is not None:
            alternate_dist = torch.cdist(
                feats.float(), alternate_anchors.float(), p=2
            )
            dist = torch.where(alternate_mask.view(-1, 1), alternate_dist, dist)
        row = torch.arange(dist.size(0), device=dist.device)
        gap = self.mcl_gap_value()
        dist = add_mcl_gap(dist, y, distance_anchors, gap,
            class_cap=getattr(self, "mcl_gap_class_cap", False))

        same = dist[row, y]
        neg = dist.masked_fill(
            F.one_hot(y, self.num_classes).to(device=dist.device, dtype=torch.bool),
            float("inf"),
        ).min(dim=1).values
        margin = neg - same
        if self.mcl_variant == "hard_negative":
            # Boundary-focused variant: only examples whose closest
            # competing anchor is within the requested margin contribute.
            # This gives the adaptive anchor a direct class-boundary signal;
            # the default remains the FedSA-compatible distance softmax.
            per_sample = F.relu(
                float(self.mcl_hard_margin) - margin
            )
        else:
            per_sample = F.cross_entropy(-dist, y, reduction="none")
        weights = sample_weight.to(device=dist.device, dtype=per_sample.dtype).clamp_min(0.0)
        metrics = {
            "mcl_tail_frac": (margin.detach() < 0).float().mean(),
            "mcl_sample_margin": margin.detach().float().mean(),
        }
        if getattr(self, "mcl_gap_class_cap", False):
            metrics.update(sample_gap_metrics(distance_anchors, gap, y, class_cap=True))
        return (per_sample * weights).mean(), metrics

    def mcl_gap_value(self):
        local_gap = float(getattr(self, "local_gap", 0.0))
        global_gap = float(getattr(self, "global_gap", 0.0))
        if local_gap <= 0 or global_gap <= 0:
            return 0.0
        if getattr(self, "mcl_gap_mode", "min") == "max":
            return max(local_gap, global_gap)
        return min(local_gap, global_gap)

    def test_metrics(self):
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

        try:
            with torch.no_grad():
                for x, y in self.load_test_data():
                    x, y = self.move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        _, logits = self.forward_rep_logits(model, x)
                    test_acc += logits.argmax(dim=1).eq(y).sum().item()
                    test_num += y.size(0)
                    y_prob.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                    nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                    labels = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        labels = labels[:, :2]
                    y_true.append(labels)
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
        return test_acc, test_num, auc

    def train_metrics(self):
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
                for x, y in self.load_train_data(drop_last=False, phase="metrics"):
                    x, y = self.move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        _, logits = self.forward_rep_logits(model, x)
                        loss = self.loss(logits, y)
                    train_num += y.size(0)
                    losses += loss.item() * y.size(0)
        finally:
            self.release_model_from_device(model)

        return losses, train_num
