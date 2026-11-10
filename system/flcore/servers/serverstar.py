import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientstar import clientstar
from flcore.servers.serverbase import Server


class FedSTAR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientstar)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.num_classes = int(args.num_classes)
        self.feat_channel = int(args.feature_dim)
        self.anchor_momentum = 0.0
        self.stable_anchor_mode = str(args.stable_anchor_mode).lower()
        if self.stable_anchor_mode not in {"single", "dual"}:
            self.stable_anchor_mode = "dual"
        self.anchor_momentum = float(getattr(args, "anchor_momentum", 0.9999))
        self.anchor_momentum = min(max(self.anchor_momentum, 0.0), 1.0)
        self.stable_anchor_source = str(getattr(args, "stable_anchor_source", "prototype")).lower()
        if self.stable_anchor_source not in {"prototype", "guarded_fast"}:
            self.stable_anchor_source = "prototype"
        self.fast_anchor_bound = str(getattr(args, "fast_anchor_bound", "off")).lower()
        if self.fast_anchor_bound not in {"off", "trust_region"}:
            self.fast_anchor_bound = "off"
        self.fast_anchor_radius_rho = float(getattr(args, "fast_anchor_radius_rho", 0.5))
        if self.fast_anchor_radius_rho < 0.0:
            self.fast_anchor_radius_rho = 0.0
        self.fast_anchor_trust_norm = str(getattr(args, "fast_anchor_trust_norm", "off")).lower()
        if self.fast_anchor_trust_norm not in {"off", "radius"}:
            self.fast_anchor_trust_norm = "off"
        self.fast_anchor_proto_trust = str(getattr(args, "fast_anchor_proto_trust", "auto")).lower()
        if self.fast_anchor_proto_trust not in {"off", "auto"}:
            self.fast_anchor_proto_trust = "auto"
        self.fast_anchor_semantic_gate = str(getattr(args, "fast_anchor_semantic_gate", "off")).lower()
        if self.fast_anchor_semantic_gate not in {"off", "margin"}:
            self.fast_anchor_semantic_gate = "off"
        self.collapse_guard_mode = "off"
        self.anchor_agreement_gate = str(getattr(args, "anchor_agreement_gate", "off")).lower()
        if self.anchor_agreement_gate not in {"off", "auto"}:
            self.anchor_agreement_gate = "off"
        self.anchor_spectrum_ref = None

        print("=" * 50)
        print("FedSTAR config")
        print(
            f"feat_dim={self.feat_channel} "
            f"anchor_mse={float(args.lamda):.4f} "
            f"mcl={float(args.margin_contrastive):.4f} "
            "classifier=cosine"
        )
        print(
            f"stable={self.stable_anchor_mode}/{self.anchor_momentum:.4f}/{self.stable_anchor_source} "
            f"fast_bound={self.fast_anchor_bound} rho={self.fast_anchor_radius_rho_label()} "
            f"trust={self.fast_anchor_proto_trust}/{self.fast_anchor_trust_norm}/{self.fast_anchor_semantic_gate}"
        )
        print(
            f"anchor_agreement_gate={self.anchor_agreement_gate}"
        )
        print("=" * 50)

        self.Budget = []
        self.global_protos = {}
        self.uploaded_proto_groups = {}
        self.proto_disagreement = {}
        self.proto_support = {}
        self.previous_anchor_eval_summary = None
        self.last_anchor_geometry = 1.0
        self.last_anchor_norm = 0.0
        self.last_anchor_pair_mean = 0.0
        self.last_anchor_pair_min = 0.0
        self.last_anchor_effective_rank = 0.0
        self.last_stable_anchor_norm = 0.0
        self.last_stable_anchor_pair_mean = 0.0
        self.last_stable_anchor_pair_min = 0.0
        self.last_stable_anchor_effective_rank = 0.0
        self.last_collapse_guard_delta = 0.0
        self.last_collapse_guard_weight = 0.0
        self.last_collapse_guard_min_ratio = 0.0
        self.last_collapse_guard_geometry_ratio = 1.0
        self.last_collapse_guard_spectrum_error = 0.0
        self.last_fast_anchor_bound_radius = 0.0
        self.last_fast_anchor_radius_rho = 0.0
        self.last_fast_anchor_bound_delta = 0.0
        self.last_fast_anchor_bound_clip_frac = 0.0
        self.last_fast_anchor_trust_beta = 0.0
        self.last_fast_anchor_trust_lag = 0.0
        self.last_fast_anchor_trust_signal_value = 0.0
        self.last_fast_anchor_trust_noise = 0.0
        self.last_fast_anchor_trust_support = 0.0
        self.last_fast_anchor_effective_beta = 0.0
        self.last_fast_anchor_semantic_gate = 0.0
        self.last_fast_anchor_semantic_slow_margin = 0.0
        self.last_fast_anchor_semantic_fast_margin = 0.0

        norm_cfg = {"type": "SyncBN", "requires_grad": True}
        self.anchor_projector = nn.Sequential(
            nn.Linear(self.feat_channel, self.feat_channel),
            build_norm_layer(norm_cfg, self.feat_channel)[1],
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channel, self.feat_channel),
            build_norm_layer(norm_cfg, self.feat_channel)[1],
            nn.ReLU(inplace=True),
        ).to(self.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + 3407)
        random_center = torch.randn(self.num_classes, self.feat_channel, generator=generator)
        with torch.no_grad():
            self.semantic_anchors = self.anchor_projector(random_center.to(self.device)).detach()
            self.semantic_anchor_initialized = torch.zeros(
                self.num_classes,
                device=self.device,
                dtype=torch.bool,
            )
            self.stable_semantic_anchors = (
                self.semantic_anchors.detach().clone() if self.stable_anchor_mode == "dual" else None
            )
            self.anchor_spectrum_ref = self.anchor_singular_values(self.semantic_anchors).detach()

        for client in self.clients:
            client.global_protos = self.global_protos

    def train(self):
        self.send_semantic_state()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            self.send_semantic_state(self.selected_clients)
            results = self.run_clients(self.selected_clients, return_exceptions=True)
            for client, result in results:
                if isinstance(result, Exception):
                    raise RuntimeError(f"FedSTAR client {client.id} failed during local training.") from result

            self.receive_protos()
            self.update_semantic_anchors()
            self.apply_fast_anchor_bound()
            self.update_stable_anchors_from_fast()
            self.update_anchor_geometry_metric()
            self.record_anchor_3d_snapshot(i, self.semantic_anchors)
            self.send_semantic_state()
            self.log_train_diagnostics(i)

            self.Budget.append(time.time() - s_t)
            if i % self.eval_gap == 0 or i == self.global_rounds:
                print("-" * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.maybe_save_anchor_3d()
        self.maybe_save_tsne()
        self.save_results()

    def send_semantic_state(self, clients=None):
        clients = self.clients if clients is None else clients
        anchors = self.semantic_anchors.detach()
        stable_anchors = None
        if self.stable_anchor_mode == "dual" and self.stable_semantic_anchors is not None:
            stable_anchors = self.stable_semantic_anchors.detach()
        for client in clients:
            client.set_semantic_anchors(anchors)
            client.set_stable_semantic_anchors(stable_anchors)

    def log_train_diagnostics(self, round_num):
        if round_num % self.eval_gap != 0 and round_num != self.global_rounds:
            return

        metric_sums = defaultdict(float)
        total_weight = 0.0

        for client in self.selected_clients:
            metrics = getattr(client, "latest_train_metrics", None)
            if not metrics:
                continue
            weight = float(max(getattr(client, "train_samples", 1), 1))
            total_weight += weight
            for key, value in metrics.items():
                metric_sums[key] += weight * float(value)

        if total_weight <= 0:
            return

        summary = {key: value / total_weight for key, value in metric_sums.items()}
        proto_alignment = self.compute_proto_anchor_alignment(self.semantic_anchors)
        fields = [
            f"task={summary.get('task_loss', 0.0):.4f}",
            f"mse={summary.get('anchor_mse_loss', 0.0):.4f}",
            f"mcl={summary.get('mcl_loss', 0.0):.4f}",
            f"total={summary.get('total_loss', 0.0):.4f}",
            f"proto_nn={proto_alignment['nearest_acc']:.4f}",
            f"mcl_tail_frac={summary.get('mcl_tail_frac', 0.0):.4f}",
            f"rank={self.last_anchor_effective_rank:.2f}",
            f"stable_rank={self.last_stable_anchor_effective_rank:.2f}",
            f"vol_ratio={self.last_collapse_guard_geometry_ratio:.4f}",
            f"spec_min={self.last_collapse_guard_min_ratio:.4f}",
        ]

        if self.anchor_agreement_gate == "auto":
            fields.extend(
                [
                    f"agree_h={summary.get('anchor_agree_head_acc', 0.0):.4f}",
                    f"agree_bw={summary.get('anchor_agree_both_wrong', 0.0):.4f}",
                    f"agree_floor={summary.get('anchor_agree_recovery_floor', 1.0):.4f}",
                ]
            )
        print(f"[Round {round_num}] FedSTARTrain " + " ".join(fields))

    def short_model_key(self, model_key):
        text = str(model_key)
        lower = text.lower()
        for name in (
            "googlenet",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "mobilenet_v2",
            "alexnet",
            "vgg",
            "vit",
        ):
            if name in lower:
                return name
        if "(" in text:
            return text.split("(", 1)[0].split(".")[-1]
        return text[:40]

    def test_metrics(self):
        self.send_semantic_state()

        ids = []
        num_samples = []
        total_correct = []
        total_auc = []
        anchor_eval_samples = 0.0
        anchor_same_dist_sum = 0.0
        anchor_neg_dist_sum = 0.0
        anchor_margin_sum = 0.0
        anchor_nn_correct_sum = 0.0

        for client in self.clients:
            correct, samples, auc = client.test_metrics()
            ids.append(client.id)
            num_samples.append(samples)
            total_correct.append(float(correct))
            total_auc.append(float(auc) * samples)

            eval_metrics = getattr(client, "latest_eval_metrics", {})
            anchor_samples = float(eval_metrics.get("anchor_eval_samples", 0.0))
            if anchor_samples > 0:
                anchor_same_dist_sum += anchor_samples * float(eval_metrics.get("anchor_same_dist", 0.0))
                anchor_neg_dist_sum += anchor_samples * float(eval_metrics.get("anchor_neg_dist", 0.0))
                anchor_margin_sum += anchor_samples * float(eval_metrics.get("anchor_margin", 0.0))
                anchor_nn_correct_sum += anchor_samples * float(eval_metrics.get("nearest_anchor_acc", 0.0))
                anchor_eval_samples += anchor_samples

        if anchor_eval_samples > 0:
            summary = {
                "same_dist": anchor_same_dist_sum / anchor_eval_samples,
                "neg_dist": anchor_neg_dist_sum / anchor_eval_samples,
                "margin": anchor_margin_sum / anchor_eval_samples,
                "nearest_anchor_acc": anchor_nn_correct_sum / anchor_eval_samples,
            }
            prev = self.previous_anchor_eval_summary
            if prev is None:
                delta_msg = "delta=n/a"
            else:
                delta_msg = (
                    f"d_margin={summary['margin'] - prev['margin']:+.4f} "
                    f"d_nn={summary['nearest_anchor_acc'] - prev['nearest_anchor_acc']:+.4f}"
                )
            print(
                f"FedSTARAnchorEval "
                f"margin={summary['margin']:.4f} "
                f"nearest_acc={summary['nearest_anchor_acc']:.4f} | {delta_msg}"
            )
            self.previous_anchor_eval_summary = summary

        return ids, num_samples, total_correct, total_auc

    def receive_protos(self):
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        uploaded = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = getattr(client, "protos", None)
            if protos is None:
                protos = load_item(client.role, "protos", client.save_folder_name)
            if protos:
                uploaded.append(protos)

        self.global_protos, self.uploaded_proto_groups = self.aggregate_uploaded_protos(uploaded)
        for client in self.clients:
            client.global_protos = self.global_protos
        save_item(self.global_protos, self.role, "global_protos", self.save_folder_name)

    def aggregate_uploaded_protos(self, uploaded):
        proto_groups = defaultdict(list)

        for item in uploaded:
            protos = item
            if not protos:
                continue
            for label, proto in protos.items():
                if proto is None or isinstance(proto, list):
                    continue
                label = int(label)
                proto = torch.as_tensor(proto, dtype=torch.float32).detach().cpu().view(-1)
                if proto.numel() != self.feat_channel:
                    continue
                proto_groups[label].append(proto)

        self.proto_disagreement = {}
        self.proto_support = {}

        global_protos = {}
        proto_disagreement = {}
        proto_support = {}

        for label, proto_list in proto_groups.items():
            if len(proto_list) == 0:
                continue
            proto_tensor = torch.stack(proto_list, dim=0)
            global_proto = proto_tensor.mean(dim=0)
            global_protos[int(label)] = global_proto
            disagreement = (proto_tensor - global_proto.detach().cpu().view(1, -1)).pow(2).sum(dim=1)
            proto_disagreement[int(label)] = float(disagreement.mean().item())
            proto_support[int(label)] = float(len(proto_list))

        self.proto_disagreement = proto_disagreement
        self.proto_support = proto_support

        return global_protos, dict(proto_groups)

    def proto2anchors(self, global_protos):
        if not global_protos:
            return torch.empty(0, self.feat_channel, device=self.device)

        anchors = []
        for label in sorted(global_protos.keys()):
            proto = global_protos[label]
            if proto is None or isinstance(proto, list):
                continue
            anchors.append(torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1))

        if len(anchors) == 0:
            return torch.empty(0, self.feat_channel, device=self.device)
        return torch.stack(anchors, dim=0)

    @torch.no_grad()
    def update_semantic_anchors(self):
        if not self.global_protos:
            return

        stable_proto_weight = 1.0 - self.anchor_momentum
        dual_labels = []
        dual_protos = []
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= self.semantic_anchors.size(0):
                continue
            proto = torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1)
            if proto.numel() != self.semantic_anchors.size(1):
                continue
            if self.stable_anchor_mode == "single":
                self.semantic_anchors[label].mul_(self.anchor_momentum).add_(
                    proto,
                    alpha=stable_proto_weight,
                )
                self.semantic_anchor_initialized[label] = True
                continue

            dual_labels.append(label)
            dual_protos.append(proto)
            if self.stable_semantic_anchors is not None and self.stable_anchor_source == "prototype":
                self.stable_semantic_anchors[label].mul_(self.anchor_momentum).add_(
                    proto,
                    alpha=stable_proto_weight,
                )

        if self.stable_anchor_mode != "dual" or len(dual_labels) == 0:
            return

        labels = torch.as_tensor(dual_labels, device=self.device, dtype=torch.long)
        protos = torch.stack(dual_protos, dim=0).to(device=self.device, dtype=torch.float32)
        self.semantic_anchors[labels] = protos.to(dtype=self.semantic_anchors.dtype)
        self.semantic_anchor_initialized[labels] = True

    @torch.no_grad()
    def update_stable_anchors_from_fast(self):
        if (
            self.stable_anchor_source != "guarded_fast"
            or self.stable_semantic_anchors is None
            or self.semantic_anchors is None
        ):
            return

        if self.stable_semantic_anchors.shape != self.semantic_anchors.shape:
            self.stable_semantic_anchors = self.semantic_anchors.detach().clone()
            return

        fast_anchors = self.semantic_anchors.detach().to(
            device=self.stable_semantic_anchors.device,
            dtype=self.stable_semantic_anchors.dtype,
        )
        stable_weight = 1.0 - self.anchor_momentum
        self.stable_semantic_anchors.mul_(self.anchor_momentum).add_(
            fast_anchors,
            alpha=stable_weight,
        )

    @torch.no_grad()
    def resolve_fast_anchor_bound_radius(self):
        radii = self.resolve_fast_anchor_bound_radii()
        if radii.numel() == 0:
            return 0.0
        radii = radii[torch.isfinite(radii) & (radii > 0)]
        if radii.numel() == 0:
            return 0.0
        radius = torch.median(radii).clamp_min(0.0)
        if not torch.isfinite(radius) or radius <= 0:
            return 0.0
        return float(radius.item())

    def fast_anchor_radius_rho_label(self):
        return f"auto({self.current_fast_anchor_radius_rho():.4g}*nn)"

    def collapse_guard_mode_label(self):
        if self.collapse_guard_mode == "auto":
            return "auto(1-min_spectrum_ratio)"
        return "off"

    def current_fast_anchor_radius_rho(self):
        return float(self.fast_anchor_radius_rho)

    @torch.no_grad()
    def resolve_fast_anchor_nearest_distances(self, slow_anchors=None):
        if slow_anchors is None:
            slow_anchors = self.stable_semantic_anchors
        if slow_anchors is None or slow_anchors.ndim != 2 or slow_anchors.size(0) < 2:
            return torch.empty(0, device=self.device, dtype=torch.float32)

        slow_anchors = torch.nan_to_num(
            slow_anchors.float().to(device=self.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        dist = torch.cdist(slow_anchors, slow_anchors, p=2)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.min(dim=1).values
        return torch.where(torch.isfinite(nearest), nearest.clamp_min(0.0), torch.zeros_like(nearest))

    @torch.no_grad()
    def resolve_fast_anchor_bound_radii(self, slow_anchors=None):
        rho = self.current_fast_anchor_radius_rho()
        if rho <= 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)

        if slow_anchors is None:
            slow_anchors = self.stable_semantic_anchors
        if slow_anchors is None or slow_anchors.ndim != 2 or slow_anchors.size(0) < 2:
            return torch.empty(0, device=self.device, dtype=torch.float32)

        nearest = self.resolve_fast_anchor_nearest_distances(slow_anchors)
        return nearest * float(rho)


    @torch.no_grad()
    def compute_proto_anchor_alignment(self, anchors):
        empty = {
            "same_dist": 0.0,
            "neg_dist": 0.0,
            "margin": 0.0,
            "nearest_acc": 0.0,
        }
        anchors = torch.as_tensor(anchors, device=self.device, dtype=torch.float32)
        if (
            not self.global_protos
            or anchors.ndim != 2
            or anchors.size(0) < 2
            or anchors.size(1) != self.feat_channel
        ):
            return empty

        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        labels = []
        protos = []
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= anchors.size(0) or proto is None or isinstance(proto, list):
                continue
            proto = torch.as_tensor(proto, device=anchors.device, dtype=torch.float32).view(-1)
            if proto.numel() != anchors.size(1):
                continue
            labels.append(label)
            protos.append(torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0))

        if not protos:
            return empty

        proto_matrix = torch.stack(protos, dim=0)
        label_tensor = torch.as_tensor(labels, device=anchors.device, dtype=torch.long)
        dist = torch.cdist(proto_matrix, anchors, p=2)
        row_idx = torch.arange(label_tensor.numel(), device=anchors.device)
        same = dist[row_idx, label_tensor]
        neg_dist = dist.clone()
        neg_dist[row_idx, label_tensor] = float("inf")
        nearest_neg = neg_dist.min(dim=1).values
        nearest = dist.min(dim=1).indices
        margin = nearest_neg - same
        finite = torch.isfinite(same) & torch.isfinite(nearest_neg) & torch.isfinite(margin)
        if not torch.any(finite):
            return empty

        return {
            "same_dist": float(same[finite].mean().item()),
            "neg_dist": float(nearest_neg[finite].mean().item()),
            "margin": float(margin[finite].mean().item()),
            "nearest_acc": float((nearest[finite] == label_tensor[finite]).float().mean().item()),
        }

    @torch.no_grad()
    def resolve_fast_anchor_semantic_gate(self, slow_anchors, candidate_anchors, radii):
        gate = torch.ones(candidate_anchors.size(0), device=candidate_anchors.device, dtype=torch.float32)
        self.last_fast_anchor_semantic_gate = 0.0
        self.last_fast_anchor_semantic_slow_margin = 0.0
        self.last_fast_anchor_semantic_fast_margin = 0.0
        if self.fast_anchor_semantic_gate != "margin":
            return gate

        slow_anchors = torch.as_tensor(slow_anchors, device=candidate_anchors.device, dtype=torch.float32)
        candidate_anchors = torch.as_tensor(candidate_anchors, device=candidate_anchors.device, dtype=torch.float32)
        if (
            not self.global_protos
            or slow_anchors.ndim != 2
            or candidate_anchors.ndim != 2
            or slow_anchors.shape != candidate_anchors.shape
            or candidate_anchors.size(0) < 2
        ):
            return gate
        radii = torch.as_tensor(radii, device=candidate_anchors.device, dtype=torch.float32).view(-1)
        if radii.numel() != candidate_anchors.size(0):
            return gate

        labels = []
        protos = []
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= candidate_anchors.size(0) or proto is None or isinstance(proto, list):
                continue
            proto = torch.as_tensor(proto, device=candidate_anchors.device, dtype=torch.float32).view(-1)
            if proto.numel() != candidate_anchors.size(1):
                continue
            labels.append(label)
            protos.append(torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0))

        if not protos:
            return gate

        proto_matrix = torch.stack(protos, dim=0)
        label_tensor = torch.as_tensor(labels, device=candidate_anchors.device, dtype=torch.long)
        row_idx = torch.arange(label_tensor.numel(), device=candidate_anchors.device)
        slow_dist = torch.cdist(proto_matrix, slow_anchors, p=2).pow(2)
        fast_dist = torch.cdist(proto_matrix, candidate_anchors, p=2).pow(2)

        slow_same = slow_dist[row_idx, label_tensor]
        fast_same = fast_dist[row_idx, label_tensor]
        slow_neg = slow_dist.clone()
        fast_neg = fast_dist.clone()
        slow_neg[row_idx, label_tensor] = float("inf")
        fast_neg[row_idx, label_tensor] = float("inf")
        slow_margin = slow_neg.min(dim=1).values - slow_same
        fast_margin = fast_neg.min(dim=1).values - fast_same

        num_classes = candidate_anchors.size(0)
        valid = torch.zeros(num_classes, device=candidate_anchors.device, dtype=torch.bool)
        margin_gain = torch.zeros(num_classes, device=candidate_anchors.device, dtype=torch.float32)
        finite_self = torch.isfinite(slow_margin) & torch.isfinite(fast_margin)
        if torch.any(finite_self):
            valid_labels = label_tensor[finite_self]
            valid[valid_labels] = True
            margin_gain[valid_labels] = fast_margin[finite_self] - slow_margin[finite_self]

        class_mask = torch.ones_like(slow_dist, dtype=torch.bool)
        class_mask[row_idx, label_tensor] = False
        slow_intrusion = slow_dist - slow_same.view(-1, 1)
        fast_intrusion = fast_dist - fast_same.view(-1, 1)
        intrusion_gain = fast_intrusion - slow_intrusion
        intrusion_gain = intrusion_gain.masked_fill(~class_mask, float("inf"))
        incoming_gain = intrusion_gain.min(dim=0).values
        finite_incoming = torch.isfinite(incoming_gain)
        margin_gain[valid & finite_incoming] = torch.minimum(
            margin_gain[valid & finite_incoming],
            incoming_gain[valid & finite_incoming],
        )

        if not torch.any(valid):
            return gate

        scale = radii.to(device=candidate_anchors.device, dtype=torch.float32).pow(2).clamp_min(1e-12)
        semantic_gate = (1.0 + margin_gain / scale).clamp(0.0, 1.0)
        gate[valid] = semantic_gate[valid]

        valid_gate = gate[valid]
        self.last_fast_anchor_semantic_gate = float(valid_gate.mean().item()) if valid_gate.numel() > 0 else 0.0
        self.last_fast_anchor_semantic_slow_margin = (
            float(slow_margin[finite_self].mean().item()) if torch.any(finite_self) else 0.0
        )
        self.last_fast_anchor_semantic_fast_margin = (
            float(fast_margin[finite_self].mean().item()) if torch.any(finite_self) else 0.0
        )
        return gate

    @torch.no_grad()
    def resolve_fast_anchor_stale_signal(self, slow_anchors, num_classes, device):
        stale = torch.zeros(num_classes, device=device, dtype=torch.float32)
        valid = torch.zeros(num_classes, device=device, dtype=torch.bool)
        slow_anchors = torch.nan_to_num(
            slow_anchors.float().to(device=device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= num_classes or label >= slow_anchors.size(0):
                continue
            if proto is None or isinstance(proto, list):
                continue
            proto = torch.as_tensor(proto, device=device, dtype=torch.float32).view(-1)
            if proto.numel() != slow_anchors.size(1):
                continue
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            stale[label] = (proto - slow_anchors[label]).pow(2).sum().clamp_min(0.0)
            valid[label] = True
        return stale, valid

    @torch.no_grad()
    def resolve_fast_anchor_trust_beta(self, delta, slow_anchors=None):
        num_classes = int(delta.size(0))
        device = delta.device
        beta = torch.ones(num_classes, device=device, dtype=torch.float32)
        self.last_fast_anchor_trust_beta = 0.0
        self.last_fast_anchor_trust_lag = 0.0
        self.last_fast_anchor_trust_signal_value = 0.0
        self.last_fast_anchor_trust_noise = 0.0
        self.last_fast_anchor_trust_support = 0.0

        if self.fast_anchor_bound != "trust_region":
            return beta

        if self.fast_anchor_proto_trust == "off":
            self.last_fast_anchor_trust_beta = 1.0
            return beta

        lag2 = delta.float().pow(2).sum(dim=1).clamp_min(0.0)
        signal2 = lag2
        stale2, has_stale = self.resolve_fast_anchor_stale_signal(slow_anchors, num_classes, device)
        if torch.any(has_stale):
            signal2 = torch.where(has_stale, stale2, signal2)

        noise = torch.zeros(num_classes, device=device, dtype=torch.float32)
        support = torch.zeros(num_classes, device=device, dtype=torch.float32)
        has_stats = torch.zeros(num_classes, device=device, dtype=torch.bool)

        for label, value in self.proto_disagreement.items():
            label = int(label)
            if 0 <= label < num_classes:
                noise[label] = max(float(value), 0.0)
                support[label] = max(float(self.proto_support.get(label, 0.0)), 0.0)
                has_stats[label] = True

        if torch.any(has_stats):
            noise_scale = 1.0
            if self.fast_anchor_trust_norm == "radius":
                noise_scale = self.current_fast_anchor_radius_rho() ** 2
            effective_noise = noise * float(noise_scale)
            raw_beta = signal2 / (signal2 + effective_noise + 1e-12)
            support_trust = support / (support + 1.0)
            beta[has_stats] = (raw_beta * support_trust).clamp(0.0, 1.0)[has_stats]
            stat_beta = beta[has_stats]
            stat_lag = lag2[has_stats].sqrt()
            stat_signal = signal2[has_stats].sqrt()
            stat_noise = effective_noise[has_stats].sqrt()
            stat_support = support[has_stats]
        else:
            stat_beta = beta
            stat_lag = lag2.sqrt()
            stat_signal = signal2.sqrt()
            stat_noise = noise
            stat_support = support

        self.last_fast_anchor_trust_beta = float(stat_beta.mean().item()) if stat_beta.numel() > 0 else 0.0
        self.last_fast_anchor_trust_lag = float(stat_lag.mean().item()) if stat_lag.numel() > 0 else 0.0
        self.last_fast_anchor_trust_signal_value = (
            float(stat_signal.mean().item()) if stat_signal.numel() > 0 else 0.0
        )
        self.last_fast_anchor_trust_noise = float(stat_noise.mean().item()) if stat_noise.numel() > 0 else 0.0
        self.last_fast_anchor_trust_support = float(stat_support.mean().item()) if stat_support.numel() > 0 else 0.0
        return beta

    @torch.no_grad()
    def apply_fast_anchor_bound(self, accumulate=False):
        if not accumulate:
            self.last_fast_anchor_bound_radius = 0.0
            self.last_fast_anchor_radius_rho = self.current_fast_anchor_radius_rho()
            self.last_fast_anchor_bound_delta = 0.0
            self.last_fast_anchor_bound_clip_frac = 0.0
            self.last_fast_anchor_trust_beta = 0.0
            self.last_fast_anchor_trust_lag = 0.0
            self.last_fast_anchor_trust_signal_value = 0.0
            self.last_fast_anchor_trust_noise = 0.0
            self.last_fast_anchor_trust_support = 0.0
            self.last_fast_anchor_effective_beta = 0.0
            self.last_fast_anchor_semantic_gate = 0.0
            self.last_fast_anchor_semantic_slow_margin = 0.0
            self.last_fast_anchor_semantic_fast_margin = 0.0

        has_trust_region = self.fast_anchor_bound == "trust_region"
        has_semantic_gate = self.fast_anchor_semantic_gate == "margin"
        if (
            (not has_trust_region and not has_semantic_gate)
            or self.semantic_anchors is None
            or self.stable_semantic_anchors is None
            or self.semantic_anchors.shape != self.stable_semantic_anchors.shape
        ):
            return

        fast_anchors = torch.nan_to_num(
            self.semantic_anchors.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        slow_anchors = torch.nan_to_num(
            self.stable_semantic_anchors.float().to(device=fast_anchors.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        delta = fast_anchors - slow_anchors
        if has_trust_region:
            radii = self.resolve_fast_anchor_bound_radii(slow_anchors).to(device=fast_anchors.device)
            if radii.numel() != fast_anchors.size(0):
                return
            valid_radii = radii[torch.isfinite(radii) & (radii > 0)]
            self.last_fast_anchor_bound_radius = float(valid_radii.mean().item()) if valid_radii.numel() > 0 else 0.0
            if valid_radii.numel() == 0:
                return
        else:
            radii = self.resolve_fast_anchor_bound_radii(slow_anchors).to(device=fast_anchors.device)
            if radii.numel() != fast_anchors.size(0):
                return
            valid_radii = radii[torch.isfinite(radii) & (radii > 0)]
            self.last_fast_anchor_bound_radius = float(valid_radii.mean().item()) if valid_radii.numel() > 0 else 0.0
            if valid_radii.numel() == 0:
                return

        delta_norm = delta.norm(p=2, dim=1).clamp_min(1e-12)
        if has_trust_region:
            clip_mask = delta_norm > radii.clamp_min(1e-12)
            scale = torch.ones_like(delta_norm)
            scale[clip_mask] = radii[clip_mask] / delta_norm[clip_mask]
            bounded_delta = delta * scale.view(-1, 1)
            beta = self.resolve_fast_anchor_trust_beta(delta, slow_anchors).to(
                device=fast_anchors.device,
                dtype=fast_anchors.dtype,
            )
        else:
            clip_mask = torch.zeros_like(delta_norm, dtype=torch.bool)
            bounded_delta = delta
            beta = torch.ones(fast_anchors.size(0), device=fast_anchors.device, dtype=fast_anchors.dtype)

        candidate = slow_anchors + bounded_delta
        if has_semantic_gate:
            semantic_gate = self.resolve_fast_anchor_semantic_gate(slow_anchors, candidate, radii).to(
                device=fast_anchors.device,
                dtype=fast_anchors.dtype,
            )
            beta = beta * semantic_gate
        valid_beta = beta[torch.isfinite(beta)]
        self.last_fast_anchor_effective_beta = float(valid_beta.mean().item()) if valid_beta.numel() > 0 else 0.0
        bounded = slow_anchors + bounded_delta * beta.view(-1, 1)
        correction = (bounded - fast_anchors).norm(p=2, dim=1).mean()

        if torch.isfinite(correction):
            self.semantic_anchors.copy_(bounded.to(device=self.semantic_anchors.device, dtype=self.semantic_anchors.dtype))
            self.last_fast_anchor_bound_delta += float(correction.item())
            self.last_fast_anchor_bound_clip_frac = max(
                self.last_fast_anchor_bound_clip_frac,
                float(clip_mask.float().mean().item()),
            )

    def anchor_singular_values(self, anchors):
        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if anchors.ndim != 2 or anchors.size(0) < 2:
            return torch.empty(0, device=anchors.device, dtype=anchors.dtype)

        centered = anchors - anchors.mean(dim=0, keepdim=True)
        centered = centered / float(max(centered.size(1), 1)) ** 0.5
        rank = max(min(centered.size(0) - 1, centered.size(1)), 1)
        singular_values = torch.linalg.svdvals(centered)
        return singular_values[:rank].clamp_min(0.0)

    @torch.no_grad()
    def anchor_volume_stats(self):
        singular_values = self.anchor_singular_values(self.semantic_anchors)

        if self.anchor_spectrum_ref is None:
            return 1.0, 0.0

        ref = self.anchor_spectrum_ref.to(
            device=singular_values.device,
            dtype=singular_values.dtype,
        )

        if singular_values.numel() == 0 or singular_values.numel() != ref.numel():
            return 1.0, 0.0

        ratio = singular_values.clamp_min(1e-12) / ref.clamp_min(1e-12)
        log_ratio = torch.log(ratio.clamp_min(1e-12))
        vol_ratio = torch.exp(2.0 * log_ratio.mean())
        spec_min = ratio.min()

        if not torch.isfinite(vol_ratio):
            vol_ratio = torch.ones((), device=singular_values.device)
        if not torch.isfinite(spec_min):
            spec_min = torch.zeros((), device=singular_values.device)

        return float(vol_ratio.item()), float(spec_min.item())

    def collapse_guard_reference_spectrum(self, singular_values):
        if self.anchor_spectrum_ref is None:
            return None
        if singular_values.numel() == 0 or singular_values.numel() != self.anchor_spectrum_ref.numel():
            return None
        return self.anchor_spectrum_ref.to(
            device=singular_values.device,
            dtype=singular_values.dtype,
        ).clamp_min(1e-12)

    def current_collapse_guard_weight(self, spectrum_ratio):
        if self.collapse_guard_mode != "auto":
            return 0.0

        if spectrum_ratio is None or spectrum_ratio.numel() == 0:
            return 0.0
        min_ratio = torch.nan_to_num(
            spectrum_ratio.float(),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        ).min()
        weight = (1.0 - min_ratio).clamp(0.0, 1.0)
        if not torch.isfinite(weight):
            return 0.0
        return float(weight.item())

    @torch.no_grad()
    def apply_anchor_collapse_guard(self):
        self.last_collapse_guard_delta = 0.0
        self.last_collapse_guard_weight = 0.0
        if (
            self.collapse_guard_mode != "auto"
            or self.semantic_anchors is None
            or self.semantic_anchors.ndim != 2
            or self.semantic_anchors.size(0) < 2
        ):
            return

        anchors = torch.nan_to_num(
            self.semantic_anchors.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        center = anchors.mean(dim=0, keepdim=True)
        centered = (anchors - center) / float(max(anchors.size(1), 1)) ** 0.5
        rank = max(min(centered.size(0) - 1, centered.size(1)), 1)

        try:
            u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            return

        singular_values = singular_values[:rank].clamp_min(0.0)
        u = u[:, :rank]
        vh = vh[:rank, :]
        if singular_values.numel() == 0:
            return

        ref = self.collapse_guard_reference_spectrum(singular_values)
        if ref is None:
            return

        ratio = singular_values.clamp_min(1e-12) / ref
        log_ratio = torch.log(ratio.clamp_min(1e-12))
        if torch.isfinite(log_ratio).all():
            vol_ratio = torch.exp(2.0 * log_ratio.mean())
            if torch.isfinite(vol_ratio):
                self.last_collapse_guard_geometry_ratio = float(vol_ratio.item())
            self.last_collapse_guard_spectrum_error = float(torch.sqrt(log_ratio.pow(2).mean()).item())

        self.last_collapse_guard_min_ratio = float(ratio.min().item())
        guard_weight = self.current_collapse_guard_weight(ratio)
        self.last_collapse_guard_weight = guard_weight
        if guard_weight <= 0:
            return

        guarded_centered = (u * ref.view(1, -1)) @ vh
        guarded = guarded_centered * float(max(anchors.size(1), 1)) ** 0.5 + center
        updated = anchors + guard_weight * (guarded - anchors)

        delta = (updated - anchors).norm(p=2, dim=1).mean()
        if torch.isfinite(delta):
            self.last_collapse_guard_delta = float(delta.item())
            self.semantic_anchors.copy_(updated.detach())

    @torch.no_grad()
    def update_anchor_geometry_metric(self):
        geometry = self.compute_anchor_geometry(self.semantic_anchors)
        self.last_anchor_geometry = geometry["geometry"]
        self.last_anchor_norm = geometry["norm"]
        self.last_anchor_pair_mean = geometry["pair_mean"]
        self.last_anchor_pair_min = geometry["pair_min"]
        self.last_anchor_effective_rank = geometry["effective_rank"]
        stable_geometry = self.compute_anchor_geometry(self.stable_semantic_anchors)
        self.last_stable_anchor_norm = stable_geometry["norm"]
        self.last_stable_anchor_pair_mean = stable_geometry["pair_mean"]
        self.last_stable_anchor_pair_min = stable_geometry["pair_min"]
        self.last_stable_anchor_effective_rank = stable_geometry["effective_rank"]
        self.update_anchor_spectrum_metric(self.semantic_anchors)

    @torch.no_grad()
    def update_anchor_spectrum_metric(self, anchors):
        self.last_collapse_guard_min_ratio = 0.0
        self.last_collapse_guard_geometry_ratio = 1.0
        self.last_collapse_guard_spectrum_error = 0.0
        if anchors is None:
            return

        anchors = torch.as_tensor(anchors, device=self.device, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.size(0) < 2:
            return

        singular_values = self.anchor_singular_values(anchors)
        if singular_values.numel() == 0:
            return

        ref = self.collapse_guard_reference_spectrum(singular_values)
        if ref is None:
            return

        ratio = singular_values.clamp_min(1e-12) / ref
        log_ratio = torch.log(ratio.clamp_min(1e-12))
        if torch.isfinite(log_ratio).all():
            vol_ratio, spec_min = self.anchor_volume_stats()
            self.last_collapse_guard_geometry_ratio = vol_ratio
            self.last_collapse_guard_spectrum_error = float(torch.sqrt(log_ratio.pow(2).mean()).item())
            self.last_collapse_guard_min_ratio = spec_min

    def anchor_log_geometry_tensor(self, anchors):
        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        centered = anchors - anchors.mean(dim=0, keepdim=True)
        gram = centered @ centered.t()
        gram = gram / float(max(centered.size(1), 1))
        gram = 0.5 * (gram + gram.t())
        eigvals = torch.linalg.eigvalsh(gram)
        rank = max(min(centered.size(0) - 1, centered.size(1)), 1)
        eigvals = eigvals[-rank:]
        eps = 1e-4 * eigvals.detach().abs().mean().clamp_min(1.0)
        eigvals = eigvals.clamp_min(eps)
        log_geometry = torch.log(eigvals).mean()
        if not torch.isfinite(log_geometry):
            return torch.zeros((), device=anchors.device, dtype=anchors.dtype)
        return log_geometry

    @torch.no_grad()
    def compute_anchor_geometry(self, anchors):
        empty = {
            "geometry": 1.0,
            "norm": 0.0,
            "pair_mean": 0.0,
            "pair_min": 0.0,
            "effective_rank": 0.0,
        }
        if anchors is None:
            return empty
        anchors = torch.as_tensor(anchors, device=self.device, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.size(0) < 2:
            return empty

        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        value = float(self.anchor_log_geometry_tensor(anchors).detach().item())
        geometry = float(np.exp(np.clip(value, -80.0, 80.0)))

        norm = float(anchors.norm(p=2, dim=1).mean().item())
        dist = torch.cdist(anchors, anchors, p=2)
        off_diag = ~torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        pair_values = dist[off_diag]
        pair_mean = float(pair_values.mean().item()) if pair_values.numel() > 0 else 0.0
        pair_min = float(pair_values.min().item()) if pair_values.numel() > 0 else 0.0

        centered = anchors - anchors.mean(dim=0, keepdim=True)
        gram = centered @ centered.t()
        gram = 0.5 * (gram + gram.t())
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        eig_sum = eigvals.sum()
        if eig_sum <= 0:
            effective_rank = 0.0
        else:
            probs = eigvals / eig_sum.clamp_min(1e-12)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
            effective_rank = float(torch.exp(entropy).item())

        return {
            "geometry": geometry,
            "norm": norm,
            "pair_mean": pair_mean,
            "pair_min": pair_min,
            "effective_rank": effective_rank,
        }
