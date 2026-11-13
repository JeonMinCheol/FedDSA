from collections import defaultdict
import math
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientfeddsa import clientfeddsa
from flcore.servers.serverbase import Server
from utils.feddsa_utils import prototype_anchor_metrics, configure_utility_module, aggregate_count_prototypes
from utils.raar_marginal_refinement import marginal_raar_direction_scores, propose_marginal_raar_beta
from utils.raar_prior_refinement import validate_raar_proposal, apply_raar_beta, compare_raar_movement
from utils.raar_refinement_probe import CachedRAARQueries
from utils.candidate_noise_statistics import cohort_candidate_confidence
from utils.raar_next_gap import candidate_next_gap_kwargs


class FedDSA(Server):
    anchor_momentum = 0.9999
    # Equal-weight candidate averaging; prototype trust is applied afterwards.
    anchor_candidate_mix = 0.5

    def __init__(self, args, times):
        configure_utility_module(args)
        super().__init__(args, times)
        self.raar_enabled = args.raar_enabled
        self.candidate_next_gap = args.candidate_next_gap
        self.anchor_momentum = args.anchor_momentum
        self.candidate_trust_max_weight = args.candidate_trust_max_weight
        self.utility_trust_penalty = args.utility_trust_penalty
        self.utility_trust_temperature = args.utility_trust_temperature
        self.set_slow_clients()
        self.set_clients(clientfeddsa)
        for client in self.clients:
            model = client.model
            if (model.shared_classifier_mode != "linear"
                    or not isinstance(model.shared_head, torch.nn.Linear)
                    or model.shared_head.bias is None
                    or not isinstance(model.shared_projector, torch.nn.Linear)):
                raise ValueError("FedDSA requires a linear classifier and linear projector.")

        self.num_classes = int(args.num_classes)
        self.feat_dim = int(args.feature_dim)
        self.anchor_mode = "single"
        self.anchor_codebook = str(
            getattr(args, "anchor_codebook", "gaussian")
        ).lower()
        if self.anchor_codebook not in {"gaussian", "simplex"}:
            self.anchor_codebook = "gaussian"
        self.global_protos = {}
        self.proto_disagreement = {}
        self.proto_support = {}
        self.proto_reliability = {}
        self.reliability_weighted_aggregation = bool(
            getattr(args, "reliability_weighted_aggregation", False)
        )
        self.reliability_aggregation_temperature = max(
            float(getattr(args, "reliability_aggregation_temperature", 1.0)),
            1e-6,
        )
        self.reliability_aggregation_geometry = str(
            getattr(args, "reliability_aggregation_geometry", "euclidean")
        ).lower()
        if self.reliability_aggregation_geometry not in {"euclidean", "cosine"}:
            self.reliability_aggregation_geometry = "euclidean"
        self.proto_records = []
        self.mapping_diagnostics = defaultdict(list)
        self.last_proto_update_round = defaultdict(lambda: -1)
        self.anchor_spectrum_ref = None
        self.fast_radius_rho = float(getattr(args, "fast_anchor_radius_rho", 0.6))
        if self.fast_radius_rho < 0.0:
            self.fast_radius_rho = 0.0
        self.fast_anchor_ema = min(
            max(float(getattr(args, "fast_anchor_ema", 1.0)), 0.0), 1.0
        )
        self.moving_anchor_ema = args.moving_anchor_ema
        self.anchor_candidate_mix = args.anchor_candidate_mix
        self.dual_disagreement_gate = bool(
            getattr(args, "dual_disagreement_gate", False)
        )
        self.fast_anchor_proto_trust = str(getattr(args, "fast_anchor_proto_trust", "auto")).lower()
        if self.fast_anchor_proto_trust not in {"off", "auto"}:
            self.fast_anchor_proto_trust = "auto"
        self.semantic_gate_mode = str(
            getattr(args, "semantic_gate_mode", "off")
        ).lower()
        if self.semantic_gate_mode not in {
            "memory",
            "meta",
            "off",
        }:
            self.semantic_gate_mode = "off"
        self.meta_anchor_lr = float(getattr(args, "meta_anchor_lr", 1000.0))
        self.memory_max_per_class = max(
            1, int(getattr(args, "memory_max_per_class", 2))
        )
        self.memory_min_support = max(
            0.0, float(getattr(args, "memory_min_support", 1.0))
        )
        self.memory_confidence_floor = float(
            getattr(args, "memory_confidence_floor", 0.7)
        )
        self.memory_temperature = max(
            1e-6, float(getattr(args, "memory_temperature", 1.0))
        )
        self.memory_update_enabled = bool(
            getattr(args, "memory_update_enabled", self.semantic_gate_mode == "memory")
        )
        self.fixed_anchor = bool(getattr(args, "fixed_anchor", False))
        self.freeze_semantic_anchor = bool(
            getattr(args, "freeze_semantic_anchor", False)
        )
        self.scalar_reliability_state = bool(
            getattr(args, "scalar_reliability_state", False)
        )
        self.prototype_utility = bool(
            getattr(args, "prototype_utility", False)
        )
        self.utility_ema = min(
            max(float(getattr(args, "utility_ema", 0.25)), 0.0), 1.0
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
        self.utility_signal_mode = str(
            getattr(args, "utility_signal_mode", "observed")
        ).lower()
        if self.utility_signal_mode not in {"observed", "shuffled", "uniform", "zscore"}:
            self.utility_signal_mode = "observed"
        self.utility_positive_only = bool(
            getattr(args, "utility_positive_only", False)
        )
        self.reliability_state_alpha = min(
            max(float(getattr(args, "reliability_state_alpha", 0.25)), 0.0), 1.0
        )
        self.reliability_staleness_tau = max(
            float(getattr(args, "reliability_staleness_tau", 20.0)), 1e-6
        )
        self.trusted_proto_memory = {}
        self.last_anchor_update_norm = 0.0
        self.meta_anchor_grads = None
        self.meta_anchor_support = 0.0

        # Explicit role modes update their own states without prototype trust.
        self.new_anchor_mode = str(getattr(args, "new_anchor_mode", "none")).lower()
        if self.new_anchor_mode not in {
            "none",
            "relation_memory",
            "semantic_discriminative",
            "observation_validation",
            "meta_separated",
            "boundary",
            "class_boundary",
        }:
            self.new_anchor_mode = "none"
        self.new_anchor_ema = min(
            max(float(getattr(args, "new_anchor_ema", 0.25)), 0.0), 1.0
        )
        self.relation_memory_lr = min(
            max(float(getattr(args, "relation_memory_lr", 0.25)), 0.0), 1.0
        )
        self.relation_anchor_lr = max(
            float(getattr(args, "relation_anchor_lr", 0.02)), 0.0
        )
        self.relation_observation_weight = max(
            float(getattr(args, "relation_observation_weight", 0.1)), 0.0
        )
        self.relation_step_scale = max(
            float(getattr(args, "relation_step_scale", 0.25)), 0.0
        )
        self.discriminative_anchor_lr = max(
            float(getattr(args, "discriminative_anchor_lr", 0.05)), 0.0
        )
        self.discriminative_anchor_reg = max(
            float(getattr(args, "discriminative_anchor_reg", 0.01)), 0.0
        )
        self.discriminative_step_scale = max(
            float(getattr(args, "discriminative_step_scale", 0.25)), 0.0
        )
        self.discriminative_codebook_reg = max(
            float(getattr(args, "discriminative_codebook_reg", 0.0)), 0.0
        )
        self.discriminative_metric = str(
            getattr(args, "discriminative_metric", "euclidean")
        ).lower()
        if self.discriminative_metric not in {"euclidean", "cosine"}:
            self.discriminative_metric = "euclidean"
        self.validation_anchor_lr = max(
            float(getattr(args, "validation_anchor_lr", 0.05)), 0.0
        )
        self.validation_step_scale = max(
            float(getattr(args, "validation_step_scale", 0.25)), 0.0
        )
        self.meta_step_scale = max(
            float(getattr(args, "meta_step_scale", 0.25)), 0.0
        )
        self.boundary_anchor_ema = min(
            max(float(getattr(args, "boundary_anchor_ema", 0.001)), 0.0), 1.0
        )
        self.class_boundary_ema = min(
            max(float(getattr(args, "class_boundary_ema", 0.1)), 0.0), 1.0
        )
        self.class_boundary_min = max(
            float(getattr(args, "class_boundary_min", 0.1)), 0.0
        )
        self.class_boundary_max = max(
            float(getattr(args, "class_boundary_max", 1.5)),
            self.class_boundary_min,
        )
        self.meta_separated_initialized = False
        self.semantic_discriminative_initialized = False
        self.relation_targets = None
        self.relation_confidence = None
        self.relation_observed = None
        self.validation_anchor_grads = None
        self.validation_anchor_support = 0.0
        self.validation_anchor_gains = None
        self.validation_gain_support = None
        self.utility_validation_probe = bool(
            getattr(args, "utility_validation_probe", False)
        )
        self.utility_validation_metrics = {
            "pearson": 0.0,
            "spearman": 0.0,
            "sign_agreement": 0.0,
            "negative_utility_rate": 0.0,
            "class_coverage": 0.0,
            "pairs": 0,
        }
        self.new_anchor_metrics = {}
        run_name = os.path.basename(os.path.normpath(str(self.save_folder_name))) or "run"
        metrics_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../logs")
        )
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_log_path = os.path.join(
            metrics_dir, f"feddsa_metrics_{run_name}.jsonl"
        )
        self.utility_gate_log_path = os.path.join(
            metrics_dir, f"feddsa_utility_gates_{run_name}.jsonl"
        )
        self.previous_utility_gate_vectors = {}
        self.last_train_diagnostics = {}
        self.current_round_elapsed_sec = None
        self.total_elapsed_sec = 0.0
        self.current_proto_upload_bytes = 0
        self.current_proto_feature_upload_bytes = 0
        self.current_proto_count_upload_bytes = 0
        self.current_anchor_broadcast_bytes = 0

        norm_cfg = {"type": "SyncBN", "requires_grad": True}
        self.anchor_projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            build_norm_layer(norm_cfg, self.feat_dim)[1],
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
            build_norm_layer(norm_cfg, self.feat_dim)[1],
            nn.ReLU(inplace=True),
        ).to(self.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + 3407)
        random_center = torch.randn(self.num_classes, self.feat_dim, generator=generator)
        with torch.no_grad():
            gaussian_anchors = self.anchor_projector(
                random_center.to(self.device)
            ).detach()
            self.semantic_anchors = self.initial_anchor_bank(
                gaussian_anchors, generator
            )
            self.stable_semantic_anchors = self.semantic_anchors.detach().clone()
            # Explicit role separation for the new dual-anchor path:
            # semantic_anchors track class meaning; adaptive_anchors are
            # updated by discriminative/performance signals.
            self.adaptive_anchors = self.semantic_anchors.detach().clone()
            self.initial_semantic_anchors = self.semantic_anchors.detach().clone()
            self.anchor_spectrum_ref = self.anchor_singular_values(self.semantic_anchors).detach()
            self.class_reliability = torch.ones(
                self.num_classes, dtype=torch.float32, device=self.device
            )
            self.class_utility = torch.zeros(
                self.num_classes, dtype=torch.float32, device=self.device
            )
            self.class_utility_count = torch.zeros(
                self.num_classes, dtype=torch.float32, device=self.device
            )
            self.class_boundary_margin = torch.full(
                (self.num_classes,),
                min(
                    max(float(getattr(args, "mcl_hard_margin", 0.5)), self.class_boundary_min),
                    self.class_boundary_max,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            self.class_observation_age = torch.zeros(
                self.num_classes, dtype=torch.float32, device=self.device
            )
            initial_relations = torch.cdist(
                self.stable_semantic_anchors.float(),
                self.stable_semantic_anchors.float(),
                p=2,
            ).pow(2)
            self.relation_targets = initial_relations.detach().clone()
            self.relation_confidence = torch.zeros_like(initial_relations)
            self.relation_observed = torch.zeros(
                self.num_classes, dtype=torch.bool, device=self.device
            )

        print("=" * 50)
        print("FedDSA config")
        print(
            f"feat_dim={self.feat_dim} "
            "anchor_losses=raw(mse+mcl) "
            f"classifier=linear agreement_head=classifier_logits"
        )
        print(
            f"anchors={self.anchor_mode} "
            f"anchor_update={'role_specific' if self.new_anchor_mode != 'none' else 'prototype_trust'} "
            f"proto_trust={self.fast_anchor_proto_trust} "
            f"semantic_gate={self.semantic_gate_mode} "
            f"fixed_anchor={self.fixed_anchor} "
            f"freeze_semantic={self.freeze_semantic_anchor} "
            f"codebook={self.anchor_codebook} "
            f"scalar_reliability={self.scalar_reliability_state} "
            f"reliability_agg={self.reliability_weighted_aggregation} "
            f"separate_roles={getattr(args, 'separate_anchor_roles', True)} "
            f"memory_slots={self.memory_max_per_class} "
            f"new_anchor_mode={self.new_anchor_mode}"
        )
        if self.prototype_utility:
            print(f"utility_control=local_only anchor_candidate_mix={self.anchor_candidate_mix}")
        print(f"raar=NextGap enabled={self.raar_enabled} prototype_aggregation=class_sample_count "
              f"gamma={self.anchor_candidate_mix} reference_momentum={self.anchor_momentum}")
        print("=" * 50)

    @torch.no_grad()
    def initial_anchor_bank(self, gaussian_anchors, generator):
        """Build the initial single-anchor codebook used by all roles.

        The simplex option keeps the Gaussian bank's average norm while
        replacing its class geometry with an equal-angle codebook.  It is a
        diagnostic for random geometry; it does not change local losses.
        """

        if self.anchor_codebook == "gaussian" or self.num_classes < 2:
            return gaussian_anchors.detach().clone()

        classes = self.num_classes
        simplex = torch.eye(classes, device=self.device, dtype=torch.float32)
        simplex -= simplex.mean(dim=0, keepdim=True)
        if self.feat_dim >= classes:
            codebook = torch.nn.functional.pad(
                simplex, (0, self.feat_dim - classes)
            )
        else:
            # A low-dimensional projection is only a fallback for feature
            # spaces smaller than the class count.
            projection = torch.randn(
                classes, self.feat_dim, generator=generator, dtype=torch.float32
            )
            codebook = simplex @ projection.to(self.device)
        codebook = torch.nn.functional.normalize(codebook, dim=1)
        if self.feat_dim >= classes:
            pairwise = codebook @ codebook.t()
            off_diagonal = ~torch.eye(
                classes, device=self.device, dtype=torch.bool
            )
            expected = -1.0 / float(classes - 1)
            if (pairwise[off_diagonal] - expected).abs().max() > 1e-4:
                raise RuntimeError("simplex anchor geometry check failed")
        target_norm = gaussian_anchors.float().norm(dim=1).mean().clamp_min(1e-6)
        codebook = codebook * target_norm
        if not torch.isfinite(codebook).all():
            raise RuntimeError("non-finite anchor codebook")
        return codebook.to(device=self.device, dtype=gaussian_anchors.dtype)

    def record_mapping_diagnostics(self, round_idx, adaptive_anchors, stable_anchors):
        metrics = prototype_anchor_metrics(
            self.proto_records,
            adaptive_anchors.cpu(),
            stable_anchors.cpu(),
        )
        for name in (
            "proto_retrieval_accuracy",
            "proto_anchor_margin",
            "positive_alignment_cosine",
            "prototype_dispersion",
            "relational_distance_error",
        ):
            self.mapping_diagnostics[name].append(float(metrics[name]))

        reentry_values = defaultdict(list)
        for label, values in metrics["by_class"].items():
            previous = self.last_proto_update_round[label]
            if previous < 0:
                continue
            staleness = round_idx - previous
            bucket = "short" if staleness <= 5 else "medium" if staleness <= 20 else "long"
            reentry_values[f"reentry_{bucket}_proto_retrieval_accuracy"].append(
                values["proto_retrieval_accuracy"]
            )
            reentry_values[f"reentry_{bucket}_proto_anchor_margin"].append(
                values["proto_anchor_margin"]
            )
        for bucket in ("short", "medium", "long"):
            for metric in ("proto_retrieval_accuracy", "proto_anchor_margin"):
                name = f"reentry_{bucket}_{metric}"
                self.mapping_diagnostics[name].append(float(np.mean(reentry_values[name])) if reentry_values[name] else np.nan)

        for label in metrics["by_class"]:
            self.last_proto_update_round[label] = round_idx

    def train(self):
        train_started = time.perf_counter()
        self.send_semantic_state()

        for round_idx in range(self.global_rounds + 1):
            round_started = time.perf_counter()
            self.current_round = round_idx
            for client in self.clients:
                if hasattr(client, "set_round"):
                    client.set_round(round_idx)
            self.selected_clients = self.select_clients()

            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            gate_client = None
            if self.prototype_utility and self.utility_local_gate:
                gate_client = next(
                    (
                        client for client in self.selected_clients
                        if client.train_samples >= client.batch_size
                        and client.local_epochs > 0
                    ),
                    None,
                )
                if gate_client is not None:
                    gate_client.last_local_utility_gate = None
                    gate_client.capture_utility_gate = True
            results = self.run_clients(self.selected_clients, return_exceptions=True)
            for client, result in results:
                if isinstance(result, Exception):
                    raise RuntimeError(f"FedDSA client {client.id} failed during local training.") from result

            self.receive_protos()
            if self.scalar_reliability_state:
                self.update_scalar_reliability_state()
            if self.new_anchor_mode in {
                "relation_memory",
                "semantic_discriminative",
                "observation_validation",
                "meta_separated",
                "boundary",
                "class_boundary",
            }:
                mapping_anchor = self.adaptive_anchors.detach()
                mapping_reference = self.semantic_anchors.detach()
            else:
                mapping_anchor = self.semantic_anchors.detach()
                mapping_reference = self.stable_semantic_anchors.detach()
            self.record_mapping_diagnostics(
                round_idx,
                mapping_anchor,
                mapping_reference,
            )
            anchor_before = (
                self.semantic_anchors.detach().clone()
                if not self.fixed_anchor
                else None
            )
            if not self.fixed_anchor:
                if self.new_anchor_mode != "none":
                    self.update_new_anchor_mode()
                elif self.semantic_gate_mode == "meta":
                    self.update_meta_semantic_anchors()
                else:
                    self.update_semantic_anchors()
                if self.new_anchor_mode == "none":
                    self.apply_prototype_trust(anchor_before)
                    self.update_stable_anchors()
                if self.memory_update_enabled and self.new_anchor_mode == "none":
                    self.update_trusted_proto_memory()
            if anchor_before is None:
                self.last_anchor_update_norm = 0.0
                anchor_step_norms = None
            else:
                delta = self.semantic_anchors.detach().float() - anchor_before.float()
                anchor_step_norms = delta.norm(dim=1)
                self.last_anchor_update_norm = float(
                    anchor_step_norms.mean().item()
                )
            self.send_semantic_state()
            if self.prototype_utility or self.raar_enabled:
                self.write_utility_gate_diagnostics(round_idx, gate_client, anchor_step_norms)
            if gate_client is not None:
                gate_client.capture_utility_gate = False
            self.current_round_elapsed_sec = float(time.perf_counter() - round_started)
            self.total_elapsed_sec = float(time.perf_counter() - train_started)

            if round_idx % self.eval_gap == 0 or round_idx == self.global_rounds:
                self.log_train_diagnostics(round_idx)
                self.log_anchor_geometry(round_idx)
                self.log_mapping_diagnostics(round_idx)
                self.write_round_metrics(round_idx)
                print("-" * 50)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        self.save_results()

    def log_train_diagnostics(self, round_idx):
        metric_names = [
            "task_loss",
            "anchor_loss",
            "anchor_mse_loss",
            "mcl_loss",
            "mcl_sample_margin",
            "total_loss",
            "anchor_agree_head_acc",
            "anchor_agree_both_wrong",
            "utility_weight_mean",
            "utility_weight_std",
            "boundary_loss",
            "boundary_active_rate",
            "boundary_margin",
            "class_boundary_loss",
            "class_boundary_active_rate",
            "mcl_tail_frac",
            "nonfinite_batches",
            "meta_grad_norm",
            "validation_grad_norm",
            "validation_before_loss",
            "validation_observation_loss",
        ]
        weighted = defaultdict(float)
        total_weight = 0.0
        for client in self.selected_clients:
            metrics = getattr(client, "latest_train_metrics", {})
            if not metrics:
                continue
            weight = float(max(client.train_samples, 1))
            total_weight += weight
            for name in metric_names:
                weighted[name] += float(metrics.get(name, 0.0)) * weight

        if total_weight <= 0:
            return
        for name in metric_names:
            weighted[name] /= total_weight
        self.last_train_diagnostics = {
            name: float(weighted[name]) for name in metric_names
        }

        print(
            f"[Round {round_idx}] FedDSATrain "
            f"task={weighted['task_loss']:.4f} "
            f"anchor={weighted['anchor_loss']:.4f} "
            f"mse={weighted['anchor_mse_loss']:.4f} "
            f"mcl={weighted['mcl_loss']:.4f} "
            f"total={weighted['total_loss']:.4f} "
            f"agree_head={weighted['anchor_agree_head_acc']:.4f} "
            f"both_wrong={weighted['anchor_agree_both_wrong']:.4f} "
            f"utility_weight={weighted['utility_weight_mean']:.4f} "
            f"utility_local_std={weighted['utility_weight_std']:.4f} "
            f"anchor_step={self.last_anchor_update_norm:.4f} "
            f"round_sec={self.current_round_elapsed_sec:.1f} "
            f"total_min={self.total_elapsed_sec / 60.0:.1f} "
            f"tail={weighted['mcl_tail_frac']:.4f} "
            f"skip={weighted['nonfinite_batches']:.1f} "
            f"val_grad={weighted['validation_grad_norm']:.4f}"
        )
        if self.new_anchor_mode != "none":
            self.log_new_anchor_diagnostics(round_idx)

    def write_utility_gate_diagnostics(self, round_idx, gate_client, anchor_step_norms):
        """Record local weights, class utility and probe measurements each round."""
        def vector(tensor):
            if tensor is None:
                return None
            values = tensor.detach().cpu().reshape(-1).tolist()
            return [value if math.isfinite(value) else None for value in values]

        def masked(values, counts):
            if values is None or counts is None:
                return None
            return [value if count is not None and count > 0 else None
                    for value, count in zip(values, counts)]

        def summary(values, name, neutral):
            if values is None:
                return None
            arr = np.asarray([np.nan if value is None else value for value in values])
            valid = np.isfinite(arr)
            if not valid.any():
                return None
            finite = arr[valid]
            quantiles = np.quantile(finite, [0.1, 0.5, 0.9])
            result = {
                "count": int(finite.size), "mean": float(finite.mean()),
                "std": float(finite.std()), "min": float(finite.min()),
                "p10": float(quantiles[0]), "median": float(quantiles[1]),
                "p90": float(quantiles[2]), "max": float(finite.max()),
                "neutral_value": float(neutral),
                "near_neutral_tolerance": float(0.01 * (abs(neutral) if neutral else 1.0)),
                "within_1pct_neutral_fraction": float(
                    np.mean(np.abs(finite - neutral) <= 0.01 * (abs(neutral) if neutral else 1.0))
                ),
            }
            previous = self.previous_utility_gate_vectors.get(name)
            if previous is not None:
                prev_round, prev_arr = previous
                common = valid & np.isfinite(prev_arr)
                if common.any():
                    change = np.abs(arr[common] - prev_arr[common])
                    result["change"] = {
                        "from_round": int(prev_round), "count": int(change.size),
                        "mean_abs": float(change.mean()),
                        "p90_abs": float(np.quantile(change, 0.9)),
                        "max_abs": float(change.max()),
                    }
            self.previous_utility_gate_vectors[name] = (int(round_idx), arr)
            return result

        def clipping(values):
            if values is None:
                return None
            finite = np.asarray([value for value in values if value is not None])
            if not finite.size:
                return None
            return {
                "count": int(finite.size),
                "below_min": int(np.count_nonzero(finite < self.utility_min_weight)),
                "above_max": int(np.count_nonzero(finite > self.utility_max_weight)),
            }

        captured = getattr(gate_client, "last_local_utility_gate", None)
        local = {key: vector(value) for key, value in captured.items()} if captured else {}
        local_weights = local.get("weight")
        normalize = bool(getattr(self.args, "utility_weight_normalize", True))
        local_neutral = self.utility_min_weight if self.utility_positive_only and not normalize else 1.0
        local_summary = summary(local_weights, "local", local_neutral)
        observation_count = vector(self.last_utility_observation_count)
        probe_query_count = vector(self.utility_validation_query_count)
        probe_gain_count = vector(self.utility_validation_gain_count)
        probe_gain = masked(vector(self.utility_validation_gain_value), probe_gain_count)

        def gate_by_gain(values):
            if values is None or probe_gain is None or probe_query_count is None:
                return None
            pairs = [
                (weight, gain)
                for weight, gain, count in zip(values, probe_gain, probe_query_count)
                if weight is not None and gain is not None and count is not None and count > 0
            ]
            if not pairs:
                return None
            negative = [weight for weight, gain in pairs if gain < 0]
            nonnegative = [weight for weight, gain in pairs if gain >= 0]
            negative_mean = float(np.mean(negative)) if negative else None
            nonnegative_mean = float(np.mean(nonnegative)) if nonnegative else None
            return {
                "negative_gain_classes": len(negative),
                "nonnegative_gain_classes": len(nonnegative),
                "negative_gain_mean_gate": negative_mean,
                "nonnegative_gain_mean_gate": nonnegative_mean,
                "nonnegative_minus_negative": (
                    nonnegative_mean - negative_mean
                    if nonnegative_mean is not None and negative_mean is not None else None
                ),
            }

        record = {
            "round": int(round_idx), "seed": int(self.random_seed),
            "goal": str(getattr(self.args, "goal", "")),
            "class_index": "arrays are indexed by class label; null means unavailable",
            "timing": "local uses the broadcast state; utility and probe follow this round's aggregation",
            "anchor_candidate_mix": float(self.anchor_candidate_mix),
            "temperature": float(self.utility_temperature),
            "signal_mode": str(self.utility_signal_mode),
            "utility_ema": float(self.utility_ema),
            "positive_only": bool(self.utility_positive_only),
            "local": {
                "enabled": bool(self.utility_local_gate),
                "captured_client": int(gate_client.id) if captured else None,
                "normalize": normalize,
                "min_weight": float(self.utility_min_weight),
                "max_weight": float(self.utility_max_weight),
                **local, "summary": local_summary,
                "first_clipping": clipping(local.get("before_clip")),
                "normalization_clipping": clipping(local.get("normalized_before_clip")),
            },
            "utility": {
                "query_count": observation_count,
                "raw_observed": masked(vector(self.last_utility_observed), observation_count),
                "signal_after_transform": masked(vector(self.last_utility_signal), observation_count),
                "ema_after_update": vector(self.class_utility),
                "contributing_clients": vector(self.last_utility_client_count),
                "client_sample_weighted_std": masked(vector(self.last_utility_client_std), observation_count),
                "positive_clients": vector(self.last_utility_positive_clients),
                "negative_clients": vector(self.last_utility_negative_clients),
            },
            "probe": {
                "enabled": bool(self.utility_validation_probe),
                "summary": {
                    key: value if math.isfinite(value) else None
                    for key, value in self.utility_validation_metrics.items()
                },
                "paired_query_count": probe_query_count,
                "paired_validation_count": probe_gain_count,
                "query_utility": masked(vector(self.utility_validation_query_value), probe_query_count),
                "validation_gain": probe_gain,
                "local_weight_by_gain": gate_by_gain(local_weights),
            },
            "anchor_step_norm": vector(anchor_step_norms),
        }
        record["raar_nextgap"] = getattr(self, "candidate_gain_stats", None)
        with open(self.utility_gate_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

        if round_idx % self.eval_gap == 0 or round_idx == self.global_rounds:
            def brief(stats):
                if stats is None:
                    return "inactive"
                change = stats.get("change", {}).get("mean_abs")
                delta = f"{change:.4g}" if change is not None else "n/a"
                return (f"min/p10/p90/max={stats['min']:.4f}/{stats['p10']:.4f}/"
                        f"{stats['p90']:.4f}/{stats['max']:.4f} delta={delta}")

            print(f"[Round {round_idx}] FedDSAGate local({brief(local_summary)})")

    def write_round_metrics(self, round_idx):
        """Persist evaluation and anchor diagnostics for aligned comparisons."""

        class_boundary_mode = self.new_anchor_mode == "class_boundary"
        role_mode = self.new_anchor_mode in {
            "semantic_discriminative",
            "meta_separated",
            "boundary",
            "class_boundary",
        }
        legacy_update_path = self.new_anchor_mode == "none"
        # In the explicit role-separated modes the legacy stable field is a
        # compatibility alias, not a third recurrent state or communication
        # payload.  Record this distinction so a reviewer can audit the
        # claimed number of states directly from the run log.
        anchor_mode = self.anchor_mode
        anchor_role_count = (
            1 if anchor_mode == "single" else 2 if role_mode else 3
        )
        independent_state_count = (
            2 if class_boundary_mode or anchor_mode != "single" else 1
        )

        record = {
            "round": int(round_idx),
            "goal": str(getattr(self.args, "goal", "")),
            "seed": int(self.random_seed),
            "classifier_scale": float(
                getattr(self.args, "shared_classifier_scale", 0.0)
            ),
            "experimental_linear_cosine_agreement": bool(
                True
            ),
            "mcl_variant": str(getattr(self.args, "mcl_variant", "fedsa")),
            "mcl_hard_margin": float(getattr(self.args, "mcl_hard_margin", 0.5)),
            "new_anchor_mode": str(self.new_anchor_mode),
            "anchor_mode": self.anchor_mode,
            "anchor_codebook": self.anchor_codebook,
            "anchor_loss_components": {
                "mse": bool(float(getattr(self.args, "lamda", 0.0)) > 0.0),
                "mcl": bool(
                    float(getattr(self.args, "margin_contrastive", 0.0)) > 0.0
                ),
                "agreement_weighting": bool(
                    getattr(self.args, "agreement_weighting", True)
                ),
            },
            "semantic_gate_mode": str(self.semantic_gate_mode),
            "dual_disagreement_gate": bool(self.dual_disagreement_gate),
            "dual_disagreement_temperature": float(
                getattr(self.args, "dual_disagreement_temperature", 0.05)
            ),
            "fast_anchor_ema": float(self.fast_anchor_ema),
            "reliability_aggregation": {
                "enabled": bool(self.reliability_weighted_aggregation),
                "temperature": float(self.reliability_aggregation_temperature),
                "geometry": str(self.reliability_aggregation_geometry),
                "mean_weight": float(np.mean(list(self.proto_reliability.values())))
                if self.proto_reliability
                else 1.0,
                "min_weight": float(np.min(list(self.proto_reliability.values())))
                if self.proto_reliability
                else 1.0,
            },
            "prototype_utility": {
                "enabled": bool(self.prototype_utility),
                "local_gate": bool(self.utility_local_gate),
                "anchor_candidate_mix": float(self.anchor_candidate_mix),
                "weight_normalize": bool(
                    getattr(self.args, "utility_weight_normalize", True)
                ),
                "signal_mode": str(self.utility_signal_mode),
                "temperature": float(self.utility_temperature),
                "ema": float(self.utility_ema),
                "mean": float(self.class_utility.mean().item())
                if self.prototype_utility
                else 0.0,
                "positive_rate": float(
                    (self.class_utility > 0).float().mean().item()
                )
                if self.prototype_utility
                else 0.0,
            },
            "utility_validation": {
                "enabled": bool(self.utility_validation_probe),
                **self.utility_validation_metrics,
            },
            "anchor_update_norm": float(self.last_anchor_update_norm),
            "pcw_prediction_source": "classifier_logits",
            "virtual_anchor_supervision": {"mcl_gap": True, "agreement_weighting": False},
            "prototype_trust": self.fast_anchor_proto_trust,
            "raar_version": "nextgap_prior_marginal_class_count_aggregation",
            "raar_enabled": self.raar_enabled,
            "anchor_candidate_mix": self.anchor_candidate_mix,
            "anchor_momentum": self.anchor_momentum,
            "prototype_aggregation": "class_sample_count",
            "mcl_gap_class_cap": True,
            "raar_nextgap": getattr(self, "candidate_gain_stats", None),
            "role_step_clipping": bool(role_mode),
            "active_anchor_update": (
                "semantic_observation_plus_discriminative"
                if self.new_anchor_mode == "semantic_discriminative"
                else "semantic_observation_plus_meta"
                if self.new_anchor_mode == "meta_separated"
                else "boundary_rival"
                if self.new_anchor_mode == "boundary"
                else "class_boundary_margin"
                if class_boundary_mode
                else "prototype_trust_ema"
            ),
            "fixed_anchor": bool(self.fixed_anchor),
            "freeze_semantic_anchor": bool(self.freeze_semantic_anchor),
            "reliability_anchor_source": str(
                getattr(self.args, "reliability_anchor_source", "mcl")
            ),
            "scalar_reliability_state": bool(self.scalar_reliability_state),
            "separate_anchor_roles": bool(
                getattr(self.args, "separate_anchor_roles", True)
            ),
            "round_elapsed_sec": self.current_round_elapsed_sec,
            "total_elapsed_sec": float(self.total_elapsed_sec),
            "communication": {
                "prototype_upload_bytes": int(self.current_proto_upload_bytes),
                "prototype_feature_upload_bytes": int(self.current_proto_feature_upload_bytes),
                "prototype_class_count_upload_bytes": int(self.current_proto_count_upload_bytes),
                "prototype_class_count_encoding": "int64 logical payload",
                "anchor_broadcast_bytes": int(self.current_anchor_broadcast_bytes),
                "anchor_broadcast_roles": anchor_role_count,
                "selected_clients": int(len(self.selected_clients)),
            },
            "test_accuracy": float(self.rs_test_acc[-1])
            if self.rs_test_acc
            else None,
            "test_auc": float(self.rs_test_auc[-1]) if self.rs_test_auc else None,
            "train_loss": float(self.rs_train_loss[-1])
            if self.rs_train_loss
            else None,
            "train": dict(self.last_train_diagnostics),
            "geometry": {
                "rank": float(self.anchor_effective_rank(self.semantic_anchors)),
                "stable_rank": float(
                    self.anchor_effective_rank(self.stable_semantic_anchors)
                ),
                "adaptive_rank": float(
                    self.anchor_effective_rank(
                        getattr(self, "adaptive_anchors", self.stable_semantic_anchors)
                    )
                ),
                "volume_ratio": float(self.anchor_volume_stats()[0]),
                "spectrum_min": float(self.anchor_volume_stats()[1]),
                "stable_volume_ratio": float(
                    self.anchor_volume_stats_for(self.stable_semantic_anchors)[0]
                ),
                "stable_spectrum_min": float(
                    self.anchor_volume_stats_for(self.stable_semantic_anchors)[1]
                ),
                "adaptive_volume_ratio": float(
                    self.anchor_volume_stats_for(
                        getattr(self, "adaptive_anchors", self.stable_semantic_anchors)
                    )[0]
                ),
                "adaptive_spectrum_min": float(
                    self.anchor_volume_stats_for(
                        getattr(self, "adaptive_anchors", self.stable_semantic_anchors)
                    )[1]
                ),
                "semantic_initial_drift": float(
                    self.anchor_initial_drift(self.semantic_anchors)
                ),
                "stable_initial_drift": float(
                    self.anchor_initial_drift(self.stable_semantic_anchors)
                ),
                "adaptive_initial_drift": float(
                    self.anchor_initial_drift(
                        getattr(self, "adaptive_anchors", self.stable_semantic_anchors)
                    )
                ),
            },
            "mapping": {
                name: float(entries[-1])
                for name, entries in self.mapping_diagnostics.items()
                if entries and np.isfinite(entries[-1])
            },
            "new_anchor": {
                key: float(value)
                for key, value in self.new_anchor_metrics.items()
                if isinstance(value, (int, float)) and np.isfinite(value)
            },
            "state_accounting": {
                "independent_recurrent_states": independent_state_count,
                "stable_state": (
                    "server_only_guard"
                    if anchor_mode == "single" and legacy_update_path
                    else "unused_single_control"
                    if anchor_mode == "single"
                    else "compatibility_alias"
                    if role_mode
                    else "ema_state"
                ),
                "semantic_state": (
                    "single_anchor"
                    if anchor_mode == "single"
                    else "observed_class_centers"
                    if role_mode
                    else "adaptive_or_legacy"
                ),
                "adaptive_state": (
                    "class_boundary_margin"
                    if class_boundary_mode
                    else "unused_single_control"
                    if anchor_mode == "single"
                    else "discriminative_or_meta_target"
                    if role_mode
                    else "legacy_fast_target"
                ),
            },
        }
        if hasattr(self, "adaptive_anchors"):
            semantic_float = self.semantic_anchors.float()
            adaptive_float = self.adaptive_anchors.float()
            semantic_adaptive_distance = float(
                (semantic_float - adaptive_float).norm(dim=1).mean().item()
            )
            semantic_adaptive_cosine = float(
                torch.nn.functional.cosine_similarity(
                    semantic_float, adaptive_float, dim=1
                ).mean().item()
            )
            semantic_rank = self.anchor_effective_rank(self.semantic_anchors)
            adaptive_rank = self.anchor_effective_rank(self.adaptive_anchors)
            record["semantic_adaptive_distance"] = semantic_adaptive_distance
            record["adaptive_stable_distance"] = float(
                (self.adaptive_anchors.float() - self.stable_semantic_anchors.float())
                .norm(dim=1)
                .mean()
                .item()
            )
            record["role_separation"] = {
                "semantic_target": "mse",
                "adaptive_target": (
                    "boundary_rival"
                    if self.new_anchor_mode == "boundary"
                    and bool(getattr(self.args, "separate_anchor_roles", True))
                    else "class_boundary_margin"
                    if class_boundary_mode
                    and bool(getattr(self.args, "separate_anchor_roles", True))
                    else "mcl"
                    if self.new_anchor_mode in {
                        "relation_memory",
                        "semantic_discriminative",
                        "observation_validation",
                        "meta_separated",
                        "boundary",
                    }
                    and bool(getattr(self.args, "separate_anchor_roles", True))
                    else (
                        "adaptive_both"
                        if self.new_anchor_mode in {
                            "relation_memory",
                            "semantic_discriminative",
                            "observation_validation",
                            "meta_separated",
                            "boundary",
                        }
                        and self.anchor_mode == "dual"
                        else "legacy_or_single"
                    )
                ),
                "distinct_training_targets": bool(
                    self.anchor_mode == "dual"
                    and bool(getattr(self.args, "separate_anchor_roles", True))
                    and self.new_anchor_mode in {
                        "relation_memory",
                        "semantic_discriminative",
                        "observation_validation",
                        "meta_separated",
                        "boundary",
                        "class_boundary",
                    }
                ),
                "semantic_adaptive_distance": semantic_adaptive_distance,
                "semantic_adaptive_cosine": semantic_adaptive_cosine,
                "adaptive_minus_semantic_rank": float(adaptive_rank - semantic_rank),
                "stable_state": (
                    "server_only_guard"
                    if anchor_mode == "single" and legacy_update_path
                    else "unused_single_control"
                    if anchor_mode == "single"
                    else "compatibility_alias"
                    if role_mode
                    else "independent_ema"
                ),
                "independent_recurrent_states": independent_state_count,
            }
        if self.scalar_reliability_state:
            record["reliability_state"] = {
                "mean": float(self.class_reliability.mean().item()),
                "min": float(self.class_reliability.min().item()),
                "max": float(self.class_reliability.max().item()),
                "mean_age": float(self.class_observation_age.mean().item()),
            }
        with open(self.metrics_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        
    def send_semantic_state(self, clients=None):
        clients = self.clients if clients is None else clients
        anchors = self.semantic_anchors.detach()
        class_boundary_mode = self.new_anchor_mode == "class_boundary"
        role_mode = self.new_anchor_mode in {
            "semantic_discriminative",
            "meta_separated",
            "boundary",
            "class_boundary",
        }
        stable_anchors = self.stable_semantic_anchors.detach()
        single_control = self.anchor_mode == "single"
        adaptive_anchors = getattr(self, "adaptive_anchors", stable_anchors).detach()
        if clients:
            if class_boundary_mode:
                self.current_anchor_broadcast_bytes = int(
                    len(clients)
                    * (self.num_classes * self.feat_dim + self.num_classes)
                    * 4
                )
            else:
                self.current_anchor_broadcast_bytes = int(
                    len(clients)
                    * (1 if single_control else 2 if role_mode else 3)
                    * self.num_classes
                    * self.feat_dim
                    * 4
                )
        for client in clients:
            client.set_semantic_anchors(anchors)
            if single_control:
                # A single-anchor run must not send unused stable/adaptive
                # copies.  This also makes the communication measurement match
                # the actual method being evaluated.
                client.set_stable_semantic_anchors(None)
                if hasattr(client, "set_adaptive_anchors"):
                    client.set_adaptive_anchors(None)
            elif role_mode:
                # The new paths consume semantic for MSE and adaptive for
                # MCL/meta signals.  Clear the legacy slot instead of
                # broadcasting a hidden third anchor.
                client.set_stable_semantic_anchors(None)
                if hasattr(client, "set_adaptive_anchors"):
                    client.set_adaptive_anchors(
                        None if class_boundary_mode else adaptive_anchors
                    )
            else:
                client.set_stable_semantic_anchors(stable_anchors)
                if hasattr(client, "set_adaptive_anchors"):
                    client.set_adaptive_anchors(adaptive_anchors)
            if hasattr(client, "set_class_reliability"):
                client.set_class_reliability(
                    self.class_reliability if self.scalar_reliability_state else None
                )
            if hasattr(client, "set_class_utility"):
                client.set_class_utility(
                    self.class_utility if self.prototype_utility else None
                )
            if hasattr(client, "set_class_boundary_margin"):
                client.set_class_boundary_margin(
                    self.class_boundary_margin
                    if self.new_anchor_mode == "class_boundary"
                    else None
                )

    def require_finite_upload(self, value, name, client=None):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(
                f"FedDSA nonfinite {name}: round={getattr(self, 'current_round', 'unknown')}, "
                f"client={getattr(client, 'id', 'aggregate')}; refusing to update server state"
            )

    def receive_protos(self):
        if hasattr(self, "raw_class_utility_count"):
            self.raw_class_utility_count.zero_()
        self.last_utility_observed = None
        self.last_utility_signal = None
        self.last_utility_observation_count = None
        self.last_utility_client_count = None
        self.last_utility_client_std = None
        self.last_utility_positive_clients = None
        self.last_utility_negative_clients = None
        self.utility_validation_query_count = None
        self.utility_validation_gain_count = None
        self.utility_validation_query_value = None
        self.utility_validation_gain_value = None
        uploaded = []
        records = []
        meta_grads = []
        meta_weights = []
        validation_grads = []
        validation_weights = []
        validation_gains = []
        validation_gain_counts = []
        utility_values = []
        utility_counts = []
        paired_utility_values = []
        paired_utility_counts = []
        paired_validation_values = []
        paired_validation_counts = []
        for client in self.selected_clients:
            client_validation_gain = None
            client_validation_count = None
            protos = getattr(client, "protos", None)
            if protos is None:
                protos = load_item(client.role, "protos", client.save_folder_name)
            if protos:
                for label, proto in protos.items():
                    if proto is not None and not isinstance(proto, list):
                        self.require_finite_upload(torch.as_tensor(proto), f"prototype[{label}]", client)
                counts = getattr(client, "proto_sample_count", {})
                if (not isinstance(counts, dict) or set(counts) != set(protos)
                        or any(type(value) is not int or value <= 0 for value in counts.values())):
                    raise ValueError(f"Client {client.id} must upload a positive actual sample count for every prototype")
                uploaded.append({"client": int(client.id), "protos": protos, "counts": counts})
                records.append((protos, counts))
            meta_grad = getattr(client, "meta_anchor_grad", None)
            if meta_grad is not None:
                grad = torch.as_tensor(meta_grad, dtype=torch.float32).detach().cpu()
                if grad.ndim == 2 and grad.shape == (self.num_classes, self.feat_dim):
                    meta_grads.append(grad)
                    meta_weights.append(float(max(client.train_samples, 1)))
            validation_grad = getattr(client, "validation_anchor_grad", None)
            if validation_grad is not None:
                grad = torch.as_tensor(
                    validation_grad, dtype=torch.float32
                ).detach().cpu()
                if grad.ndim == 2 and grad.shape == (self.num_classes, self.feat_dim):
                    validation_grads.append(grad)
                    validation_weights.append(float(max(client.train_samples, 1)))
            validation_gain = getattr(client, "validation_anchor_gain", None)
            validation_count = getattr(client, "validation_anchor_gain_count", None)
            if validation_gain is not None and validation_count is not None:
                gain = torch.as_tensor(validation_gain, dtype=torch.float32).detach().cpu()
                count = torch.as_tensor(validation_count, dtype=torch.float32).detach().cpu()
                if (
                    gain.ndim == 1
                    and count.ndim == 1
                    and gain.numel() == self.num_classes
                    and count.numel() == self.num_classes
                ):
                    self.require_finite_upload(gain, "validation_gain", client)
                    self.require_finite_upload(count, "validation_count", client)
                    validation_gains.append(gain)
                    validation_gain_counts.append(count.clamp_min(0.0))
                    client_validation_gain = gain
                    client_validation_count = count.clamp_min(0.0)
            utility = getattr(client, "anchor_utility", None)
            utility_count = getattr(client, "anchor_utility_count", None)
            if utility is not None and utility_count is not None:
                utility = torch.as_tensor(utility, dtype=torch.float32).detach().cpu()
                utility_count = torch.as_tensor(
                    utility_count, dtype=torch.float32
                ).detach().cpu()
                if (
                    utility.ndim == 1
                    and utility_count.ndim == 1
                    and utility.numel() == self.num_classes
                    and utility_count.numel() == self.num_classes
                ):
                    self.require_finite_upload(utility, "utility", client)
                    self.require_finite_upload(utility_count, "utility_count", client)
                    utility_values.append(utility)
                    utility_counts.append(utility_count.clamp_min(0.0))
                    if (
                        self.utility_validation_probe
                        and client_validation_gain is not None
                    ):
                        paired_utility_values.append(utility)
                        paired_utility_counts.append(utility_count.clamp_min(0.0))
                        paired_validation_values.append(client_validation_gain)
                        paired_validation_counts.append(client_validation_count)

        self.proto_records = records
        self.current_proto_feature_upload_bytes = int(
            sum(len(protos) * self.feat_dim * 4 for protos, _ in records)
        )
        self.current_proto_count_upload_bytes = sum(len(counts) * 8 for _, counts in records)
        self.current_proto_upload_bytes = self.current_proto_feature_upload_bytes + self.current_proto_count_upload_bytes
        if meta_grads:
            weights = torch.as_tensor(meta_weights, dtype=torch.float32)
            weights = weights / weights.sum().clamp_min(1e-12)
            self.meta_anchor_grads = sum(
                (grad * weight for grad, weight in zip(meta_grads, weights)),
                torch.zeros_like(meta_grads[0]),
            )
            self.meta_anchor_support = float(len(meta_grads))
        else:
            self.meta_anchor_grads = None
            self.meta_anchor_support = 0.0
        if validation_grads:
            weights = torch.as_tensor(validation_weights, dtype=torch.float32)
            weights = weights / weights.sum().clamp_min(1e-12)
            self.validation_anchor_grads = sum(
                (grad * weight for grad, weight in zip(validation_grads, weights)),
                torch.zeros_like(validation_grads[0]),
            )
            self.validation_anchor_support = float(len(validation_grads))
        else:
            self.validation_anchor_grads = None
            self.validation_anchor_support = 0.0
        if validation_gains:
            gain_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            count_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            for gain, count in zip(validation_gains, validation_gain_counts):
                gain_sum += gain * count
                count_sum += count
            self.require_finite_upload(gain_sum, "weighted_validation_sum")
            self.require_finite_upload(count_sum, "validation_count_sum")
            self.validation_anchor_gains = gain_sum / count_sum.clamp_min(1.0)
            self.validation_gain_support = count_sum
        else:
            self.validation_anchor_gains = None
            self.validation_gain_support = None
        self.utility_validation_metrics = {
            "pearson": 0.0,
            "spearman": 0.0,
            "sign_agreement": 0.0,
            "negative_utility_rate": 0.0,
            "class_coverage": 0.0,
            "pairs": 0,
        }
        if (
            self.utility_validation_probe
            and paired_utility_values
            and paired_validation_values
        ):
            utility_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            utility_count = torch.zeros(self.num_classes, dtype=torch.float32)
            gain_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            gain_count = torch.zeros(self.num_classes, dtype=torch.float32)
            for value, count in zip(paired_utility_values, paired_utility_counts):
                utility_sum += value * count
                utility_count += count
            for value, count in zip(
                paired_validation_values, paired_validation_counts
            ):
                gain_sum += value * count
                gain_count += count
            for name, value in (("probe_utility_sum", utility_sum), ("probe_utility_count", utility_count),
                                ("probe_gain_sum", gain_sum), ("probe_gain_count", gain_count)):
                self.require_finite_upload(value, name)
            self.utility_validation_query_count = utility_count.clone()
            self.utility_validation_gain_count = gain_count.clone()
            self.utility_validation_query_value = utility_sum / utility_count.clamp_min(1.0)
            self.utility_validation_gain_value = gain_sum / gain_count.clamp_min(1.0)
            valid = (utility_count > 0) & (gain_count > 0)
            if int(valid.sum().item()) >= 2:
                utility_np = (utility_sum[valid] / utility_count[valid]).numpy()
                gain_np = (gain_sum[valid] / gain_count[valid]).numpy()
                self.utility_validation_metrics["pearson"] = float(
                    np.corrcoef(utility_np, gain_np)[0, 1]
                ) if np.std(utility_np) > 1e-12 and np.std(gain_np) > 1e-12 else 0.0
                utility_rank = np.argsort(np.argsort(utility_np)).astype(float)
                gain_rank = np.argsort(np.argsort(gain_np)).astype(float)
                self.utility_validation_metrics["spearman"] = float(
                    np.corrcoef(utility_rank, gain_rank)[0, 1]
                ) if np.std(utility_rank) > 0 and np.std(gain_rank) > 0 else 0.0
                self.utility_validation_metrics["sign_agreement"] = float(
                    np.mean((utility_np > 0) == (gain_np > 0))
                )
                self.utility_validation_metrics["negative_utility_rate"] = float(
                    np.mean(utility_np < 0)
                )
                self.utility_validation_metrics["class_coverage"] = float(
                    valid.float().mean().item()
                )
                self.utility_validation_metrics["pairs"] = int(valid.sum().item())
        if utility_values:
            utility_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            utility_count_sum = torch.zeros(self.num_classes, dtype=torch.float32)
            for utility, count in zip(utility_values, utility_counts):
                utility_sum += utility * count
                utility_count_sum += count
            self.require_finite_upload(utility_sum, "weighted_utility_sum")
            self.require_finite_upload(utility_count_sum, "utility_count_sum")
            observed = utility_sum / utility_count_sum.clamp_min(1.0)
            self.update_raw_utility(observed, utility_count_sum)
            valid = utility_count_sum > 0
            self.last_utility_observed = observed.detach().clone()
            self.last_utility_observation_count = utility_count_sum.detach().clone()
            client_values = torch.stack(utility_values)
            client_counts = torch.stack(utility_counts)
            client_valid = client_counts > 0
            difference = torch.where(client_valid, client_values - observed, 0.0)
            self.last_utility_client_count = client_valid.sum(dim=0)
            self.last_utility_client_std = (
                (difference.square() * client_counts).sum(dim=0)
                / utility_count_sum.clamp_min(1.0)
            ).clamp_min(0.0).sqrt()
            self.last_utility_positive_clients = (client_valid & (client_values > 0)).sum(dim=0)
            self.last_utility_negative_clients = (client_valid & (client_values < 0)).sum(dim=0)
            if self.prototype_utility and bool(valid.any()):
                values = observed[valid]
                if self.utility_signal_mode == "shuffled":
                    observed = observed.clone()
                    observed[valid] = values[torch.randperm(values.numel())]
                elif self.utility_signal_mode == "uniform":
                    observed = observed.clone()
                    observed[valid] = values.mean()
                elif self.utility_signal_mode == "zscore":
                    observed = observed.clone()
                    scale = values.std(unbiased=False).clamp_min(1e-6)
                    observed[valid] = (values - values.mean()) / scale
            self.last_utility_signal = observed.detach().clone()
            if self.prototype_utility:
                self.require_finite_upload(observed, "transformed_utility")
                valid_device = valid.to(self.device)
                observed_device = observed.to(self.device)
                self.class_utility[valid_device] = (
                    self.class_utility[valid_device] * (1.0 - self.utility_ema)
                    + observed_device[valid_device] * self.utility_ema
                )
            self.class_utility_count = utility_count_sum.to(self.device)
            self.new_anchor_metrics["utility_mean"] = float(
                observed[valid].mean().item()
            ) if bool(valid.any()) else 0.0
            self.new_anchor_metrics["utility_positive_rate"] = float(
                (observed[valid] > 0).float().mean().item()
            ) if bool(valid.any()) else 0.0
        elif self.prototype_utility:
            self.new_anchor_metrics["utility_mean"] = 0.0
            self.new_anchor_metrics["utility_positive_rate"] = 0.0
        self.global_protos = self.aggregate_protos(uploaded)
        save_item(self.global_protos, self.role, "global_protos", self.save_folder_name)

    def aggregate_protos(self, uploaded):
        protos, scatter, support, diagnostics = aggregate_count_prototypes(uploaded, self.feat_dim)
        self.proto_disagreement = scatter
        self.proto_support = support
        self.proto_reliability = {label: 1.0 for label in protos}
        self.prototype_aggregation_diagnostics = diagnostics
        return protos

    @torch.no_grad()
    def update_semantic_anchors(self):
        if not self.global_protos:
            return
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= self.semantic_anchors.size(0):
                continue
            proto = torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1)
            if proto.numel() == self.semantic_anchors.size(1):
                if self.dual_disagreement_gate:
                    self.semantic_anchors[label].lerp_(
                        proto.to(self.semantic_anchors.dtype),
                        float(self.fast_anchor_ema),
                    )
                else:
                    self.semantic_anchors[label].lerp_(
                        proto.to(self.semantic_anchors.dtype), self.anchor_candidate_mix
                    )

    @torch.no_grad()
    def observed_proto_tensors(self):
        """Return current aggregate prototypes and their server support."""

        labels = []
        protos = []
        supports = []
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= self.num_classes:
                continue
            proto = torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1)
            if proto.numel() != self.feat_dim or not torch.isfinite(proto).all():
                continue
            labels.append(label)
            protos.append(torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0))
            supports.append(max(float(self.proto_support.get(label, 1.0)), 1.0))
        if not protos:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, self.feat_dim, dtype=torch.float32, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
            )
        return (
            torch.as_tensor(labels, dtype=torch.long, device=self.device),
            torch.stack(protos, dim=0),
            torch.as_tensor(supports, dtype=torch.float32, device=self.device),
        )

    @torch.no_grad()
    def update_observation_anchors(self):
        """Independently track recent class semantics with a bounded EMA."""

        labels, protos, supports = self.observed_proto_tensors()
        if protos.numel() == 0:
            return labels, protos, supports
        if not self.freeze_semantic_anchor:
            base_alpha = float(self.new_anchor_ema)
            for row, label in enumerate(labels.tolist()):
                support_factor = supports[row] / (supports[row] + 1.0)
                alpha = min(max(base_alpha * float(support_factor.item()), 0.0), 1.0)
                self.semantic_anchors[label].lerp_(protos[row].to(self.semantic_anchors.dtype), alpha)
        return labels, protos, supports

    @torch.no_grad()
    def update_scalar_reliability_state(self):
        """Track evidence quality separately from either vector target.

        The state is deliberately scalar per class.  It combines support,
        cross-client prototype disagreement, and observation staleness.  It
        controls how strongly the adaptive MCL target is consumed by clients;
        it does not create a third prototype or alter the semantic codebook.
        """

        self.class_observation_age.add_(1.0)
        if not self.global_protos:
            decay = torch.exp(-self.class_observation_age / self.reliability_staleness_tau)
            self.class_reliability.mul_(decay).clamp_(min=0.25, max=1.0)
            return

        labels = [
            int(label)
            for label in self.global_protos
            if 0 <= int(label) < self.num_classes
        ]
        if not labels:
            return
        support_values = torch.as_tensor(
            [max(float(self.proto_support.get(label, 0.0)), 0.0) for label in labels],
            device=self.device,
            dtype=torch.float32,
        )
        disagreement_values = torch.as_tensor(
            [max(float(self.proto_disagreement.get(label, 0.0)), 0.0) for label in labels],
            device=self.device,
            dtype=torch.float32,
        )
        positive = disagreement_values[disagreement_values > 0]
        scale = positive.median() if positive.numel() else disagreement_values.new_tensor(1.0)
        support_conf = support_values / (support_values + 1.0)
        agreement_conf = torch.exp(-disagreement_values / scale.clamp_min(1e-6))
        target = (support_conf * agreement_conf).clamp(0.25, 1.0)
        for row, label in enumerate(labels):
            self.class_reliability[label].lerp_(
                target[row], float(self.reliability_state_alpha)
            )
            self.class_observation_age[label] = 0.0

        stale = torch.exp(-self.class_observation_age / self.reliability_staleness_tau)
        self.class_reliability.mul_(stale).clamp_(min=0.25, max=1.0)
        self.new_anchor_metrics["reliability_state_mean"] = float(
            self.class_reliability.mean().item()
        )
        self.new_anchor_metrics["reliability_state_min"] = float(
            self.class_reliability.min().item()
        )
        self.new_anchor_metrics["reliability_state_max"] = float(
            self.class_reliability.max().item()
        )

    @torch.no_grad()
    def clip_anchor_steps(self, old, candidate, scale):
        """Bound a new anchor role's movement without coupling it to the old gate."""

        if old.shape != candidate.shape or old.ndim != 2 or old.size(0) < 2:
            return candidate
        dist = torch.cdist(old.float(), old.float(), p=2)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.min(dim=1).values
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.ones_like(nearest))
        max_step = nearest.clamp_min(1e-3) * float(max(scale, 0.0))
        delta = candidate.float() - old.float()
        delta_norm = delta.norm(dim=1).clamp_min(1e-12)
        factor = torch.minimum(torch.ones_like(delta_norm), max_step / delta_norm)
        return (old.float() + delta * factor.view(-1, 1)).to(candidate.dtype)

    @torch.no_grad()
    def update_relation_memory_targets(self, labels, protos, supports):
        """Update only class-pair relations supported by current observations."""

        if labels.numel() < 2 or self.relation_targets is None:
            return
        for left in range(labels.numel()):
            i = int(labels[left].item())
            for right in range(left + 1, labels.numel()):
                j = int(labels[right].item())
                distance = (protos[left] - protos[right]).pow(2).sum()
                pair_support = torch.minimum(supports[left], supports[right])
                alpha = float(self.relation_memory_lr) * float(
                    (pair_support / (pair_support + 1.0)).item()
                )
                alpha = min(max(alpha, 0.0), 1.0)
                if self.relation_confidence[i, j].item() <= 0.0:
                    self.relation_targets[i, j] = distance
                    self.relation_targets[j, i] = distance
                    self.relation_confidence[i, j] = max(alpha, 1e-3)
                    self.relation_confidence[j, i] = max(alpha, 1e-3)
                else:
                    self.relation_targets[i, j].lerp_(distance, alpha)
                    self.relation_targets[j, i].copy_(self.relation_targets[i, j])
                    confidence = min(
                        1.0,
                        float(self.relation_confidence[i, j].item()) + alpha,
                    )
                    self.relation_confidence[i, j] = confidence
                    self.relation_confidence[j, i] = confidence
        self.relation_observed[labels] = True

    @torch.no_grad()
    def update_relation_memory_anchors(self):
        """Fit a geometry anchor to remembered pairwise class relations."""

        labels, protos, supports = self.update_observation_anchors()
        self.update_relation_memory_targets(labels, protos, supports)
        old = self.stable_semantic_anchors.detach().float().clone()
        if self.relation_targets is None:
            return
        relation_mask = torch.triu(self.relation_confidence, diagonal=1)
        relation_mass = relation_mask.sum()
        observed_weight = torch.zeros(self.num_classes, device=self.device)
        if labels.numel() > 0:
            observed_weight[labels] = supports / supports.max().clamp_min(1.0)

        with torch.enable_grad():
            geometry = old.detach().clone().requires_grad_(True)
            for _ in range(2):
                distances = torch.cdist(geometry, geometry, p=2).pow(2)
                if relation_mass.item() > 0:
                    relation_loss = (
                        (distances - self.relation_targets).pow(2) * relation_mask
                    ).sum() / relation_mass.clamp_min(1e-6)
                else:
                    relation_loss = geometry.sum() * 0.0
                if labels.numel() > 0:
                    observed_loss = (
                        (geometry.index_select(0, labels) - self.semantic_anchors[labels].float())
                        .pow(2)
                        .sum(dim=1)
                        * observed_weight[labels]
                    ).mean()
                else:
                    observed_loss = geometry.sum() * 0.0
                loss = relation_loss + float(self.relation_observation_weight) * observed_loss
                grad = torch.autograd.grad(loss, geometry, allow_unused=False)[0]
                geometry = geometry - float(self.relation_anchor_lr) * grad
                geometry = geometry.detach().requires_grad_(True)
            candidate = geometry.detach()
        self.stable_semantic_anchors.copy_(
            self.clip_anchor_steps(old, candidate, self.relation_step_scale)
        )
        if relation_mass.item() > 0:
            current = torch.cdist(self.stable_semantic_anchors.float(), self.stable_semantic_anchors.float(), p=2).pow(2)
            self.new_anchor_metrics["relation_error"] = float(
                ((current - self.relation_targets).abs() * relation_mask).sum().item()
                / relation_mass.clamp_min(1e-6).item()
            )
        self.new_anchor_metrics["relation_coverage"] = float(
            (relation_mask > 0).sum().item() / max(self.num_classes * (self.num_classes - 1) / 2, 1)
        )

    @torch.no_grad()
    def update_semantic_discriminative_anchors(self):
        """Keep a semantic prototype EMA and optimize a separate boundary anchor."""

        labels, protos, supports = self.update_observation_anchors()
        old = self.adaptive_anchors.detach().float().clone()
        if labels.numel() == 0:
            return
        if not self.semantic_discriminative_initialized:
            # Match meta-separated's initialization: the discriminative
            # target must not start from the random bank while the semantic
            # target has already incorporated the first observations.
            self.adaptive_anchors.copy_(self.semantic_anchors)
            self.stable_semantic_anchors.copy_(self.semantic_anchors)
            self.semantic_discriminative_initialized = True
            self.new_anchor_metrics["discriminative_warm_start"] = 1.0
            self.new_anchor_metrics["discriminative_grad_norm"] = 0.0
            self.new_anchor_metrics["discriminative_step_norm"] = float(
                (self.adaptive_anchors.float() - old.float()).norm(dim=1).mean().item()
            )
            semantic_pred = torch.cdist(
                protos, self.semantic_anchors.float(), p=2
            ).argmin(dim=1)
            self.new_anchor_metrics["semantic_proto_acc"] = float(
                semantic_pred.eq(labels).float().mean().item()
            )
            self.new_anchor_metrics["discriminative_proto_acc"] = float(
                self.new_anchor_metrics["semantic_proto_acc"]
            )
            self.new_anchor_metrics["discriminative_loss"] = 0.0
            return
        if self.anchor_mode.lower() == "single":
            # The single control has one semantic target.  Do not spend
            # server compute on an adaptive state that no client consumes;
            # keep a synchronized copy only for diagnostics.
            self.adaptive_anchors.copy_(self.semantic_anchors)
            self.stable_semantic_anchors.copy_(self.semantic_anchors)
            semantic_pred = torch.cdist(
                protos, self.semantic_anchors.float(), p=2
            ).argmin(dim=1)
            self.new_anchor_metrics["semantic_proto_acc"] = float(
                semantic_pred.eq(labels).float().mean().item()
            )
            self.new_anchor_metrics["discriminative_proto_acc"] = float(
                self.new_anchor_metrics["semantic_proto_acc"]
            )
            self.new_anchor_metrics["discriminative_loss"] = 0.0
            self.new_anchor_metrics["discriminative_grad_norm"] = 0.0
            self.new_anchor_metrics["discriminative_step_norm"] = 0.0
            return
        with torch.enable_grad():
            discriminative = old.detach().clone().requires_grad_(True)
            if self.discriminative_metric == "cosine":
                # Match the fixed cosine classifier scale used by the local
                # heads.  This avoids raw Euclidean-distance CE saturating
                # when the observed prototypes are already correctly ranked.
                proto_norm = torch.nn.functional.normalize(protos.float(), p=2, dim=1)
                anchor_norm = torch.nn.functional.normalize(discriminative, p=2, dim=1)
                scale = float(getattr(self.args, "shared_classifier_scale", 5.0))
                logits = scale * (proto_norm @ anchor_norm.t())
                task_loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                distances = torch.cdist(protos, discriminative, p=2).pow(2)
                task_loss = torch.nn.functional.cross_entropy(-distances, labels)
            reg_loss = (discriminative - old).pow(2).mean()
            codebook_loss = (discriminative - self.initial_semantic_anchors.float()).pow(2).mean()
            loss = (
                task_loss
                + float(self.discriminative_anchor_reg) * reg_loss
                + float(self.discriminative_codebook_reg) * codebook_loss
            )
            grad = torch.autograd.grad(loss, discriminative, allow_unused=False)[0]
            candidate = discriminative - float(self.discriminative_anchor_lr) * grad
        candidate = self.clip_anchor_steps(old, candidate.detach(), self.discriminative_step_scale)
        self.adaptive_anchors.copy_(candidate)
        # Keep the legacy stable field synchronized for diagnostics and older
        # server utilities; clients in this mode consume adaptive_anchors.
        self.stable_semantic_anchors.copy_(self.adaptive_anchors)
        semantic_pred = torch.cdist(protos, self.semantic_anchors.float(), p=2).argmin(dim=1)
        discrim_pred = torch.cdist(protos, self.adaptive_anchors.float(), p=2).argmin(dim=1)
        self.new_anchor_metrics["semantic_proto_acc"] = float(semantic_pred.eq(labels).float().mean().item())
        self.new_anchor_metrics["discriminative_proto_acc"] = float(discrim_pred.eq(labels).float().mean().item())
        self.new_anchor_metrics["discriminative_loss"] = float(task_loss.detach().item())
        self.new_anchor_metrics["discriminative_metric"] = self.discriminative_metric
        self.new_anchor_metrics["discriminative_codebook_reg_loss"] = float(codebook_loss.detach().item())
        self.new_anchor_metrics["discriminative_grad_norm"] = float(grad.detach().norm().item())
        self.new_anchor_metrics["discriminative_step_norm"] = float(
            (candidate.float() - old.float()).norm(dim=1).mean().item()
        )

    @torch.no_grad()
    def update_observation_validation_anchors(self):
        """Update observations quickly; accept movement using train-only query gradients."""

        self.update_observation_anchors()
        old = self.adaptive_anchors.detach().float().clone()
        if self.validation_anchor_grads is None:
            self.new_anchor_metrics["validation_grad_norm"] = 0.0
            return
        grad = torch.as_tensor(
            self.validation_anchor_grads, device=self.device, dtype=torch.float32
        )
        if grad.shape != old.shape:
            return
        candidate = old - float(self.validation_anchor_lr) * grad
        candidate = self.clip_anchor_steps(old, candidate, self.validation_step_scale)
        self.adaptive_anchors.copy_(candidate)
        self.stable_semantic_anchors.copy_(self.adaptive_anchors)
        self.new_anchor_metrics["validation_grad_norm"] = float(grad.norm().item())

    @torch.no_grad()
    def update_boundary_anchors(self):
        """Track a rival prototype per class for boundary-only supervision."""

        labels, protos, supports = self.update_observation_anchors()
        if protos.numel() == 0:
            return
        self.adaptive_anchors.copy_(self.semantic_anchors)
        if protos.size(0) < 2:
            return

        distances = torch.cdist(protos.float(), protos.float(), p=2)
        distances.fill_diagonal_(float("inf"))
        rival_rows = distances.argmin(dim=1)
        for row, label in enumerate(labels.tolist()):
            rival = int(rival_rows[row].item())
            candidate = protos[rival].to(self.adaptive_anchors.dtype)
            self.adaptive_anchors[label].lerp_(
                candidate,
                float(self.boundary_anchor_ema),
            )
        self.stable_semantic_anchors.copy_(self.semantic_anchors)
        self.new_anchor_metrics["boundary_active_classes"] = float(labels.numel())
        self.new_anchor_metrics["boundary_mean_step"] = float(
            (self.adaptive_anchors.float() - self.semantic_anchors.float())
            .norm(dim=1)
            .mean()
            .item()
        )

    @torch.no_grad()
    def update_class_boundary_state(self):
        """Track a class-wise margin target without moving a second vector bank."""

        labels, protos, supports = self.update_observation_anchors()
        self.adaptive_anchors.copy_(self.semantic_anchors)
        self.stable_semantic_anchors.copy_(self.semantic_anchors)
        if labels.numel() < 2:
            self.new_anchor_metrics["class_boundary_active_classes"] = float(
                labels.numel()
            )
            return

        semantic = self.semantic_anchors.float()
        distances = torch.cdist(protos.float(), protos.float(), p=2)
        positive = (protos.float() - semantic.index_select(0, labels)).norm(dim=1)
        distances.fill_diagonal_(float("inf"))
        observed_margin = distances.min(dim=1).values - positive

        base_margin = max(float(getattr(self.args, "mcl_hard_margin", 0.5)), 0.0)
        target_margin = (
            base_margin + (base_margin - observed_margin).clamp_min(0.0)
        ).clamp(self.class_boundary_min, self.class_boundary_max)
        support_factor = supports / (supports + 1.0)
        alpha = (
            float(self.class_boundary_ema) * support_factor
        ).clamp(0.0, 1.0)
        old_margin = self.class_boundary_margin.index_select(0, labels).clone()
        updated = old_margin + alpha * (target_margin - old_margin)
        self.class_boundary_margin[labels] = updated

        self.new_anchor_metrics["class_boundary_active_classes"] = float(
            labels.numel()
        )
        self.new_anchor_metrics["class_boundary_mean"] = float(
            self.class_boundary_margin.mean().item()
        )
        self.new_anchor_metrics["class_boundary_min"] = float(
            self.class_boundary_margin.min().item()
        )
        self.new_anchor_metrics["class_boundary_max"] = float(
            self.class_boundary_margin.max().item()
        )
        self.new_anchor_metrics["class_boundary_observed_margin"] = float(
            observed_margin.mean().item()
        )
        self.new_anchor_metrics["class_boundary_step"] = float(
            (updated - old_margin).abs().mean().item()
        )

    @torch.no_grad()
    def update_new_anchor_mode(self):
        self.new_anchor_metrics = {}
        if self.new_anchor_mode == "relation_memory":
            self.update_relation_memory_anchors()
        elif self.new_anchor_mode == "semantic_discriminative":
            self.update_semantic_discriminative_anchors()
        elif self.new_anchor_mode == "observation_validation":
            self.update_observation_validation_anchors()
        elif self.new_anchor_mode == "meta_separated":
            self.update_meta_separated_anchors()
        elif self.new_anchor_mode == "boundary":
            self.update_boundary_anchors()
        elif self.new_anchor_mode == "class_boundary":
            self.update_class_boundary_state()

    @torch.no_grad()
    def update_meta_separated_anchors(self):
        """Keep semantic observations and meta-learned discrimination separate.

        The semantic state follows currently observed class centers.  The
        adaptive state is updated only by the train-only support/query
        meta-gradient and is used by clients for MCL.  This makes the two
        anchors store different signals instead of being two aliases of one
        EMA state.
        """

        self.update_observation_anchors()
        old = self.adaptive_anchors.detach().float().clone()
        self.new_anchor_metrics = {
            "meta_grad_norm": 0.0,
            "meta_step_norm": 0.0,
            "meta_clients": float(self.meta_anchor_support),
            "meta_warm_start": 0.0,
        }
        if self.anchor_mode.lower() == "single":
            # The single control does not consume a meta-learned target.
            # Keep diagnostic aliases synchronized and avoid an unused
            # support/query update that would make the control unfair.
            self.adaptive_anchors.copy_(self.semantic_anchors)
            self.stable_semantic_anchors.copy_(self.semantic_anchors)
            self.meta_separated_initialized = True
            self.new_anchor_metrics["meta_warm_start"] = 0.0
            self.new_anchor_metrics["meta_step_norm"] = float(
                (self.adaptive_anchors.float() - old).norm(dim=1).mean().item()
            )
            return
        if not self.meta_separated_initialized:
            # The initial random bank is not a useful discriminative target.
            # Align the adaptive role once with the observed semantic bank;
            # subsequent rounds are driven by the meta-gradient only.
            self.adaptive_anchors.copy_(self.semantic_anchors)
            self.meta_separated_initialized = True
            self.new_anchor_metrics["meta_step_norm"] = float(
                (self.adaptive_anchors.float() - old).norm(dim=1).mean().item()
            )
            self.new_anchor_metrics["meta_warm_start"] = 1.0
            self.stable_semantic_anchors.copy_(self.semantic_anchors)
            return
        if self.meta_anchor_grads is not None:
            grad = torch.as_tensor(
                self.meta_anchor_grads,
                device=self.device,
                dtype=torch.float32,
            )
            if grad.shape == old.shape:
                raw = old - float(self.meta_anchor_lr) * grad
                candidate = self.clip_anchor_steps(
                    old, raw, self.meta_step_scale
                )
                self.adaptive_anchors.copy_(candidate)
                self.new_anchor_metrics["meta_grad_norm"] = float(grad.norm().item())
                self.new_anchor_metrics["meta_step_norm"] = float(
                    (candidate - old).norm(dim=1).mean().item()
                )

        # Keep the legacy field as a semantic diagnostic alias.  Clients in
        # this mode use adaptive_anchors explicitly, so this assignment does
        # not collapse the two training roles.
        self.stable_semantic_anchors.copy_(self.semantic_anchors)

    @torch.no_grad()
    def update_meta_semantic_anchors(self):
        """Move adaptive anchors using client query-loss gradients in anchor space."""

        if self.meta_anchor_grads is None:
            return
        grad = torch.as_tensor(
            self.meta_anchor_grads,
            device=self.device,
            dtype=self.semantic_anchors.dtype,
        )
        if grad.shape != self.semantic_anchors.shape:
            return
        self.semantic_anchors.add_(grad, alpha=-float(self.meta_anchor_lr))

    @torch.no_grad()
    def update_stable_anchors(self):
        stable_weight = 1.0 - self.anchor_momentum
        self.stable_semantic_anchors.mul_(self.anchor_momentum).add_(
            self.semantic_anchors.detach(),
            alpha=stable_weight,
        )

    @torch.no_grad()
    def apply_prototype_trust(self, anchor_before=None):
        candidate = self.semantic_anchors.detach().float()
        if not self.raar_enabled:
            self.last_prototype_trust_raw = torch.ones(candidate.size(0), device=candidate.device)
            self.last_prototype_trust_coefficient = self.last_prototype_trust_raw.clone()
            return
        if anchor_before is None:
            raise ValueError("RAAR requires the pre-update anchor bank")
        self.apply_raar_prior_refinement(anchor_before.detach().float(), candidate)

    @torch.no_grad()
    def candidate_update_confidence(self, stable, candidate, prior_beta):
        scores, masks, variances, weights, audits = [], [], [], [], []
        for client in self.selected_clients:
            cache = getattr(client, "anchor_candidate_evidence", None)
            if (cache is None or cache.get("round") != self.current_round
                    or not client.candidate_next_gap or cache.get("candidate_next_gap") is not True):
                raise RuntimeError("NextGap RAAR requires fresh evidence from every selected client")
            evidence = marginal_raar_direction_scores(
                cache["support_features"], cache["support_labels"],
                cache["query_features"], cache["query_labels"],
                client.model.shared_projector, client.model.shared_head, stable, candidate,
                prior_beta, client.utility_inner_lr, client.lamda, client.mcl, cache["mcl_gap"],
                max_weight=self.candidate_trust_max_weight, mcl_gap_class_cap=True,
                **candidate_next_gap_kwargs(client, cache))
            audit = evidence.get("next_gap_point_audit")
            if not audit or len(audit["raw_gaps"]) != 3:
                raise RuntimeError("NextGap B3 point audit is missing")
            audits.append({"client": int(client.id), **audit})
            scores.append(evidence["segment_scores"])
            masks.append(evidence["segment_valid"])
            variances.append(evidence["segment_query_noise_variance"])
            weights.append(evidence["query_samples"])
        shape = (0, 3, stable.size(0))
        scores = torch.stack(scores) if scores else stable.new_empty(shape)
        masks = torch.stack(masks) if masks else torch.empty(shape, device=stable.device, dtype=torch.bool)
        variances = torch.stack(variances) if variances else stable.new_empty(shape)
        previous = getattr(self, "candidate_noise_state", [None] * 3)
        estimates = [cohort_candidate_confidence(
            scores[:, point], masks[:, point], variances[:, point],
            torch.tensor(weights, device=stable.device), self.num_clients,
            previous_state=previous[point], ema=self.utility_ema) for point in range(3)]
        self.candidate_noise_state = [value["state"] for value in estimates]
        proposal = propose_marginal_raar_beta(prior_beta, estimates, self.candidate_trust_max_weight)
        self.candidate_gain_stats = {
            **proposal["summary"], "available_clients": len(weights),
            "candidate_trust_scoring": "prior_marginal",
            "confidence_noise_model": "query_split_cohort",
            "score_query_batch": 3, "acceptance_query_batch": 2,
            "mcl_gap_included": True, "mcl_gap_class_cap": True,
            "agreement_in_virtual_update": False, "utility_weights_in_virtual_update": False,
            "quality_mean": [value["mean"].cpu().tolist() for value in estimates],
            "standard_error": [value["standard_error"].cpu().tolist() for value in estimates],
            "current_quality_mean": [value["current_mean"].cpu().tolist() for value in estimates],
            "current_standard_error": [value["current_standard_error"].cpu().tolist() for value in estimates],
            "effective_clients": [value["effective_clients"].cpu().tolist() for value in estimates],
            "favorable_evidence": proposal["favorable_evidence"].cpu().tolist(),
            "adverse_evidence": proposal["adverse_evidence"].cpu().tolist(),
            "candidate_next_gap_score_audit": {"clients": audits, "full_cohort": True,
                "query_batch": 3, "bank_dependent_global_gap_gradient": True},
            "evidence_role": "training signal; conditional query uncertainty, not a calibrated full-training interval",
            # shortcut: cached oracle calls omit network payloads; account for them before cost comparisons.
            "oracle_communication": {"accounted": False,
                "scope": "Simulator accesses cached client features and local projector/head; existing payload totals exclude oracle calls"},
        }
        for key in ("query_mean_variance", "client_sampling_variance", "temporal_mean_variance",
                    "negative_mass", "observed_population_fraction"):
            self.candidate_gain_stats[key] = [value[key].cpu().tolist() for value in estimates]
        return proposal["beta"].to(stable)

    @torch.no_grad()
    def apply_raar_prior_refinement(self, current, candidate):
        stable = self.stable_semantic_anchors.detach().float()
        prior = self.prototype_trust(candidate - stable, stable).detach()
        observed = torch.zeros_like(prior, dtype=torch.bool)
        for label in self.global_protos:
            if 0 <= int(label) < prior.numel():
                observed[int(label)] = True
        score_origin = stable.clone()
        score_origin[~observed] = candidate[~observed]
        proposed = self.candidate_update_confidence(score_origin, candidate, prior)
        proposed = torch.where(observed, proposed, prior)
        acceptance = CachedRAARQueries(self.selected_clients, self.current_round, stable,
                                      candidate, self.num_clients, "allocation")
        decision = validate_raar_proposal(prior, proposed, acceptance.evaluate,
                                          self.candidate_trust_max_weight)
        beta = decision["beta"]
        updated, components = apply_raar_beta(current, candidate, stable, beta, self.anchor_candidate_mix)
        self.last_prototype_trust_raw = prior.clone()
        self.last_prototype_trust_coefficient = beta.clone()
        self.candidate_gain_stats.update(
            proposal_only=False, original_geometry_beta=prior.cpu().tolist(),
            proposed_beta=proposed.cpu().tolist(), applied_beta=beta.cpu().tolist(),
            effective_prototype_weights=(self.anchor_candidate_mix * beta).cpu().tolist(),
            observed_prototype_classes=int(observed.sum()),
            finite_acceptance={**{key: value for key, value in decision.items() if key != "beta"},
                "query_batch": 2, "uses_test_data": False,
                "bank_evaluations": acceptance.bank_evaluations},
            component_audit={
                "forward_mean_l2": float(components["forward_component"].norm(dim=1).mean()),
                "stable_return_mean_l2": float(components["return_component"].norm(dim=1).mean()),
                "actual_step_mean_l2": float((updated - current).norm(dim=1).mean()),
                "decomposition_max_abs_l2_error": components["decomposition_max_abs_l2_error"]},
            same_state_movement_comparison=compare_raar_movement(
                current, candidate, stable, prior, beta, self.anchor_candidate_mix),
            stable_return_retained=True, coefficient_one_is_fixed_ema=True,
            prototype_aggregation="class_sample_count")
        if getattr(self.args, "raar_diagnostics", False):
            diagnostic = CachedRAARQueries(self.selected_clients, self.current_round, stable,
                                          candidate, self.num_clients, "diagnostic")
            self.candidate_gain_stats["independent_refinement_diagnostic"] = {
                **diagnostic.evaluate(beta, prior), "feeds_update": False, "query_batch": 4}
        self.semantic_anchors.copy_(updated.to(self.semantic_anchors))

    @torch.no_grad()
    def prototype_trust(self, delta, stable):
        num_classes = delta.size(0)
        device = delta.device
        beta = torch.ones(num_classes, device=device, dtype=torch.float32)
        if not self.raar_enabled:
            self.last_base_prototype_trust = beta.clone()
            self.last_trust_utility_multiplier = beta.clone()
            return beta

        signal = delta.float().pow(2).sum(dim=1).clamp_min(0.0)

        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= num_classes:
                continue
            proto = torch.as_tensor(proto, device=device, dtype=torch.float32).view(-1)
            if proto.numel() == stable.size(1):
                signal[label] = (proto - stable[label]).pow(2).sum().clamp_min(0.0)

        has_stats = torch.zeros(num_classes, device=device, dtype=torch.bool)
        noise = torch.zeros(num_classes, device=device, dtype=torch.float32)
        support = torch.zeros(num_classes, device=device, dtype=torch.float32)
        for label, value in self.proto_disagreement.items():
            label = int(label)
            if 0 <= label < num_classes:
                noise[label] = max(float(value), 0.0) * (self.fast_radius_rho ** 2)
                support[label] = max(float(self.proto_support.get(label, 0.0)), 0.0)
                has_stats[label] = True

        if torch.any(has_stats):
            support_trust = support / (support + 1.0)
            harm = self.negative_raw_utility(signal)
            uncertainty = noise + self.utility_trust_penalty * signal * harm
            confidence = signal / (signal + uncertainty + 1e-12) * support_trust
            beta[has_stats] = confidence.clamp(0, 1)[has_stats]
            # Diagnostic decomposition only; there is no independently configurable trust branch.
            base = torch.ones_like(beta)
            base[has_stats] = (signal / (signal + noise + 1e-12) * support_trust)[has_stats]
            multiplier = torch.ones_like(beta)
            positive = has_stats & (base > 0)
            multiplier[positive] = beta[positive] / base[positive]
            self.last_base_prototype_trust = base
            self.last_trust_utility_multiplier = multiplier
        else:
            self.last_base_prototype_trust = beta.clone()
            self.last_trust_utility_multiplier = beta.clone()
        return beta


    @torch.no_grad()
    def update_raw_utility(self, observed, count):
        """Keep a separate EMA in loss units, before any z-score or shuffle."""
        if not self.prototype_utility and not self.raar_enabled:
            return
        self.require_finite_upload(observed, "raw_utility")
        self.require_finite_upload(count, "raw_utility_count")
        if not hasattr(self, "raw_class_utility"):
            self.raw_class_utility = torch.zeros(self.num_classes, device=self.device)
        observed = observed.detach().to(device=self.device, dtype=torch.float32)
        count = count.detach().to(device=self.device, dtype=torch.float32)
        valid = count > 0
        self.raw_class_utility[valid] = (
            self.raw_class_utility[valid] * (1 - self.utility_ema)
            + observed[valid] * self.utility_ema
        )
        self.raw_class_utility_count = count.clone()
        if not hasattr(self, "raw_class_utility_total_count"):
            self.raw_class_utility_total_count = torch.zeros_like(count)
        self.raw_class_utility_total_count += count


    @torch.no_grad()
    def utility_for_update(self):
        """Change score-to-class assignment without changing score estimation or training RNG."""
        raw = self.raw_class_utility
        count = self.raw_class_utility_count
        applied = raw.detach().clone()
        sources = torch.arange(raw.numel(), device=raw.device)
        valid = torch.nonzero(count > 0, as_tuple=False).view(-1)
        mode = getattr(self, "utility_signal_mode", "observed")
        if mode == "shuffled" and valid.numel() > 1:
            generator = torch.Generator(device='cpu')
            generator.manual_seed((0x5A17 + 1000003 * int(self.random_seed) + int(self.current_round)) % (2 ** 63 - 1))
            permutation = torch.randperm(valid.numel(), generator=generator).to(raw.device)
            source = valid.index_select(0, permutation)
            applied[valid] = raw.index_select(0, source)
            sources[valid] = source
        elif mode == "uniform" and valid.numel():
            # Preserve the mean of the eligible class scores, while removing
            # their class-specific variation. Missing-evidence masking remains unchanged.
            applied.fill_(raw.index_select(0, valid).mean())
            sources.fill_(-1)
        elif mode == "geometry":
            # Explicit component ablation: utility is measured for diagnostics
            # but contributes zero uncertainty to the otherwise unchanged rule.
            applied.zero_()
            sources.fill_(-2)
        self.last_applied_raw_utility = applied
        self.last_utility_source_class = sources
        return applied


    @torch.no_grad()
    def negative_raw_utility(self, reference):
        """Task-harm evidence in the same round; missing observations are neutral."""
        harm = torch.zeros_like(reference)
        raw = getattr(self, "raw_class_utility", None)
        count = getattr(self, "raw_class_utility_count", None)
        if raw is None or count is None:
            return harm
        self.require_finite_upload(raw, "raw_utility_ema")
        self.require_finite_upload(count, "raw_utility_count")
        raw = self.utility_for_update().to(device=reference.device, dtype=reference.dtype)
        valid = count.to(reference.device) > 0
        harm[valid] = (-torch.tanh(raw[valid] / self.utility_trust_temperature)).clamp_min(0)
        return harm

    @torch.no_grad()
    def set_trusted_proto_memory(self, memory):
        """Replace the historical prototype memory with detached CPU copies.

        ``memory`` may map a class to one prototype or to a list of entry
        dictionaries.  Each dictionary can contain ``proto``, ``weight``,
        ``support`` and ``confidence`` fields.  Keeping the stored tensors on
        CPU prevents the memory from silently increasing the server's GPU
        footprint between rounds.
        """

        normalized = {}
        if memory is None:
            self.trusted_proto_memory = normalized
            return

        for label, raw_entries in memory.items():
            label = int(label)
            if label < 0 or label >= self.num_classes:
                continue
            if isinstance(raw_entries, dict) and "proto" in raw_entries:
                raw_entries = [raw_entries]
            elif torch.is_tensor(raw_entries) or isinstance(raw_entries, (list, tuple)):
                if torch.is_tensor(raw_entries) and raw_entries.ndim == 1:
                    raw_entries = [{"proto": raw_entries}]
                elif isinstance(raw_entries, (list, tuple)) and raw_entries and not isinstance(
                    raw_entries[0], dict
                ):
                    raw_entries = [{"proto": raw_entries}]
            else:
                raw_entries = []

            entries = []
            for entry in raw_entries:
                if isinstance(entry, dict):
                    proto = entry.get("proto")
                    weight = entry.get("weight", 1.0)
                    support = entry.get("support", 0.0)
                    confidence = entry.get("confidence", 1.0)
                else:
                    proto = entry
                    weight = 1.0
                    support = 0.0
                    confidence = 1.0
                if proto is None:
                    continue
                proto = torch.as_tensor(proto, dtype=torch.float32).detach().view(-1).cpu()
                if proto.numel() != self.feat_dim or not torch.isfinite(proto).all():
                    continue
                entries.append(
                    {
                        "proto": proto.clone(),
                        "weight": max(float(weight), 0.0),
                        "support": max(float(support), 0.0),
                        "confidence": min(max(float(confidence), 0.0), 1.0),
                    }
                )
            if entries:
                entries.sort(key=lambda item: item["weight"], reverse=True)
                normalized[label] = entries[: self.memory_max_per_class]
        self.trusted_proto_memory = normalized

    @torch.no_grad()
    def memory_entry_tensors(self, device):
        protos = []
        labels = []
        weights = []
        for label, entries in self.trusted_proto_memory.items():
            for entry in entries:
                proto = torch.as_tensor(entry["proto"], device=device, dtype=torch.float32).view(-1)
                if proto.numel() != self.feat_dim:
                    continue
                protos.append(proto)
                labels.append(int(label))
                weights.append(max(float(entry.get("weight", 1.0)), 0.0))
        if not protos:
            return (
                torch.empty(0, self.feat_dim, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, device=device),
            )
        return (
            torch.stack(protos, dim=0),
            torch.as_tensor(labels, dtype=torch.long, device=device),
            torch.as_tensor(weights, dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def trusted_memory_metrics(self, old_anchors, new_anchors):
        """Measure discriminative preservation on the stored prototype memory."""

        memory_protos, memory_labels, memory_weights = self.memory_entry_tensors(
            new_anchors.device
        )
        if memory_protos.size(0) == 0:
            return {
                "kl": 0.0,
                "old_confidence": 0.0,
                "new_confidence": 0.0,
                "old_accuracy": 0.0,
                "new_accuracy": 0.0,
                "count": 0,
            }

        temperature = float(self.memory_temperature)
        old_dist = torch.cdist(memory_protos, old_anchors, p=2).pow(2)
        new_dist = torch.cdist(memory_protos, new_anchors, p=2).pow(2)
        old_log_prob = torch.log_softmax(-old_dist / temperature, dim=1)
        new_log_prob = torch.log_softmax(-new_dist / temperature, dim=1)
        old_prob = old_log_prob.exp()
        new_prob = new_log_prob.exp()
        row = torch.arange(memory_labels.numel(), device=new_anchors.device)
        weights = memory_weights.clamp_min(0.0)
        weight_sum = weights.sum().clamp_min(1e-12)
        kl = (weights * (old_prob * (old_log_prob - new_log_prob)).sum(dim=1)).sum() / weight_sum
        old_confidence = (weights * old_prob[row, memory_labels]).sum() / weight_sum
        new_confidence = (weights * new_prob[row, memory_labels]).sum() / weight_sum
        old_accuracy = (weights * old_prob.argmax(dim=1).eq(memory_labels).float()).sum() / weight_sum
        new_accuracy = (weights * new_prob.argmax(dim=1).eq(memory_labels).float()).sum() / weight_sum
        return {
            "kl": float(kl.item()),
            "old_confidence": float(old_confidence.item()),
            "new_confidence": float(new_confidence.item()),
            "old_accuracy": float(old_accuracy.item()),
            "new_accuracy": float(new_accuracy.item()),
            "count": int(memory_protos.size(0)),
        }

    @torch.no_grad()
    def update_trusted_proto_memory(self):
        """Add reliable current prototypes to the bounded historical memory."""

        if not self.global_protos:
            return
        labels = []
        protos = []
        for label, proto in self.global_protos.items():
            label = int(label)
            if label < 0 or label >= self.num_classes:
                continue
            proto = torch.as_tensor(proto, device=self.device, dtype=torch.float32).view(-1)
            if proto.numel() == self.feat_dim:
                labels.append(label)
                protos.append(torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0))
        if not protos:
            return

        protos = torch.stack(protos, dim=0)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        distances = torch.cdist(protos, self.semantic_anchors.float(), p=2).pow(2)
        probabilities = torch.softmax(-distances / self.memory_temperature, dim=1)
        row = torch.arange(labels_tensor.numel(), device=self.device)
        confidence = probabilities[row, labels_tensor]

        disagreement_values = torch.as_tensor(
            [max(float(self.proto_disagreement.get(label, 0.0)), 0.0) for label in labels],
            device=self.device,
            dtype=torch.float32,
        )
        positive = disagreement_values[disagreement_values > 0]
        disagreement_scale = (
            float(positive.median().item()) if positive.numel() else 1.0
        )
        disagreement_scale = max(disagreement_scale, 1e-6)

        for idx, label in enumerate(labels):
            support = max(float(self.proto_support.get(label, 0.0)), 0.0)
            conf = float(confidence[idx].item())
            if support < self.memory_min_support or conf < self.memory_confidence_floor:
                continue
            disagreement = float(disagreement_values[idx].item())
            reliability = (support / (support + 1.0)) * math.exp(
                -disagreement / disagreement_scale
            ) * conf
            entry = {
                "proto": protos[idx].detach().cpu(),
                "weight": float(max(reliability, 0.0)),
                "support": support,
                "confidence": conf,
            }
            entries = list(self.trusted_proto_memory.get(int(label), []))
            entries.append(entry)
            entries.sort(key=lambda item: item["weight"], reverse=True)
            self.trusted_proto_memory[int(label)] = entries[: self.memory_max_per_class]

    def anchor_singular_values(self, anchors):
        anchors = torch.nan_to_num(anchors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if anchors.ndim != 2 or anchors.size(0) < 2:
            return torch.empty(0, device=anchors.device)
        centered = anchors - anchors.mean(dim=0, keepdim=True)
        centered = centered / float(max(centered.size(1), 1)) ** 0.5
        rank = max(min(centered.size(0) - 1, centered.size(1)), 1)
        return torch.linalg.svdvals(centered)[:rank].clamp_min(0.0)

    @torch.no_grad()
    def anchor_effective_rank(self, anchors):
        singular_values = self.anchor_singular_values(anchors)
        if singular_values.numel() == 0:
            return 0.0
        energy = singular_values.pow(2)
        total = energy.sum()
        if total <= 0:
            return 0.0
        probs = energy / total.clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        return float(torch.exp(entropy).item())

    @torch.no_grad()
    def anchor_initial_drift(self, anchors):
        """Mean per-class movement normalized by the initial nearest spacing."""

        initial = getattr(self, "initial_semantic_anchors", None)
        if initial is None or anchors is None:
            return 0.0
        initial = initial.float()
        anchors = torch.as_tensor(anchors, device=initial.device, dtype=torch.float32)
        if anchors.shape != initial.shape:
            return 0.0
        nearest = torch.cdist(initial, initial, p=2)
        nearest.fill_diagonal_(float("inf"))
        nearest = nearest.min(dim=1).values.clamp_min(1e-12)
        drift = (anchors - initial).norm(dim=1) / nearest
        return float(torch.nan_to_num(drift, nan=0.0, posinf=0.0).mean().item())

    @torch.no_grad()
    def anchor_volume_stats_for(self, anchors):
        singular_values = self.anchor_singular_values(anchors)

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

    @torch.no_grad()
    def anchor_volume_stats(self):
        return self.anchor_volume_stats_for(self.semantic_anchors)

    def log_anchor_geometry(self, round_idx):
        vol_ratio, spec_min = self.anchor_volume_stats()
        print(
            f"[Round {round_idx}] FedDSAGeom "
            f"rank={self.anchor_effective_rank(self.semantic_anchors):.2f} "
            f"stable_rank={self.anchor_effective_rank(self.stable_semantic_anchors):.2f} "
            f"vol_ratio={vol_ratio:.4f} "
            f"spec_min={spec_min:.4f}"
        )

    def log_mapping_diagnostics(self, round_idx):
        values = {name: entries[-1] for name, entries in self.mapping_diagnostics.items() if entries}
        print(
            f"[Round {round_idx}] FedDSAMap "
            f"PAR={values.get('proto_retrieval_accuracy', np.nan):.4f} "
            f"PAM={values.get('proto_anchor_margin', np.nan):.4f} "
            f"PAC={values.get('positive_alignment_cosine', np.nan):.4f} "
            f"PCD={values.get('prototype_dispersion', np.nan):.4f} "
            f"RDE={values.get('relational_distance_error', np.nan):.4f}"
        )

    def log_new_anchor_diagnostics(self, round_idx):
        values = self.new_anchor_metrics
        if self.new_anchor_mode == "relation_memory":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=relation_memory coverage={values.get('relation_coverage', 0.0):.4f} "
                f"relation_error={values.get('relation_error', np.nan):.4f}"
            )
        elif self.new_anchor_mode == "semantic_discriminative":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=semantic_discriminative "
                f"semantic_acc={values.get('semantic_proto_acc', np.nan):.4f} "
                f"discriminative_acc={values.get('discriminative_proto_acc', np.nan):.4f} "
                f"discriminative_loss={values.get('discriminative_loss', np.nan):.4f}"
            )
        elif self.new_anchor_mode == "observation_validation":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=observation_validation "
                f"grad_norm={values.get('validation_grad_norm', 0.0):.4f}"
            )
        elif self.new_anchor_mode == "meta_separated":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=meta_separated "
                f"grad_norm={values.get('meta_grad_norm', 0.0):.4f} "
                f"step_norm={values.get('meta_step_norm', 0.0):.4f} "
                f"clients={values.get('meta_clients', 0.0):.1f}"
            )
        elif self.new_anchor_mode == "boundary":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=boundary "
                f"classes={values.get('boundary_active_classes', 0.0):.0f} "
                f"step_norm={values.get('boundary_mean_step', 0.0):.4f}"
            )
        elif self.new_anchor_mode == "class_boundary":
            print(
                f"[Round {round_idx}] FedDSANew "
                f"mode=class_boundary "
                f"classes={values.get('class_boundary_active_classes', 0.0):.0f} "
                f"margin={values.get('class_boundary_mean', np.nan):.4f} "
                f"observed={values.get('class_boundary_observed_margin', np.nan):.4f} "
                f"step={values.get('class_boundary_step', 0.0):.4f} "
                f"loss={self.last_train_diagnostics.get('class_boundary_loss', 0.0):.4f} "
                f"active={self.last_train_diagnostics.get('class_boundary_active_rate', 0.0):.4f}"
            )
