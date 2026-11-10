import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientsa import clientSA
from flcore.servers.serverbase import Server


class FedSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientSA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.num_classes = int(args.num_classes)
        self.class_numbers = self.num_classes
        self.feat_channel = int(args.feature_dim)
        self.anchor_momentum = float(getattr(args, "fedsa_anchor_momentum", 0.9999))
        self.anchor_momentum = min(max(self.anchor_momentum, 0.0), 1.0)
        self.print_margins = bool(getattr(args, "fedsa_print_margins", False))

        print("=" * 50)
        print("FedSA Hyper-parameters")
        print(f"Anchor-based Regularization: {args.lamda}")
        print(f"Anchor-based Margin-enhanced Contrastive Learning: {args.margin_contrastive}")
        print(f"Anchor-based Classifier Calibration: {args.classifier_calibration}")
        print(f"Semantic anchor momentum: {self.anchor_momentum}")
        print("=" * 50)

        self.Budget = []
        self.global_protos = {}
        self.previous_anchor_eval_summary = None

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
        random_center = torch.randn(self.class_numbers, self.feat_channel, generator=generator)
        with torch.no_grad():
            self.semantic_anchors = self.anchor_projector(random_center.to(self.device)).detach()

        for client in self.clients:
            client.global_protos = self.global_protos

    def train(self):
        self.send_semantic_anchors()
        self.cal_class_margins(self.semantic_anchors, "Initial semantic anchors")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            results = self.run_clients(self.selected_clients, return_exceptions=True)
            for client, result in results:
                if isinstance(result, Exception):
                    raise RuntimeError(f"FedSA client {client.id} failed during local training.") from result
            self.log_train_diagnostics(i)

            self.receive_protos()
            self.update_semantic_anchors()
            self.send_semantic_anchors()

            proto_anchors = self.proto2anchors(self.global_protos)
            self.cal_class_margins(proto_anchors, "Uploaded prototypes")
            self.cal_class_margins(self.semantic_anchors, "Semantic anchors")

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.maybe_save_tsne()
        self.save_results()

    def send_semantic_anchors(self):
        assert len(self.clients) > 0
        anchors = self.semantic_anchors.detach()
        for client in self.clients:
            client.set_semantic_anchors(anchors)

    def log_train_diagnostics(self, round_num):
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
            print(f"[Round {round_num}] FedSATrainDiag n/a")
            return

        summary = {key: value / total_weight for key, value in metric_sums.items()}
        print(
            f"[Round {round_num}] FedSATrainDiag "
            f"task={summary.get('task_loss', 0.0):.4f} "
            f"anchor={summary.get('anchor_loss', 0.0):.4f} "
            f"mse={summary.get('fedsa_mse_loss', 0.0):.4f} "
            f"mcl={summary.get('fedsa_mcl_loss', 0.0):.4f} "
            f"cc={summary.get('fedsa_cc_loss', 0.0):.4f} "
            f"total={summary.get('total_loss', 0.0):.4f} "
            f"local_gap={summary.get('local_gap', 0.0):.4f} "
            f"global_gap={summary.get('global_gap', 0.0):.4f}"
        )

    def test_metrics(self):
        ids = []
        num_samples = []
        total_correct = []
        total_auc = []
        anchor_eval_samples = 0.0
        anchor_same_dist_sum = 0.0
        anchor_neg_dist_sum = 0.0
        anchor_margin_sum = 0.0
        anchor_margin_values = []
        anchor_nn_correct_sum = 0.0

        for client in self.clients:
            correct, samples, auc = client.test_metrics()
            ids.append(client.id)
            num_samples.append(samples)
            total_correct.append(float(correct))
            total_auc.append(float(auc) * samples)

            task_acc = float(correct) / max(float(samples), 1e-12)
            eval_metrics = getattr(client, "latest_eval_metrics", {})
            anchor_samples = float(eval_metrics.get("anchor_eval_samples", 0.0))
            if anchor_samples > 0:
                anchor_eval_samples += anchor_samples
                anchor_same_dist_sum += anchor_samples * float(eval_metrics.get("anchor_same_dist", 0.0))
                anchor_neg_dist_sum += anchor_samples * float(eval_metrics.get("anchor_neg_dist", 0.0))
                anchor_margin_sum += anchor_samples * float(eval_metrics.get("anchor_margin", 0.0))
                anchor_nn_correct_sum += anchor_samples * float(eval_metrics.get("nearest_anchor_acc", 0.0))
                margin_values = eval_metrics.get("anchor_margin_values")
                if margin_values is not None:
                    margin_values = np.asarray(margin_values, dtype=np.float32)
                    if margin_values.size > 0:
                        anchor_margin_values.append(margin_values)

                print(
                    f"Client {client.id}: Acc: {task_acc}, AUC: {auc} | "
                    f"FedSAAnchorDiag same_dist={float(eval_metrics.get('anchor_same_dist', 0.0)):.4f} "
                    f"neg_dist={float(eval_metrics.get('anchor_neg_dist', 0.0)):.4f} "
                    f"margin={float(eval_metrics.get('anchor_margin', 0.0)):.4f} "
                    f"margin_p10={float(eval_metrics.get('anchor_margin_p10', 0.0)):.4f} "
                    f"nearest_acc={float(eval_metrics.get('nearest_anchor_acc', 0.0)):.4f}"
                )
            else:
                print(f"Client {client.id}: Acc: {task_acc}, AUC: {auc} | FedSAAnchorDiag n/a")

        if anchor_eval_samples > 0:
            margins = np.concatenate(anchor_margin_values, axis=0) if anchor_margin_values else np.array([0.0])
            summary = {
                "same_dist": anchor_same_dist_sum / anchor_eval_samples,
                "neg_dist": anchor_neg_dist_sum / anchor_eval_samples,
                "margin": anchor_margin_sum / anchor_eval_samples,
                "margin_p10": float(np.percentile(margins, 10)),
                "nearest_anchor_acc": anchor_nn_correct_sum / anchor_eval_samples,
            }
            prev = self.previous_anchor_eval_summary
            if prev is None:
                delta_msg = "delta=n/a"
            else:
                delta_msg = (
                    f"delta same_dist={summary['same_dist'] - prev['same_dist']:+.4f} "
                    f"neg_dist={summary['neg_dist'] - prev['neg_dist']:+.4f} "
                    f"margin={summary['margin'] - prev['margin']:+.4f} "
                    f"margin_p10={summary['margin_p10'] - prev['margin_p10']:+.4f} "
                    f"nearest_acc={summary['nearest_anchor_acc'] - prev['nearest_anchor_acc']:+.4f}"
                )
            print(
                f"FedSAAnchorDiagAvg same_dist={summary['same_dist']:.4f} "
                f"neg_dist={summary['neg_dist']:.4f} "
                f"margin={summary['margin']:.4f} "
                f"margin_p10={summary['margin_p10']:.4f} "
                f"nearest_acc={summary['nearest_anchor_acc']:.4f} | {delta_msg}"
            )
            self.previous_anchor_eval_summary = summary

        return ids, num_samples, total_correct, total_auc

    def receive_protos(self):
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = getattr(client, "protos", None)
            if protos is None:
                protos = load_item(client.role, "protos", client.save_folder_name)
            if protos:
                uploaded_protos.append(protos)

        self.global_protos = proto_aggregation(uploaded_protos)
        for client in self.clients:
            client.global_protos = self.global_protos
        save_item(self.global_protos, self.role, "global_protos", self.save_folder_name)

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
        for label, proto in self.global_protos.items():
            anchor = self.semantic_anchors[label]
            anchor.copy_(
                self.anchor_momentum * anchor
                + (1.0 - self.anchor_momentum) * proto.to(anchor)
            )

    @torch.no_grad()
    def cal_class_margins(self, anchors, name="Anchors"):
        anchors = torch.as_tensor(anchors, device=self.device, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.size(0) < 2:
            print(f"{name} margins | classes={anchors.size(0) if anchors.ndim == 2 else 0} min=0.0000 avg=0.0000 max=0.0000")
            return 0.0, 0.0, 0.0

        anchors = torch.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0)
        dist_matrix = torch.cdist(anchors, anchors, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        min_margins = torch.min(dist_matrix, dim=1).values
        finite_margins = min_margins[torch.isfinite(min_margins)]
        if finite_margins.numel() == 0:
            min_margin = avg_margin = max_margin = 0.0
        else:
            min_margin = float(torch.min(finite_margins).item())
            avg_margin = float(torch.mean(finite_margins).item())
            max_margin = float(torch.max(finite_margins).item())

        print(
            f"{name} margins | classes={anchors.size(0)} "
            f"min={min_margin:.4f} avg={avg_margin:.4f} max={max_margin:.4f}"
        )
        if self.print_margins:
            print("Distance Matrix:")
            print(dist_matrix.detach().cpu())
        return min_margin, avg_margin, max_margin


def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        if not local_protos:
            continue
        for label, proto in local_protos.items():
            if proto is None or isinstance(proto, list):
                continue
            proto = torch.as_tensor(proto, dtype=torch.float32).detach().cpu().view(-1)
            agg_protos[int(label)].append(proto)

    for label, proto_list in list(agg_protos.items()):
        if len(proto_list) == 0:
            del agg_protos[label]
        else:
            agg_protos[label] = torch.stack(proto_list, dim=0).mean(dim=0)

    return dict(agg_protos)
