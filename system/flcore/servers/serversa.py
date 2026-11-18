import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientsa import FUSE_ALPHAS, clientSA
from flcore.servers.serverbase import Server
from utils.feddsa_utils import simplex_anchor_bank



@torch.no_grad()
def twin_coherence(states, prev):
    """Cross-client coherence of the twin backbone updates d_i = w_i - w_prev (coherent-gradients view of averaging).
    states: [(state_dict, weight)]; prev: {key: flat float64 cpu tensor}. Returns (rho, mean pairwise cosine, rms norm),
    rho = ||sum_i p_i d_i||^2 / sum_i p_i ||d_i||^2: 1 = identical updates, sum_i p_i^2 = mutually orthogonal."""
    p = torch.tensor([w for _, w in states], dtype=torch.float64)
    G = torch.zeros(len(states), len(states), dtype=torch.float64)
    for k in prev:  # Gram matrix of the updates, one parameter tensor at a time (no full-model copies)
        D = torch.stack([st[k].double().cpu().flatten() - prev[k] for st, _ in states])
        G += D @ D.T
    diag = G.diag()
    n = len(states)
    cos = G / torch.sqrt(diag[:, None] * diag[None, :]).clamp_min(1e-30)
    return float(p @ G @ p / (p @ diag)), float((cos.sum() - cos.diag().sum()) / (n * (n - 1))), float((p @ diag).sqrt())


@torch.no_grad()
def twin_step(states, avg, prev, server_lr=1.0, agree=0.0):
    """Server step for the twin backbone, in place on `avg` (the plain FedAvg) for the keys of `prev`.
    agree > 0: keep only coordinates where the weighted sign agreement |sum_i p_i sign(w_i - prev)| >= agree (the
    clients' common direction; disagreeing coordinates are client-specific, i.e. memorization). server_lr scales the
    averaged step: prev + server_lr * (avg - prev). Returns the kept fraction of coordinates (1.0 without agree)."""
    kept = total = 0
    for k, p in prev.items():
        v = avg[k]
        p = p.view(v.shape).to(v.device, torch.float64)
        step = v.double() - p
        if agree > 0:
            sgn = sum(w * torch.sign(st[k].to(v.device, torch.float64) - p) for st, w in states)
            mask = sgn.abs() >= agree
            step = step * mask
            kept += int(mask.sum())
        total += step.numel()
        avg[k] = (p + server_lr * step).to(v.dtype)
    return kept / max(total, 1) if agree > 0 else 1.0


def _lead(full, like):
    """Leading-index slice of `full` with the shape of `like` (width-nested models share their first channels)."""
    return full[tuple(slice(0, n) for n in like.shape)].clone() if like.dim() > 0 else full.clone()


class FedSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args

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
            if getattr(args, "fedsa_anchor_codebook", "gaussian") == "simplex":
                self.semantic_anchors = simplex_anchor_bank(
                    self.semantic_anchors, self.class_numbers, self.feat_channel, generator)

        for client in self.clients:
            client.global_protos = self.global_protos

        if getattr(args, "fedsa_common_model", False):
            import copy
            from flcore.trainmodel.models import BaseHeadSplit
            cargs = copy.copy(args)
            arch = getattr(args, "fedsa_common_arch", "cnn")
            cargs.models = [{"bn": 'SmallBNCNN(in_features=3, num_classes=args.num_classes)',
                             "r18": 'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                             "mnv2": 'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                             "gnet": 'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                             "sres": 'WidthSenResNet(args.sen_shape, args.num_classes)',
                             }.get(arch, 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)')]
            cifar = arch[0] == "c" and arch[1:].isdigit()  # c<depth>: CIFAR ResNet-depth twin, channels x --fedsa_cm_width
            if cifar:
                cargs.models = [f'CIFARResNet({int(arch[1:])}, width={float(getattr(args, "fedsa_cm_width", 1.0))}, num_classes=args.num_classes)']
            cargs.fedsa_projector = "linear"
            widths = getattr(args, "fedsa_cm_widths", "")
            if widths:
                # nested twin: client c runs a width-r_c ResNet18 whose tensors are leading slices of one full model
                ws = [float(w) for w in widths.split(",")]
                if arch == "sres":
                    spec = lambda r: [f'WidthSenResNet(args.sen_shape, args.num_classes, width={r})']
                elif cifar:
                    spec = lambda r: [f'CIFARResNet({int(arch[1:])}, width={r}, num_classes=args.num_classes)']
                else:
                    spec = lambda r: [f'WidthResNet18(width={r}, num_classes=args.num_classes)']
                cargs.models = spec(1.0)
                self.cm_global = BaseHeadSplit(cargs, 0)
                full = self.cm_global.state_dict()
                for client in self.clients:
                    cargs.models = spec(ws[client.id % len(ws)])
                    m = BaseHeadSplit(cargs, 0)
                    m.load_state_dict({k: _lead(full[k], v) for k, v in m.state_dict().items()})
                    client.common_model = m
                sizes = sorted({sum(p.numel() for p in c.common_model.base[0].parameters()) for c in self.clients})
                print(f"[nested twin] widths {ws}, backbone params per width: {[f'{n / 1e6:.2f}M' for n in sizes]}")
            else:
                common = BaseHeadSplit(cargs, 0)
                for client in self.clients:
                    client.common_model = copy.deepcopy(common)

    def train(self):
        self.send_semantic_anchors()
        self.cal_class_margins(self.semantic_anchors, "Initial semantic anchors")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            for c in self.clients:
                c.set_round(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            results = self.run_clients(self.selected_clients, return_exceptions=True)
            for client, result in results:
                if isinstance(result, Exception):
                    raise RuntimeError(f"FedSA client {client.id} failed during local training.") from result
            if getattr(self.args, "fedsa_head_avg", False):
                self.average_heads()
            if getattr(self.args, "fedsa_pub_kd", 0.0) > 0 and i + 1 >= int(getattr(self.args, "fedsa_pub_start", 10)):
                self.edge_ensemble_targets()
            if getattr(self.args, "fedsa_vgen", 0.0) > 0 and not getattr(self.args, "fedsa_vgen_private", False):
                self.average_module("vgen")
            if getattr(self.args, "fedsa_vgen", 0.0) > 0 and getattr(self.args, "fedsa_vgen_mode", "local") in ("global", "transport"):
                self.aggregate_vgen_stats()
            if getattr(self.args, "fedsa_faa", 0.0) > 0 and not getattr(self.args, "fedsa_faa_private", False):
                self.average_augmenters()
            if getattr(self.args, "fedsa_shape_avg", False):
                self.average_matching_layers()
            if getattr(self.args, "fedsa_memref", 0.0) > 0:
                self.aggregate_memref()
            if getattr(self.args, "fedsa_peer_invert", 0.0) > 0:
                self.exchange_synthetic()
            if int(getattr(self.args, "fedsa_kernel_bank", 0)) > 0 and not getattr(self.args, "fedsa_bank_private", False):
                self.average_kernel_banks()
            if getattr(self.args, "fedsa_group_avg", False):
                self.average_same_architecture()
            if getattr(self.args, "fedsa_common_model", False) and not getattr(self.args, "fedsa_cm_private", False) \
                    and not getattr(self, "_cm_frozen", False):
                if getattr(self.args, "fedsa_cm_widths", ""):
                    self.average_nested_common_models()
                else:
                    self.average_common_models(i)
            # the gate restores one shared backbone to everyone, so it only makes sense for an averaged twin
            if int(getattr(self.args, "fedsa_cm_gate", 0)) > 0 and getattr(self.args, "fedsa_common_model", False) \
                    and not getattr(self.args, "fedsa_cm_private", False):
                self.gate_common_model(i)
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
            + (f" view_shift_norm={summary['view_shift_norm']:.4f} view_noise_norm={summary['view_noise_norm']:.4f}"
               if "view_shift_norm" in summary else "")
            + (f" priv_frac={summary['priv_frac']:.4f}" if "priv_frac" in summary else "")
            + "".join(f" {k}={summary[k]:.4f}" for k in ("train_acc", "cm_train_acc", "cm_from_priv_kl", "teacher_train_acc", "syn_teacher_acc", "syn_recv_acc", "memref_w", "faa_adv_acc", "faa_delta", "t_copy", "t_inv", "t_post", "syn_keep", "vgen_rec", "vgen_acc", "vgen_delta") if k in summary)
        )
        if round_num % self.eval_gap == 0:
            print(f"[Round {round_num}] Per-client train_acc: " + " ".join(
                f"{c.id}:{float(c.latest_train_metrics.get('train_acc', 0.0)):.4f}"
                for c in self.selected_clients if getattr(c, "latest_train_metrics", None)))
            print(f"[Round {round_num}] Per-client local_gap: " + " ".join(
                f"{c.id}:{float(c.latest_train_metrics.get('local_gap', 0.0)):.4f}"
                for c in self.selected_clients if getattr(c, "latest_train_metrics", None)))

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

        cm_correct = cm_samples = 0
        mem_aucs = defaultdict(list)
        fused_tot = defaultdict(int)
        for client in self.clients:
            if getattr(client, "common_model", None) is not None:
                cc, cn = client.common_model_test()
                cm_correct, cm_samples = cm_correct + cc, cm_samples + cn
                for k, v in getattr(client, "fused_correct", {}).items():
                    fused_tot[k] += v
                if getattr(self.args, "fedsa_mem_probe", False):
                    for k, v in client.memorization_probe().items():
                        mem_aucs[k].append(v)
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

        if cm_samples > 0:
            print(f"Common model (twin) test accuracy: {cm_correct / cm_samples:.4f}")
            if fused_tot:
                print("Fused (private+twin) test accuracy " + " ".join(f"{k}={v / cm_samples:.4f}" for k, v in fused_tot.items()))
            alphas = [c.alpha_choice for c in self.clients if hasattr(c, "alpha_choice")]
            if alphas:
                print(f"FusedAlpha heldout mean={sum(alphas) / len(alphas):.3f} "
                      + " ".join(f"n{a:g}={alphas.count(a)}" for a in sorted(set(alphas)))
                      + " per_client=" + ",".join(f"{c.id}:{c.alpha_choice:g}" for c in self.clients if hasattr(c, "alpha_choice")))
            hcs = [c for c in self.clients if hasattr(c, "heldout_counts")]
            if hcs:  # one shared alpha from all clients' held-out correct counts (clients send 6 integers)
                pooled = [sum(c.heldout_counts[i] for c in hcs) for i in range(len(FUSE_ALPHAS))]
                pa = FUSE_ALPHAS[pooled.index(max(pooled))]  # first max -> smaller alpha on ties
                pn = [sum(c.heldout_nll[i] for c in hcs) for i in range(len(FUSE_ALPHAS))]
                pna = FUSE_ALPHAS[pn.index(min(pn))]
                # per client: id:held-out n|held-out correct|test correct|held-out NLL|held-out masked correct|test masked
                # correct (each per alpha) -> any selection rule offline
                print(f"FusedAlphaCounts pooled_choice={pa:g} a_pooled={fused_tot[f'a{pa:g}'] / cm_samples:.4f} "
                      f"pooled_nll_choice={pna:g} a_pooled_nll={fused_tot[f'a{pna:g}'] / cm_samples:.4f} "
                      f"n_test={cm_samples} alphas=" + ",".join(f"{a:g}" for a in FUSE_ALPHAS)
                      + " per_client=" + ";".join(
                          f"{c.id}:{c.heldout_n}|" + ",".join(map(str, c.heldout_counts)) + "|"
                          + ",".join(str(c.fused_correct[f"a{a:g}"]) for a in FUSE_ALPHAS) + "|"
                          + ",".join(f"{v:.1f}" for v in c.heldout_nll) + "|" + ",".join(map(str, c.heldout_mask)) + "|"
                          + ",".join(str(c.fused_correct[f"m{a:g}"]) for a in FUSE_ALPHAS) for c in hcs))
        if mem_aucs:
            print("MemProbe membership AUC " + " ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in mem_aucs.items()))
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
        if getattr(self.args, "fedsa_relation", 0.0) > 0:
            sums, cnts = {}, defaultdict(float)
            for c in self.selected_clients:
                for label, (n, vec) in (getattr(c, "rel_stats", None) or {}).items():
                    sums[label] = sums.get(label, 0) + vec
                    cnts[label] += n
            global_rel = {label: sums[label] / cnts[label] for label in sums if cnts[label] > 0}
            for client in self.clients:
                client.global_rel = global_rel
        if getattr(self.args, "fedsa_second_view", "off") == "proto_shift":
            per_class = defaultdict(list)
            for c in self.selected_clients:
                for label, proto in (getattr(c, "protos", None) or {}).items():
                    per_class[int(label)].append(torch.as_tensor(proto, dtype=torch.float32).view(-1).cpu())
            devs = {}
            for label, plist in per_class.items():
                if len(plist) >= 2:
                    P = torch.stack(plist)
                    devs[label] = P - P.mean(dim=0, keepdim=True)
            for client in self.clients:
                client.proto_devs = devs
        if getattr(self.args, "fedsa_peer_head", 0.0) > 0:
            heads = []
            for c in self.selected_clients:
                m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
                heads.append((c.id, m.head.weight.detach().cpu().clone(), m.head.bias.detach().cpu().clone()))
            for client in self.clients:
                client.peer_heads = [(W, b) for cid, W, b in heads if cid != client.id]
        if getattr(self.args, "fedsa_second_view", "off") in ("stat", "noise", "stat_ce", "proto_shift", "noise_kl", "noise_fixed") or \
                getattr(self.args, "fedsa_bottleneck", "off") != "off":
            self.global_var = pooled_class_variance([getattr(c, "proto_stats", {}) for c in self.selected_clients])
            for client in self.clients:
                client.global_var = self.global_var
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
    def average_common_models(self, rnd=None):
        """FedAvg of the shared small model over all selected clients (every client has the same architecture)."""
        total = float(sum(c.train_samples for c in self.selected_clients))
        states = [(c.common_model.state_dict(), c.train_samples / total) for c in self.selected_clients]
        if getattr(self, "_cm_prev", None) is not None:
            rho, mcos, norm = twin_coherence(states, self._cm_prev)
            print(f"[Round {rnd}] TwinCoherence rho={rho:.4f} mean_cos={mcos:.4f} update_rms={norm:.4f}")
        base_only = getattr(self.args, "fedsa_cm_base_only", False)  # twin: average the backbone, keep projector/head personal
        bn_local = getattr(self.args, "fedsa_cm_bn_local", False)  # FedBN: BN running statistics stay per client
        avg = {k: (sum(w * st[k].float().to(self.device) for st, w in states).to(v.dtype) if v.is_floating_point()
                   else v.clone()) for k, v in states[0][0].items() if (not base_only or k.startswith("base.0."))
               and not (bn_local and ("running_" in k or "num_batches" in k))}
        server_lr, agree = float(getattr(self.args, "fedsa_cm_server_lr", 1.0)), float(getattr(self.args, "fedsa_cm_agree", 0.0))
        if (server_lr != 1.0 or agree > 0) and getattr(self, "_cm_prev", None) is not None:
            kept = twin_step(states, avg, self._cm_prev, server_lr, agree)
            if agree > 0:
                print(f"[Round {rnd}] TwinAgree kept={kept:.4f}")
        realign = int(getattr(self.args, "fedsa_cm_realign", 0))
        before = getattr(self.args, "fedsa_cm_realign_before", False)
        if realign > 0 and base_only and before:  # compute-matched control: same head epochs on my own local backbone
            with torch.enable_grad():
                for c in self.clients:
                    c.realign_twin_head(realign)
        for c in self.clients:
            c.common_model.load_state_dict(avg, strict=not base_only)
        if realign > 0 and base_only and not before:
            with torch.enable_grad():
                for c in self.clients:
                    c.realign_twin_head(realign)
        self._cm_prev = {k: v.double().cpu().flatten() for k, v in avg.items()  # start point of next round's updates
                         if k.startswith("base.0.") and v.is_floating_point() and "running_" not in k}

    @torch.no_grad()
    def gate_common_model(self, rnd):
        """Memorization gate of the shared term: after each aggregation, measure the received twin on the clients'
        held-out splits, keep the best backbone, and once it has not improved for `fedsa_cm_gate` rounds restore it and
        freeze it for the rest of training (no more twin backbone updates or averaging; the private residual keeps
        training on top). Logs the memorization gap = train accuracy - held-out accuracy of the shared term."""
        if getattr(self, "_cm_frozen", False):
            return
        assert float(getattr(self.args, "fedsa_cm_val", 0.0)) > 0, "--fedsa_cm_gate needs a held-out split (--fedsa_cm_val)"
        stats = [c.twin_gate_stats() for c in self.clients]
        n = sum(s[2] for s in stats)
        val, tr = sum(s[0] for s in stats) / n, sum(s[1] for s in stats) / n
        print(f"[Round {rnd}] TwinGate held_out={val:.4f} train={tr:.4f} gap={tr - val:.4f}")
        if val > getattr(self, "_gate_best", -1.0):
            src = self.clients[0].common_model if getattr(self, "cm_global", None) is None else self.cm_global
            self._gate_best, self._gate_round = val, rnd
            # nested / bn_local: BN statistics are local (client 0's must not be restored to everyone)
            nested = getattr(self, "cm_global", None) is not None or getattr(self.args, "fedsa_cm_bn_local", False)
            self._gate_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()
                                if k.startswith("base.0.") and not (nested and ("running_" in k or "num_batches" in k))}
        freeze_at = int(getattr(self.args, "fedsa_cm_freeze_at", 0))  # control: freeze at a fixed round instead of on a plateau
        if rnd - self._gate_round >= int(self.args.fedsa_cm_gate) or 0 < freeze_at <= rnd:
            for c in self.clients:
                own = c.common_model.state_dict()
                c.common_model.load_state_dict({k: _lead(v, own[k]).to(own[k].device, own[k].dtype)
                                                for k, v in self._gate_state.items()}, strict=False)
                c.cm_frozen = True
                if getattr(self.args, "fedsa_cm_freeze_head", False):
                    with torch.enable_grad():  # refit my personal head to the restored backbone once, then fix it too
                        c.realign_twin_head(max(1, int(getattr(self.args, "fedsa_cm_realign", 0))))
                    c.cm_head_frozen = True
            self._cm_frozen = True
            print(f"[Round {rnd}] TwinGate freeze: restored the round-{self._gate_round} backbone (held-out {self._gate_best:.4f})")

    @torch.no_grad()
    def average_nested_common_models(self):
        """HeteroFL-style FedAvg of the width-nested twin backbone (base.0.): every entry of the full model is the
        sample-weighted mean over the selected clients whose slice covers it; uncovered entries keep their value.
        Projector/head stay personal (as with --fedsa_cm_base_only)."""
        g = self.cm_global.state_dict()
        base_only = getattr(self.args, "fedsa_cm_base_only", False)  # False: projector/head averaged too (ablation)
        total = float(sum(c.train_samples for c in self.selected_clients))
        states = [(c.common_model.state_dict(), c.train_samples / total) for c in self.selected_clients]
        for k, v in g.items():
            # BN running stats stay local (FedBN/static-BN style): a width-r slice computes different activations
            # than the same channels of the full model, so averaging their statistics corrupts eval-mode outputs
            if (base_only and not k.startswith("base.0.")) or not v.is_floating_point() or "running_" in k:
                continue
            acc = torch.zeros(v.shape)
            cnt = torch.zeros(v.shape)
            for st, w in states:
                idx = tuple(slice(0, n) for n in st[k].shape)
                acc[idx] += w * st[k].float().cpu()
                cnt[idx] += w
            covered = cnt > 0
            v[covered] = (acc[covered] / cnt[covered]).to(v.dtype)
        for c in self.clients:
            own = c.common_model.state_dict()
            c.common_model.load_state_dict({k: _lead(g[k], t).to(t.device, t.dtype)
                                            if (k.startswith("base.0.") or not base_only) and t.is_floating_point()
                                            and "running_" not in k else t
                                            for k, t in own.items()})

    @torch.no_grad()
    def average_same_architecture(self):
        """FedAvg of whole models among clients that run the same architecture (restores weight averaging
        inside each architecture cohort); different architectures still interact only via the shared anchors."""
        model_of = lambda c: getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
        arch = lambda c: self.args.models[c.id % len(self.args.models)]
        cohorts = defaultdict(list)
        for c in self.selected_clients:
            cohorts[arch(c)].append(c)
        for spec, members in cohorts.items():
            if len(members) < 2:
                continue
            total = float(sum(c.train_samples for c in members))
            states = [(model_of(c).state_dict(), c.train_samples / total) for c in members]
            avg = {}
            for k, v in states[0][0].items():
                avg[k] = sum(w * st[k].float() for st, w in states).to(v.dtype) if v.is_floating_point() else v.clone()
            alpha = float(getattr(self.args, "fedsa_group_alpha", 1.0))
            for c in self.clients:
                if arch(c) == spec:
                    if alpha >= 1.0:
                        model_of(c).load_state_dict(avg)
                        continue
                    # soft, backbone-only pull toward the cohort mean (projector/head stay personal)
                    own = model_of(c).state_dict()
                    model_of(c).load_state_dict({k: ((1 - alpha) * own[k].float() + alpha * avg[k].float()).to(own[k].dtype)
                                                 if k.startswith("base.0.") and own[k].is_floating_point() else own[k]
                                                 for k in own})

    @torch.no_grad()
    def aggregate_vgen_stats(self):
        """Per-class latent mean/std of the variation generator. Shared generator: pooled over all clients (one latent
        space). Private control: each client keeps its own statistics (latent spaces differ, pooling is meaningless)."""
        private = getattr(self.args, "fedsa_vgen_private", False)
        stats = [c.vgen_stats for c in self.selected_clients if isinstance(c.vgen_stats, dict) and "sum" in c.vgen_stats]
        if not stats:
            return
        def finish(s, n, sq, shape, fallback=None):
            m = s / n.clamp(min=1).unsqueeze(1)
            v = (sq / n.clamp(min=1).unsqueeze(1) - m * m).clamp(min=1e-6)
            if fallback is not None:  # classes I (almost) never saw: use the pooled statistics (unreliable moments)
                miss = n < 10
                m[miss], v[miss] = fallback[0][miss], fallback[1][miss]
            return m.view(-1, *shape), v.sqrt().view(-1, *shape)
        shape = stats[0]["shape"]
        tot = {k: sum(s[k].to(self.device) for s in stats) for k in ("sum", "sq", "n")}
        gm, gs = finish(tot["sum"], tot["n"], tot["sq"], shape)
        transport = getattr(self.args, "fedsa_vgen_mode", "local") == "transport"
        for c in self.clients:
            st = c.vgen_stats if isinstance(c.vgen_stats, dict) and "sum" in c.vgen_stats else None
            own = None if st is None else finish(st["sum"].to(self.device), st["n"].to(self.device), st["sq"].to(self.device),
                                                 shape, (gm.flatten(1), gs.pow(2).flatten(1)))
            if transport:  # target = pooled stats (for private generators this pools incompatible latents: the control)
                c.vgen_global, c.vgen_local = (gm, gs), own
            elif private and own is not None:
                c.vgen_global = own
            else:
                c.vgen_global = (gm, gs)
            c.vgen_stats = {}  # fresh statistics next round (the generator keeps changing)

    @torch.no_grad()
    def edge_ensemble_targets(self):
        """Hierarchical IoT: each edge gateway (clients grouped by id) holds the heterogeneous models of its devices
        (cheap LAN) and an unlabeled public pool. It evaluates every device model on a fresh public subset in the
        fixed anchor space (same coordinates for all architectures), averages them into ensemble targets, and sends
        (images, targets) back. target: edge = my edge's ensemble; global = all edges (anchor-space outputs on the
        same subset are the only thing crossing the WAN); self = my own outputs (control: public data, no ensemble)."""
        source = getattr(self.args, "fedsa_pub_source", "public")
        if source in ("peer", "peer_self"):  # no public data: peers' models evaluated on each device's own images
            n_edge = int(getattr(self.args, "fedsa_edge_num", 4))
            edge_of = lambda cid: cid * n_edge // self.num_clients
            model_of = lambda c: getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            for c in self.selected_clients:
                if source == "peer":
                    peers = [(model_of(p), p.class_counts()) for p in self.selected_clients
                             if p.id != c.id and edge_of(p.id) == edge_of(c.id)]
                else:
                    peers = [(model_of(c), torch.ones(self.args.num_classes))]
                if peers:
                    c.build_peer_set(peers)
            return
        if getattr(self, "_pub_x", None) is None:
            import numpy as np
            self._pub_x = torch.from_numpy(np.load(self.args.fedsa_pub_path)["x"])
        n_edge = int(getattr(self.args, "fedsa_edge_num", 4))
        mode = getattr(self.args, "fedsa_pub_target", "edge")
        idx = torch.randperm(len(self._pub_x))[: int(getattr(self.args, "fedsa_pub_n", 512))]
        xs = self._pub_x[idx].to(self.device)
        reps = {}
        for c in self.selected_clients:
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            was = m.training
            m.to(self.device).eval()
            space = getattr(self.args, "fedsa_pub_space", "anchor")
            reps[c.id] = torch.cat([(m.head(m.base(xs[k:k + 256])) if space == "logit" else m.base(xs[k:k + 256])).float()
                                    for k in range(0, len(xs), 256)])  # logit: FedDF/DS-FL-style output ensemble
            m.train(was)
        edge_of = lambda cid: cid * n_edge // self.num_clients
        edge_mean = {}
        for e in range(n_edge):
            mem = [r for cid, r in reps.items() if edge_of(cid) == e]
            if mem:
                edge_mean[e] = torch.stack(mem).mean(0)
        glob = torch.stack(list(edge_mean.values())).mean(0)
        for c in self.clients:
            if c.id not in reps:
                continue
            tgt = {"edge": edge_mean.get(edge_of(c.id)), "global": glob, "self": reps[c.id]}[mode]
            c.pub_set = (xs, tgt)

    @torch.no_grad()
    def average_module(self, attr):
        """FedAvg of a homogeneous per-client module stored as client.<attr> (same architecture everywhere)."""
        total = float(sum(c.train_samples for c in self.selected_clients))
        states = [(getattr(c, attr).state_dict(), c.train_samples / total) for c in self.selected_clients]
        avg = {k: sum(w * st[k].float().to(self.device) for st, w in states) for k in states[0][0]}
        for c in self.clients:
            m = getattr(c, attr)
            m.load_state_dict({k: v.to(m.state_dict()[k]) for k, v in avg.items()})

    @torch.no_grad()
    def average_augmenters(self):
        """FedAvg of the shared input-space augmenter (same architecture in every client)."""
        total = float(sum(c.train_samples for c in self.selected_clients))
        states = [(c.augmenter.state_dict(), c.train_samples / total) for c in self.selected_clients]
        avg = {k: sum(w * st[k].float().to(self.device) for st, w in states) for k in states[0][0]}
        for c in self.clients:
            c.augmenter.load_state_dict({k: v.to(c.augmenter.state_dict()[k]) for k, v in avg.items()})

    @torch.no_grad()
    def average_matching_layers(self):
        """FedAvg of every backbone tensor (base.0.*, incl. BN buffers) that has the same name AND shape in >= 2
        clients, across architectures (e.g. ResNet18/34 basic blocks, ResNet50/101/152 bottlenecks). Projector and
        head stay personal, as in the homogeneous control."""
        models = [(getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name), float(c.train_samples))
                  for c in self.selected_clients]
        groups = defaultdict(list)
        for m, w in models:
            for k, v in m.state_dict().items():
                if k.startswith("base.0.") and v.is_floating_point():
                    groups[(k, tuple(v.shape))].append((v, w))
        avg = {key: sum(w * v.float().to(self.device) for v, w in vs) / sum(w for _, w in vs)
               for key, vs in groups.items() if len(vs) >= 2}
        if not getattr(self, "_shape_avg_logged", False):
            self._shape_avg_logged = True
            for c in self.selected_clients[:8]:
                m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
                sd = {k: v for k, v in m.state_dict().items() if k.startswith("base.0.") and v.is_floating_point()}
                tot = sum(v.numel() for v in sd.values())
                sh = sum(v.numel() for k, v in sd.items() if (k, tuple(v.shape)) in avg)
                print(f"[shape_avg] client {c.id} {type(m.base[0]).__name__}: shared {sh / max(tot, 1):.1%} of backbone")
        for c in self.clients:
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            sd = m.state_dict()
            for k, v in sd.items():
                if (k, tuple(v.shape)) in avg:
                    v.copy_(avg[(k, tuple(v.shape))].to(v.dtype))

    def aggregate_memref(self):
        """Federation's mean training loss per class (count-weighted over the clients that hold the class)."""
        tot = defaultdict(lambda: [0.0, 0.0])
        for c in self.selected_clients:
            for k, (n, s) in getattr(c, "memref_stats", {}).items():
                tot[k][0] += n
                tot[k][1] += s
        ref = {k: s / n for k, (n, s) in tot.items() if n > 0}
        for c in self.clients:
            c.memref_global = ref

    @torch.no_grad()
    def exchange_synthetic(self):
        """Route every client's backbone-inversion images to the OTHER clients that hold those classes (or, for the
        local control, back to the producer itself). Only input-space images + producer logits leave a client."""
        outs = [(c.id, c.syn_out) for c in self.selected_clients if c.syn_out is not None]
        if not outs:
            return
        local = getattr(self.args, "fedsa_peer_invert_local", False)
        cap = int(getattr(self.args, "fedsa_inv_recv", 512))
        for c in self.clients:
            src = [o for i, o in outs if (i == c.id) == local]
            mine = getattr(c, "local_labels", None)
            if not src or mine is None:
                continue
            pool = [torch.cat(v) for v in zip(*src)]  # xs, ys, logits, anchor-space reps
            ys, lt = pool[1], pool[2]
            keep = torch.isin(ys, mine.cpu()).nonzero().flatten()
            keep = keep[torch.randperm(len(keep))[:cap]]
            if len(keep) == 0:
                continue
            new = tuple(v[keep].to(c.device) for v in pool)
            # transferability: does the RECEIVER's own model already see these images as their class?
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            was = m.training
            m.eval()
            pred = m.head(m.base(new[0])).argmax(1)
            m.train(was)
            c._syn_recv_acc = float(pred.eq(new[1]).float().mean())
            bank = int(getattr(self.args, "fedsa_inv_bank", 0))
            if bank > 0 and c.syn_in is not None and len(c.syn_in) == len(new):  # accumulate, keep the newest `bank`
                new = tuple(torch.cat([o, n])[-bank:] for o, n in zip(c.syn_in, new))
            c.syn_in = new
            c._syn_acc = float(new[2].argmax(1).eq(new[1]).float().mean())

    @torch.no_grad()
    def average_kernel_banks(self):
        """FedAvg of the shared kernel bank: same shape in every architecture, so heterogeneous backbones
        share conv weight structure while their per-layer coefficients stay personal."""
        models = [(getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name), float(c.train_samples))
                  for c in self.selected_clients]
        avg = {}
        for k in {k for m, _ in models for k in m.kernel_bank.banks}:  # kernel sizes differ by architecture
            have = [(m.kernel_bank.banks[k], w) for m, w in models if k in m.kernel_bank.banks]
            avg[k] = sum(w * p.detach().float().to(self.device) for p, w in have) / sum(w for _, w in have)
        for c in self.clients:
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            for k, p in m.kernel_bank.banks.items():
                if k in avg:
                    p.copy_(avg[k].to(p))

    @torch.no_grad()
    def average_heads(self):
        """FedAvg of the classifier heads: every client maps into the same fixed anchor space,
        so the 512->C heads share one shape and meaning across heterogeneous backbones."""
        models = []
        for c in self.selected_clients:
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            models.append((m, float(c.train_samples)))
        total = sum(w for _, w in models)
        avg = {k: sum(w * m.head.state_dict()[k].float().to(self.device) for m, w in models) / total
               for k in models[0][0].head.state_dict()}
        for c in self.clients:
            m = getattr(c, "model", None) or load_item(c.role, "model", c.save_folder_name)
            m.head.load_state_dict({k: v.to(m.head.state_dict()[k]) for k, v in avg.items()})

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


def pooled_class_variance(stats_list):
    """Per-class, per-dimension variance pooled over clients from (count, sum, sum of squares)."""
    n, s1, s2 = defaultdict(float), {}, {}
    for stats in stats_list:
        for c, (cnt, sm, sq) in stats.items():
            n[c] += cnt
            s1[c] = s1.get(c, 0) + sm
            s2[c] = s2.get(c, 0) + sq
    return {c: (s2[c] / n[c] - (s1[c] / n[c]).square()).clamp_min(0) for c in n if n[c] > 1}
