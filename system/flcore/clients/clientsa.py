import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client, load_item, save_item
from torch.utils.data import DataLoader, Subset
from utils.augment import augment_batch, mix_batch
from utils.deep_anchor import attach as deep_attach, find_stages, make_projectors
from utils.cm_invert import synthesize, frozen_bn
from utils.fed_augmenter import Augmenter
from utils.var_gen import VarGen


def fix_head(head, seed=12345):
    """FedBABU-style fixed classifier: the same random linear head on every client (and model), never trained."""
    if getattr(head, "_fixed", False):
        return
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # nn.Linear's default range U(-1/sqrt(d), 1/sqrt(d)), zero bias
        head.weight.copy_((torch.rand(head.weight.shape, generator=g) * 2 - 1) / head.in_features ** 0.5)
        if head.bias is not None:
            head.bias.zero_()
    head.requires_grad_(False)
    head._fixed = True


def ntd_loss(student, teacher, y):
    """FedNTD (Lee et al., NeurIPS'22): match the teacher's distribution over the not-true classes only (tau = 1)."""
    keep = ~F.one_hot(y, student.size(1)).bool()
    nt = lambda z: z.float()[keep].view(z.size(0), -1)
    return F.kl_div(F.log_softmax(nt(student), 1), F.softmax(nt(teacher), 1), reduction="batchmean")


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
        self.gap_reduce = min if getattr(args, "fedsa_gap_mode", "max") == "min" else max
        self.gap_warmup = int(getattr(args, "fedsa_gap_warmup", 0))
        self.gap_scale = float(getattr(args, "fedsa_gap_scale", 1.0))
        self.frozen_head = getattr(args, "fedsa_frozen_head", "none")
        self.cm_ntd = float(getattr(args, "fedsa_cm_ntd", 0.0))
        self.cm_from_priv = float(getattr(args, "fedsa_cm_from_priv", 0.0))
        self.cm_lr_scale = float(getattr(args, "fedsa_cm_lr_scale", 1.0))
        self.res_sum_loss = bool(getattr(args, "fedsa_res_sum_loss", True))
        self.res_kl = float(getattr(args, "fedsa_res_kl", 1.0))
        self.eval_head = getattr(args, "fedsa_eval", "proto") == "head"
        self.eval_protos = getattr(args, "fedsa_proto_pass", "train") == "eval"
        self.align_detach = bool(getattr(args, "fedsa_align_detach", False))
        self.augment = bool(getattr(args, "fedsa_augment", False))
        self.mix = getattr(args, "fedsa_mix", "off")  # off | mixup | cutmix (strong-augmentation baselines)
        self.mix_prob = float(getattr(args, "fedsa_mix_prob", 0.5))
        self.anchor_mixup = float(getattr(args, "fedsa_anchor_mixup", 0.0))
        self.soft_label = float(getattr(args, "fedsa_proto_soft", 0.0))  # beta: weight of prototype soft target
        self.soft_temp = float(getattr(args, "fedsa_proto_soft_temp", 0.1))
        self.weight_decay = float(getattr(args, "fedsa_weight_decay", 0.0))
        self.common_model = None  # shared small model (set by the server when --fedsa_common_model is on)
        self.kd_weight = float(getattr(args, "fedsa_kd_weight", 1.0))
        self.kd_feat = bool(getattr(args, "fedsa_kd_feat", True))
        self.kd_frozen = bool(getattr(args, "fedsa_kd_frozen", False))  # teach from the received (un-adapted) common model
        # replace part of the one-hot CE by distillation: (1 - a) * CE + a * KD (a = kd_mix), so a teacher that has not
        # memorized my samples actually lowers the pressure to memorize them (KD is no longer only added on top)
        self.kd_mix = float(getattr(args, "fedsa_kd_mix", 0.0))
        # private model = residual expert: it is trained on (received averaged twin logits + its own logits), so it only
        # has to fix what the not-yet-memorized twin gets wrong on my data (needs --fedsa_kd_frozen; use --fedsa_kd_weight 0)
        self.residual = bool(getattr(args, "fedsa_residual", False))
        # residual ablations: standalone = also train the private model alone (CE + KL to the base, FedAPEN-style loss);
        # live = sum with the locally-updating twin (detached) instead of the frozen received copy
        self.res_standalone = bool(getattr(args, "fedsa_res_standalone", False))
        self.res_live = bool(getattr(args, "fedsa_res_live", False))
        self.res_readout = float(getattr(args, "fedsa_res_readout", 0.0))
        self.res_feat = float(getattr(args, "fedsa_res_feat", 0.0))
        self.res_ce = float(getattr(args, "fedsa_res_ce", 0.0))
        # where the fixed anchors act: on the private model and/or on the shared twin (both 1 = default)
        self.priv_anchor = float(getattr(args, "fedsa_priv_anchor", 1.0))
        self.cm_anchor = float(getattr(args, "fedsa_cm_anchor", 1.0))
        # shared-term regularization: weight decay of the twin; memorization gate (server freezes the twin backbone at its
        # best held-out round; needs a held-out split of my training data)
        self.cm_wd = float(getattr(args, "fedsa_cm_wd", 0.0))
        self.cm_val = float(getattr(args, "fedsa_cm_val", 0.0))
        self.cm_frozen = False
        self._split = None
        if self.kd_mix > 0:
            self.kd_weight = self.kd_mix
        self._teacher = None
        self.private_anchors = bool(getattr(args, "fedsa_private_anchors", False))
        self.mem_dump_rounds = tuple(int(r) for r in str(getattr(args, "fedsa_mem_dump", "")).split(",") if r.strip())
        self.run_goal = getattr(args, "goal", "run")
        self.cm_invert = float(getattr(args, "fedsa_cm_invert", 0.0))
        self.inv_cfg = dict(n=int(getattr(args, "fedsa_inv_n", 256)), steps=int(getattr(args, "fedsa_inv_steps", 30)),
                            bn_w=float(getattr(args, "fedsa_inv_bn", 1.0)), start=int(getattr(args, "fedsa_inv_start", 20)),
                            jitter=int(getattr(args, "fedsa_inv_jitter", 0)), first_bn=float(getattr(args, "fedsa_inv_first_bn", 1.0)),
                            conf=float(getattr(args, "fedsa_inv_conf", 0.0)), chunk=int(getattr(args, "fedsa_inv_chunk", 128)))
        self._syn = None
        # A: every client inverts its OWN heterogeneous backbone into class images; the server swaps them between clients
        self.peer_invert = float(getattr(args, "fedsa_peer_invert", 0.0))
        # edge-ensemble distillation on unlabeled public images (targets in the fixed anchor space, set by the server)
        self.pub_kd = float(getattr(args, "fedsa_pub_kd", 0.0))
        self.pub_frac = float(getattr(args, "fedsa_pub_frac", 0.5))
        self.pub_set = None
        self.pub_space = getattr(args, "fedsa_pub_space", "anchor")
        # mechanism control (Ash & Adams 2020): shrink-and-perturb the backbone every round, no federation involved
        self.sp_alpha = float(getattr(args, "fedsa_shrink_perturb", 1.0))
        # federated variation generator (class-conditional VAE, FedAvg'd; trained only on the ELBO, not vs the classifier)
        self.vgen_w = float(getattr(args, "fedsa_vgen", 0.0))
        self.vgen, self.vgen_opt = None, None
        if self.vgen_w > 0:
            g = torch.random.get_rng_state()
            torch.manual_seed(54321)  # identical init in every client
            self.vgen = VarGen(self.num_classes)
            torch.random.set_rng_state(g)
            self.vgen_scale = float(getattr(args, "fedsa_vgen_scale", 1.0))
            self.vgen_temp = float(getattr(args, "fedsa_vgen_temp", 1.0))
            self.vgen_start = int(getattr(args, "fedsa_vgen_start", 10))
            # global: move toward the class's latent distribution pooled over ALL clients (needs a shared latent space);
            # local: around my own encoding (original). Per-class latent stats of my data are uploaded every round.
            self.vgen_mode = getattr(args, "fedsa_vgen_mode", "local")
            self.vgen_mix = float(getattr(args, "fedsa_vgen_mix", 0.5))
            self.vgen_replace = float(getattr(args, "fedsa_vgen_replace", 0.0))  # >0: fraction of the batch replaced
            self.vgen_noise_mag = float(getattr(args, "fedsa_vgen_noise_mag", 0.1))
            self.vgen_local = None
            self.vgen_stats, self.vgen_global = {}, None
        # federated adversarial augmenter (same tiny network in every client, FedAvg'd; in front of the backbone)
        self.faa = float(getattr(args, "fedsa_faa", 0.0))
        self.augmenter = None
        if self.faa > 0:
            g = torch.random.get_rng_state()
            torch.manual_seed(12345)  # identical init in every client
            self.augmenter = Augmenter(eps=float(getattr(args, "fedsa_faa_eps", 0.25)),
                                       mode=getattr(args, "fedsa_faa_mode", "pixel"))
            torch.random.set_rng_state(g)
            lr_mult = float(getattr(args, "fedsa_faa_lr_mult", 10.0))  # the adversary must move faster than the backbone
            for p in self.augmenter.parameters():
                p.register_hook(lambda gr, m=lr_mult: gr * m)
        # federated memorization reference: per-class training-loss pace vs the federation's pace for that class
        self.memref = float(getattr(args, "fedsa_memref", 0.0))
        self.memref_mode = getattr(args, "fedsa_memref_mode", "fed")  # fed | local (control: own mean loss)
        self.memref_global = None  # class -> federation mean training loss (set by the server)
        self.memref_stats = {}
        self.sp_sigma = float(getattr(args, "fedsa_sp_sigma", 0.1))
        self.sp_weights_only = bool(getattr(args, "fedsa_sp_weights_only", False))
        # anchor protocol: invert toward the fixed anchors and transfer anchor-space representations (needs a shared space)
        self.inv_anchor = bool(getattr(args, "fedsa_inv_anchor", False))
        # multi-depth shared coordinates: also exchange each stage's projected features (stages put in the anchor
        # space by the deep-anchor projectors), so the receiver matches the producer's intermediate behaviour too
        self.inv_deep = bool(getattr(args, "fedsa_inv_deep", False))
        self.syn_out, self.syn_in = None, None
        # relay inversion: before passing received images on, refine them a few steps under MY backbone, so an image
        # accumulates the agreement of several heterogeneous backbones (an ensemble inversion without sending models)
        self.inv_relay = int(getattr(args, "fedsa_inv_relay", 0))
        self.inv_relay_n = int(getattr(args, "fedsa_inv_relay_n", 128))
        # fast relay: fresh noise images and received images are optimized together for `inv_relay` steps in one pass
        self.inv_fast = bool(getattr(args, "fedsa_inv_fast", False))
        # training-time inversion: the input-gradient of my synthetic-image loss (computed anyway in backward) refines
        # the pooled images in place, so they keep being inverted at ~zero extra cost while I train on them
        self.inv_train = float(getattr(args, "fedsa_inv_train", 0.0))
        # use a received image only if my model already ranks its label in the top-k (quality gate), 0 = off
        self.syn_filter = int(getattr(args, "fedsa_syn_filter", 0))
        # relay the OLDEST held images (they keep being refined across hops instead of being dropped), else random
        self.relay_oldest = bool(getattr(args, "fedsa_inv_relay_oldest", False))
        self.inv_every = max(1, int(getattr(args, "fedsa_inv_every", 1)))  # synthesize/exchange every k rounds
        self.syn_frac = float(getattr(args, "fedsa_syn_frac", 0.5))  # synthetic images per step, as a fraction of the batch
        if int(getattr(args, "fedsa_kernel_bank", 0)) > 0:  # assemble all backbone convs from a shared kernel bank
            from utils.kernel_bank import convert as bank_convert
            m = getattr(self, "model", None) or load_item(self.role, "model", self.save_folder_name)
            m.kernel_bank = bank_convert(m.base[0], m=int(args.fedsa_kernel_bank), chunk=int(args.fedsa_bank_chunk))
            save_item(m, self.role, "model", self.save_folder_name)
        self.deep_anchor = float(getattr(args, "fedsa_deep_anchor", 0.0))  # >0: fixed-anchor loss on intermediate stages
        self._deep_stages, self._deep_proj, self._deep_cache = None, None, []
        self.kd_temp = 2.0
        self.second_view = getattr(args, "fedsa_second_view", "off")
        self.global_var = {}  # server-pooled per-class feature variance (for the stat view)
        self.proto_devs = {}  # class -> [n_clients, D] deviations of each client's prototype from the class mean
        self.peer_heads = []  # other clients' heads (weight, bias) in the shared anchor space
        self.peer_head_weight = float(getattr(args, "fedsa_peer_head", 0.0))
        self.bottleneck = getattr(args, "fedsa_bottleneck", "off")  # off | fixed (VIB-style) | fed (federated class prior)
        self.bottleneck_beta = float(getattr(args, "fedsa_bottleneck_beta", 1.0))
        self._logvar = None
        self.stat_view_scale = float(getattr(args, "fedsa_stat_view_scale", 1.0))
        self.shift_norm = bool(getattr(args, "fedsa_shift_norm", False))
        self.shift_fixed = float(getattr(args, "fedsa_shift_fixed", 0.0))
        self.backbone_view = float(getattr(args, "fedsa_backbone_view", 0.0))  # >0: view = noise on pre-projector feature
        self.relation_weight = float(getattr(args, "fedsa_relation", 0.0))  # anchor-free relational consistency
        self.relation_view = float(getattr(args, "fedsa_relation_view", 0.3))  # view noise norm relative to ||z||; 0 = no view
        self.global_rel = {}  # >0: view norm = c * anchor nearest distance (constant)  # keep cross-client direction, fix the energy
        self.view_norm_log = []  # per-batch (applied shift norm, VN-equivalent noise norm) for diagnostics  # off | ce (control) | anchor (ours)
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
        if self.frozen_head != "none" and self.common_model is not None:
            fix_head(self.common_model.head)
            if self.frozen_head == "both":
                fix_head(model.head)
        if self.sp_alpha < 1.0 and self.current_round > 0:
            with torch.no_grad():  # p <- alpha * p + sigma * std(p) * noise, backbone only
                for p in model.base[0].parameters():
                    if self.sp_weights_only and p.dim() < 2:
                        continue  # mild variant: conv/linear weights only, BN affine and biases untouched
                    p.mul_(self.sp_alpha).add_(torch.randn_like(p) * (self.sp_sigma * p.std() if p.numel() > 1 else 0.0))
        params = list(model.parameters())
        if self.vgen is not None:
            self.vgen.to(self.device).train()
            if self.vgen_opt is None:
                self.vgen_opt = torch.optim.Adam(self.vgen.parameters(), lr=1e-3)
        if self.augmenter is not None:
            self.augmenter.to(self.device).train()
            params += list(self.augmenter.parameters())
            if getattr(self, "_x_range", None) is None:
                xr = torch.stack([x for x, _ in trainloader.dataset]).transpose(0, 1).flatten(1)
                self._x_range = tuple(v.view(1, -1, 1, 1).to(self.device) for v in (xr.min(1).values, xr.max(1).values))
        if self.bottleneck != "off":
            if self._logvar is None:  # per-client log-variance head on the backbone feature (architecture-specific width)
                self._logvar = nn.Linear(model.base[-1].in_features, model.base[-1].out_features).to(self.device)
                nn.init.zeros_(self._logvar.weight)
                nn.init.constant_(self._logvar.bias, -2.0)
            params += list(self._logvar.parameters())
        deep_hooks = []
        if self.deep_anchor > 0:
            if self._deep_stages is None:
                xb, yb = self._move_input(*next(iter(trainloader)))
                self._deep_stages = find_stages(model.base[0], xb)
                self._deep_proj = make_projectors(self._deep_stages, model.base[-1].out_features)
                print(f"[deep_anchor] client {self.id} stages {self._deep_stages}")
            self._deep_proj.to(self.device).train()
            params += list(self._deep_proj.parameters())
            deep_hooks = deep_attach(model.base[0], self._deep_stages, self._deep_cache)
        optimizer = torch.optim.SGD(params, lr=self.learning_rate,
                                    weight_decay=float(getattr(self, 'weight_decay', 0.0)))
        cm = self.common_model
        if cm is not None:
            cm.to(self.device).train()
            for p in cm.base[0].parameters():
                p.requires_grad_(not self.cm_frozen)
            if self.cm_frozen:
                cm.base[0].eval()  # frozen backbone also keeps its BN statistics
            optimizer_c = torch.optim.SGD([p for p in cm.parameters() if p.requires_grad], lr=self.learning_rate * self.cm_lr_scale,
                                          weight_decay=self.cm_wd)
            # snapshot of the FedAvg'd common model before it fits this client's data (has not memorized it)
            self._teacher = copy.deepcopy(cm).eval().requires_grad_(False) if self.kd_frozen else None
            self._syn = None
            if self.cm_invert > 0 and self.current_round >= self.inv_cfg["start"]:
                self._syn = self._make_synthetic(cm, trainloader)
        if self.peer_invert > 0:
            self._syn = self.syn_in  # other clients' backbone inversions of my classes (set by the server)
            self.syn_out = None  # only rounds that synthesize send anything
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        model.train()

        self.cal_local_gap()
        self.cal_global_gap()

        max_local_epochs = self.local_epochs
        if self.train_slow and max_local_epochs > 1:
            max_local_epochs = int(self.rng.randint(1, max(2, max_local_epochs // 2 + 1)))

        proto_sum = None
        proto_count = None
        soft_targets = self._prototype_soft_targets() if self.soft_label > 0 else None
        rel_ctx = self._relation_context() if self.relation_weight > 0 else None
        rel_sum = torch.zeros(self.num_classes, self.num_classes, device=self.device) if rel_ctx else None
        rel_cnt = torch.zeros(self.num_classes, device=self.device) if rel_ctx else None
        metric_sums = defaultdict(float)
        metric_batches = 0
        ml_sum = torch.zeros(self.num_classes, device=self.device)
        ml_cnt = torch.zeros(self.num_classes, device=self.device)
        mem_w = self._memref_weights() if self.memref > 0 else None

        try:
            for _ in range(max_local_epochs):
                for x, y in trainloader:
                    x, y = self._move_input(x, y)
                    if self.train_slow:
                        time.sleep(0.1 * abs(self.rng.rand()))

                    x_raw = x
                    if self.augment:
                        x = self._augment(x)
                    if self.vgen is not None and self.vgen_replace > 0:  # swap part of the batch for variations: no extra pass
                        self._vgen_update(x_raw, y)
                        if self.current_round >= self.vgen_start:
                            swap = torch.rand(x.shape[0], device=x.device) < self.vgen_replace
                            if swap.any():
                                x = x.clone()
                                x[swap] = self._vgen_view(x_raw[swap], y[swap])
                    mixed = None
                    if self.mix != "off" and float(torch.rand(())) < self.mix_prob:
                        x, y_b, lam = mix_batch(x, y, self.mix)
                        mixed = (y_b, lam)
                    self._deep_cache.clear()
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        if self.align_detach:
                            # anchor losses train only the projector (base[-1]); backbone gets CE only
                            feat = model.base[:-1](x)
                            rep = model.base[-1](feat)
                            align_rep = model.base[-1](feat.detach())
                        elif self.backbone_view > 0:
                            # diversity from the backbone space (not pulled by anchors); consistency in the anchor space
                            feat = model.base[:-1](x)
                            rep = model.base[-1](feat)
                            align_rep = rep
                            d = torch.randn_like(feat)
                            feat2 = feat + d / d.norm(dim=1, keepdim=True).clamp_min(1e-8) * feat.norm(dim=1, keepdim=True) * self.backbone_view
                            rep2_bb = model.base[-1](feat2)
                        elif self.bottleneck != "off":
                            # stochastic bottleneck: rep = mean (used for prototypes/eval), z = sampled code used for training
                            feat = model.base[:-1](x)
                            rep = model.base[-1](feat)
                            logvar = self._logvar(feat).clamp(-8.0, 4.0)
                            align_rep = rep + torch.randn_like(rep) * (0.5 * logvar).exp()
                            bottleneck_kl = self._bottleneck_kl(rep, logvar, y)
                        else:
                            rep = model.base(x)
                            align_rep = rep
                        output = model.head(align_rep if self.bottleneck != "off" else rep)
                        out_priv = output  # the private model's own logits (before the residual sum)
                        if self.residual and cm is not None and self._teacher is not None:
                            src = cm if self.res_live else self._teacher
                            # shortcut: second base forward (also done in _mutual_distill_loss), cache it if this becomes the method
                            with torch.no_grad():
                                base_rep = src.base(x).float()
                                base_logits = src.head(base_rep).float()
                            if self.res_sum_loss:  # False: the private model trains on its own logits; the sum is used at test only
                                output = output + base_logits
                        task_loss = self.loss(output, y) if soft_targets is None else \
                            -(soft_targets.index_select(0, y) * F.log_softmax(output.float(), dim=1)).sum(1).mean()
                        anchors_loss, anchor_metrics = self.cal_anchors_loss(align_rep, model, y)
                        anchors_loss = anchors_loss * self.priv_anchor  # 0: private model trained without anchors
                        if mixed is not None:
                            y_b, lam = mixed
                            task_loss = lam * task_loss + (1 - lam) * self.loss(output, y_b)
                            anchors_loss = lam * anchors_loss + (1 - lam) * self.cal_anchors_loss(align_rep, model, y_b)[0]
                        loss = ((1.0 - self.kd_mix) if (cm is not None and self.kd_mix > 0) else 1.0) * task_loss + anchors_loss
                        if self.residual and self.res_standalone and cm is not None and self._teacher is not None:
                            loss = loss + (self.loss(out_priv, y) if self.res_sum_loss else 0) + self.res_kl * F.kl_div(
                                F.log_softmax(out_priv.float(), 1), F.softmax(base_logits, 1), reduction="batchmean")
                        if self.residual and cm is not None and self._teacher is not None:
                            # anchor-free supervision of the private representation, using the received twin as reference
                            if self.res_readout > 0:  # private features must be readable by the twin's (frozen) head
                                loss = loss + self.res_readout * self.loss(self._teacher.head(rep), y)
                            if self.res_feat > 0:  # align with the non-memorizing twin's features, per sample
                                loss = loss + self.res_feat * (1 - F.cosine_similarity(rep.float(), base_rep, dim=1)).mean()
                            if self.res_ce > 0:  # plain standalone CE (control; no KL)
                                loss = loss + self.res_ce * self.loss(out_priv, y)
                        if self.augmenter is not None:
                            loss = loss + self.faa * self._faa_loss(model, x, y)
                        if self.vgen is not None and self.vgen_replace <= 0:
                            vl = self._vgen_loss(model, x_raw, y)
                            if vl is not None:
                                loss = loss + self.vgen_w * vl
                        if self.pub_kd > 0 and self.pub_set is not None:
                            loss = loss + self.pub_kd * self._pub_loss(model, x.shape[0])
                        if self.memref > 0:
                            ce_i = F.cross_entropy(output.float(), y, reduction="none").detach()
                            ml_sum.index_add_(0, y, ce_i)
                            ml_cnt.index_add_(0, y, torch.ones_like(ce_i))
                            if mem_w is not None:  # extra label smoothing only where I learn faster than the reference
                                w = mem_w.index_select(0, y)
                                loss = loss + self.memref * (w * -F.log_softmax(output.float(), 1).mean(1)).mean()
                        if deep_hooks:  # first len(stages) captures belong to the main forward above
                            for proj, f in zip(self._deep_proj, self._deep_cache[:len(self._deep_stages)]):
                                loss = loss + self.deep_anchor * self.cal_anchors_loss(proj(f), model, y)[0]
                        if self.anchor_mixup > 0 and self.semantic_anchors is not None:
                            loss = loss + self._anchor_mixup_loss(rep, y, model)
                        if cm is not None:
                            loss = loss + self._mutual_distill_loss(rep, output, x, y, cm, out_priv)
                        if self._syn is not None:
                            loss = loss + (self.cm_invert or self.peer_invert) * self._synthetic_loss(model, x.shape[0])
                            self._kd_acc = {**(getattr(self, "_kd_acc", None) or {}), "syn_teacher_acc": self._syn_acc,
                                            **({"syn_recv_acc": self._syn_recv_acc} if hasattr(self, "_syn_recv_acc") else {}),
                                            **getattr(self, "_t_inv", {})}
                        if self.bottleneck != "off":
                            loss = loss + self.bottleneck_beta * bottleneck_kl
                        if self.backbone_view > 0:
                            lg1, lg2 = model.head(rep), model.head(rep2_bb)
                            consist = F.kl_div(F.log_softmax(lg2.float(), 1), F.softmax(lg1.detach().float(), 1), reduction="batchmean")
                            loss = loss + self.loss(lg2, y) + self.cal_anchors_loss(rep2_bb, model, y)[0] + consist
                        if rel_ctx:
                            rel_loss, rel_main = self._relation_loss(rep, y, rel_ctx)
                            loss = loss + self.relation_weight * rel_loss
                            with torch.no_grad():
                                full = torch.zeros(y.numel(), self.num_classes, device=rep.device)
                                full[:, rel_ctx["cols"]] = rel_main.detach().float()
                                rel_sum.index_add_(0, y, full)
                                rel_cnt.index_add_(0, y, torch.ones_like(y, dtype=rel_cnt.dtype))
                        if self.peer_head_weight > 0 and self.peer_heads:
                            # other clients' heads act as auxiliary graders of my features (my own head is untouched)
                            W, b = self.peer_heads[int(torch.randint(len(self.peer_heads), (1,)))]
                            peer_logits = F.linear(rep, W.to(rep.device, rep.dtype), b.to(rep.device, rep.dtype))
                            loss = loss + self.peer_head_weight * self.loss(peer_logits, y)
                        if self.second_view in ("stat", "noise", "local", "stat_ce", "proto_shift", "noise_kl", "noise_fixed"):
                            # augmentation-free second view in feature space: stat = per-class variance pooled from
                            # all clients (shared knowledge); noise = control with isotropic noise of the same energy
                            loss = loss + self._stat_view_loss(rep, y, model)
                        elif self.second_view != "off":
                            # a second, independently augmented view of the same images
                            rep2 = model.base(self._augment(x_raw))
                            if self.second_view == "ce":  # control: just more augmented data
                                loss = loss + self.loss(model.head(rep2), y)
                            elif self.second_view == "consist":  # ablation: views agree with each other, no shared target
                                loss = loss + self.loss(model.head(rep2), y) + F.mse_loss(rep2.float(), rep.float())
                            elif self.second_view == "proto":  # ablation: shared target = moving global prototypes
                                loss = loss + self.loss(model.head(rep2), y) + F.mse_loss(rep2.float(), rep.float()) \
                                    + self._global_proto_mse(rep2, y)
                            else:  # ours: both views bound to the same shared anchor + to each other
                                loss = loss + self.loss(model.head(rep2), y) \
                                    + self.cal_anchors_loss(rep2, model, y)[0] + F.mse_loss(rep2.float(), rep.float())

                    with torch.no_grad():
                        metric_sums["train_acc"] += float(output.argmax(dim=1).eq(y).float().mean())
                        if getattr(self, "_kd_acc", None):  # memorization diagnostic: teacher / common-copy fit on local train data
                            for k, v in self._kd_acc.items():
                                metric_sums[k] += v
                        rep_detached = rep.detach().float()
                        if proto_sum is None:
                            proto_sum = torch.zeros(
                                self.num_classes,
                                rep_detached.shape[1],
                                device=rep_detached.device,
                                dtype=torch.float32,
                            )
                            proto_count = torch.zeros(self.num_classes, device=rep_detached.device, dtype=torch.float32)
                            proto_sq = torch.zeros_like(proto_sum)
                        proto_sum.index_add_(0, y, rep_detached)
                        proto_sq.index_add_(0, y, rep_detached.square())
                        proto_count.index_add_(0, y, torch.ones_like(y, dtype=proto_count.dtype))

                    optimizer.zero_grad(set_to_none=True)
                    if cm is not None:
                        optimizer_c.zero_grad(set_to_none=True)
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
                        if cm is not None:
                            optimizer_c.step()

            if self.eval_protos:
                # FedDSA-style: recollect prototypes with the final model in eval mode
                model.eval()
                proto_sum = proto_count = None
                with torch.no_grad():
                    for x, y in self.load_train_data(drop_last=False):
                        x, y = self._move_input(x, y)
                        rep = model.base(x).float()
                        if proto_sum is None:
                            proto_sum = torch.zeros(self.num_classes, rep.shape[1], device=rep.device)
                            proto_count = torch.zeros(self.num_classes, device=rep.device)
                        proto_sum.index_add_(0, y, rep)
                        proto_count.index_add_(0, y, torch.ones_like(y, dtype=proto_count.dtype))
            self.protos = self._protos_from_sums(proto_sum, proto_count)
            if self.memref > 0:
                self.memref_stats = {int(c): (float(ml_cnt[c]), float(ml_sum[c]))
                                     for c in torch.nonzero(ml_cnt > 0).view(-1).tolist()}
            self.rel_stats = {} if not rel_ctx else {int(c): (float(rel_cnt[c]), rel_sum[c].detach().cpu())
                                                     for c in torch.nonzero(rel_cnt > 0).view(-1).tolist()}
            # per-class sufficient statistics (count, sum, sum of squares) for the server's pooled variance
            self.proto_stats = {} if proto_count is None or self.eval_protos else {
                int(c): (float(proto_count[c]), proto_sum[c].detach().cpu(), proto_sq[c].detach().cpu())
                for c in torch.nonzero(proto_count > 0).view(-1).tolist()}
            denom = max(metric_batches, 1)
            self.latest_train_metrics = {
                key: value / denom for key, value in metric_sums.items()
            }
            if self.view_norm_log:
                self.latest_train_metrics["view_shift_norm"] = sum(a for a, _ in self.view_norm_log) / len(self.view_norm_log)
                self.latest_train_metrics["view_noise_norm"] = sum(b for _, b in self.view_norm_log) / len(self.view_norm_log)
                self.view_norm_log = []
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
            for h in deep_hooks:
                h.remove()
            self._deep_cache.clear()
            self.release_model_from_device(model)

        if self.peer_invert > 0 and self.current_round + 1 >= self.inv_cfg["start"] \
                and (self.current_round + 1 - self.inv_cfg["start"]) % self.inv_every == 0:
            if self.inv_fast and self.inv_relay > 0:
                if self.syn_in is not None:
                    n_in = len(self.syn_in[1])
                    pick = (torch.arange(min(n_in, self.inv_relay_n), device=self.syn_in[1].device) if self.relay_oldest
                            else torch.randperm(n_in, device=self.syn_in[1].device)[: self.inv_relay_n])
                    init = (self.syn_in[0][pick], self.syn_in[1][pick])
                else:
                    init = (torch.empty((0,) + tuple(trainloader.dataset[0][0].shape)), torch.empty(0, dtype=torch.long))
                out = self._make_synthetic(model, trainloader, init=init)
            else:
                out = self._make_synthetic(model, trainloader)
            if not self.inv_fast and self.inv_relay > 0 and self.syn_in is not None:
                pick = torch.randperm(len(self.syn_in[1]), device=self.syn_in[1].device)[: self.inv_relay_n]
                rel = self._make_synthetic(model, trainloader, init=(self.syn_in[0][pick], self.syn_in[1][pick]))
                out = tuple(torch.cat([a, b]) for a, b in zip(out, rel))
            self.syn_out = tuple(v.cpu() for v in out)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    @staticmethod
    def _augment(x):
        return augment_batch(x)

    @torch.no_grad()
    def _relation_context(self):
        """Own class prototypes (last round) as the reference set: a sample's relational code is its softmax cosine
        similarity to them. Comparable across heterogeneous models without a shared coordinate system."""
        protos = getattr(self, "protos", None) or {}
        if len(protos) < 2:
            return None
        cols = sorted(protos)
        P = F.normalize(torch.stack([torch.as_tensor(protos[c], dtype=torch.float32).view(-1) for c in cols]).to(self.device), dim=1)
        return {"cols": torch.tensor(cols, device=self.device), "P": P, "colset": set(cols)}

    def _relation_loss(self, rep, y, ctx, tau=0.1):
        code = lambda z: torch.softmax(F.normalize(z.float(), dim=1) @ ctx["P"].t() / tau, dim=1)
        main = code(rep)
        views = [main]
        loss = rep.new_zeros(())
        if self.relation_view > 0:
            # view: random direction with norm relative to the feature itself (no anchor involved)
            d = torch.randn_like(rep)
            view = code(rep + d / d.norm(dim=1, keepdim=True).clamp_min(1e-8) * rep.norm(dim=1, keepdim=True) * self.relation_view)
            loss = F.kl_div(view.clamp_min(1e-8).log(), main.detach(), reduction="batchmean")  # view consistency
            views.append(view)
        keep = torch.tensor([int(c) in self.global_rel and int(c) in ctx["colset"] for c in y.tolist()], device=rep.device)
        if bool(keep.any()):  # match the federation's relational structure of the class
            tgt = torch.stack([self.global_rel[int(c)] for c in y[keep].tolist()]).to(rep.device)[:, ctx["cols"]]
            tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-8)
            for q in (v[keep] for v in views):
                loss = loss + F.kl_div(q.clamp_min(1e-8).log(), tgt, reduction="batchmean")
        return loss, main

    def _global_proto_mse(self, rep, y):
        protos = getattr(self, "global_protos", None) or {}
        have = torch.tensor([int(c) in protos for c in y.tolist()], device=rep.device)
        if not bool(have.any()):
            return rep.new_zeros(())
        target = torch.stack([torch.as_tensor(protos[int(c)], dtype=torch.float32).view(-1) for c in y[have].tolist()]).to(rep.device)
        return F.mse_loss(rep[have].float(), target)

    def _bottleneck_kl(self, mu, logvar, y):
        """KL( N(mu, e^logvar) || N(anchor_y, s^2) ), mean over dims and batch. fed: s^2 = per-class, per-dim variance
        pooled from all clients (the federation's view of the class); fixed: one scalar of the same overall scale."""
        anchors = self.semantic_anchors[: self.num_classes].to(mu.device, mu.dtype).index_select(0, y)
        if len(self.global_var) >= self.num_classes:
            table = torch.stack([self.global_var[c] for c in range(self.num_classes)]).to(mu.device, mu.dtype).clamp_min(1e-3)
            s2 = table.index_select(0, y) if self.bottleneck == "fed" else table.mean()
        else:
            s2 = torch.ones((), device=mu.device, dtype=mu.dtype)
        var = logvar.exp()
        kl = 0.5 * ((var + (mu - anchors).square()) / s2 - 1.0 - logvar + torch.log(s2))
        return kl.mean()

    def _vgen_update(self, x_raw, y):
        """One ELBO step of the shared generator on my real images (own optimizer, no classifier gradient) and, in
        global mode, accumulation of my per-class latent statistics."""
        with torch.enable_grad():
            gl, rec = self.vgen.elbo_loss(x_raw, y)
            self.vgen_opt.zero_grad(set_to_none=True)
            gl.backward()
            self.vgen_opt.step()
        self._kd_acc = {**(getattr(self, "_kd_acc", None) or {}), "vgen_rec": rec}
        if self.vgen_mode in ("global", "transport"):  # accumulate my per-class latent statistics (sum, sum sq, count)
            mu = self.vgen.encode_mu(x_raw).flatten(1)
            if not isinstance(self.vgen_stats, dict) or "sum" not in self.vgen_stats:
                d = mu.shape[1]
                self.vgen_stats = {"sum": torch.zeros(self.num_classes, d, device=mu.device),
                                   "sq": torch.zeros(self.num_classes, d, device=mu.device),
                                   "n": torch.zeros(self.num_classes, device=mu.device), "shape": self.vgen.encode_mu(x_raw[:1]).shape[1:]}
            self.vgen_stats["sum"].index_add_(0, y, mu)
            self.vgen_stats["sq"].index_add_(0, y, mu * mu)
            self.vgen_stats["n"].index_add_(0, y, torch.ones_like(y, dtype=mu.dtype))

    def _vgen_view(self, x_raw, y):
        if self.vgen_mode == "noise":
            xv = VarGen.lowfreq_noise(x_raw, self.vgen_noise_mag)
        elif self.vgen_mode == "transport" and self.vgen_global is not None and getattr(self, "vgen_local", None) is not None:
            (gm, gs), (lm, ls) = self.vgen_global, self.vgen_local
            d = x_raw.device
            xv = self.vgen.transport(x_raw, y, gm.to(d), gs.to(d), lm.to(d), ls.to(d), self.vgen_mix, self.vgen_scale)
        elif self.vgen_mode == "global" and self.vgen_global is not None:
            mean, std = self.vgen_global
            xv = self.vgen.vary_toward(x_raw, y, mean.to(x_raw.device), std.to(x_raw.device), self.vgen_mix, self.vgen_scale)
        else:
            xv = self.vgen.vary(x_raw, y, self.vgen_scale, self.vgen_temp)
        self._kd_acc["vgen_delta"] = float((xv - x_raw).abs().mean())
        return self._augment(xv) if self.augment else xv

    def _vgen_loss(self, model, x_raw, y):
        """Extra backbone loss on generator variations of my images (BN buffers untouched)."""
        self._vgen_update(x_raw, y)
        if self.current_round < self.vgen_start:
            return None
        xv = self._vgen_view(x_raw, y)
        with frozen_bn(model):
            rep = model.base(xv)
            out = model.head(rep)
        with torch.no_grad():
            self._kd_acc["vgen_acc"] = float(out.argmax(1).eq(y).float().mean())
        return self.loss(out, y) + self.cal_anchors_loss(rep, model, y)[0]

    def _faa_loss(self, model, x, y):
        """Backbone loss on the augmenter's view of my real (already standard-augmented) images. Gradient reversal
        inside the augmenter turns this into min (backbone) - max (augmenter). BN buffers are not touched."""
        lo, hi = self._x_range
        xa = self.augmenter(x, lo, hi)
        with frozen_bn(model):
            rep = model.base(xa)
            out = model.head(rep)
        with torch.no_grad():
            self._kd_acc = {**(getattr(self, "_kd_acc", None) or {}), "faa_adv_acc": float(out.argmax(1).eq(y).float().mean()),
                            "faa_delta": float((xa - x).abs().mean())}
        return self.loss(out, y) + self.cal_anchors_loss(rep, model, y)[0]

    @torch.no_grad()
    def class_counts(self):
        if getattr(self, "_class_counts", None) is None:
            y = torch.cat([b.view(-1) for _, b in self._read_client_data_cached(is_train=True)])
            self._class_counts = torch.bincount(y, minlength=self.num_classes).float()
        return self._class_counts

    @torch.no_grad()
    def build_peer_set(self, peers):
        """Peer distillation (models move inside the edge LAN, data never leaves): targets for MY real training
        images = class-count-weighted average of my edge peers' fixed-anchor-space outputs. Peers never saw these
        samples, so their outputs are not memorized. peers: [(model, class_counts)]; [(my own model, 1)] = control."""
        data = self._read_client_data_cached(is_train=True)
        X = torch.stack([x for x, _ in data]); Y = torch.cat([y.view(-1) for _, y in data])
        W = torch.stack([cnt.float().cpu()[Y] for _, cnt in peers])  # [P, N]
        valid = W.sum(0) > 0
        T = None
        for (m, _), w in zip(peers, W):
            was = m.training
            m.to(self.device).eval()
            # logit space (default for peers): class scores are the one coordinate system every architecture shares,
            # so no anchors / projector alignment are needed; anchor space kept as an option
            f = (lambda z: m.head(m.base(z))) if self.pub_space == "logit" else m.base
            reps = torch.cat([f(X[i:i + 256].to(self.device)).float().cpu() for i in range(0, len(X), 256)])
            m.train(was)
            T = w[:, None] * reps if T is None else T + w[:, None] * reps
        self.pub_set = (X[valid], T[valid] / W.sum(0)[valid][:, None])

    def _pub_loss(self, model, batch):
        """Match ensemble anchor-space targets: on unlabeled public images (edge mode) or on my own training images
        (peer mode, standard augmentation applied to the input). BN buffers untouched."""
        xs, tgt = self.pub_set
        k = max(1, int(batch * self.pub_frac))
        idx = torch.randint(len(xs), (k,))
        xb = xs[idx.to(xs.device)].to(self.device)
        if xs.device.type == "cpu" and self.augment:  # peer mode: my own images, same augmentation as training
            xb = self._augment(xb)
        tgt = tgt[idx.to(tgt.device)].to(self.device)
        idx = torch.arange(k, device=self.device)
        with frozen_bn(model):
            rep = model.base(xb[idx])
            if self.pub_space == "logit":  # baseline: match the ensemble's (personalized-head) logits with KL
                T = self.kd_temp
                out = model.head(rep).float()
                return F.kl_div(F.log_softmax(out / T, 1), F.softmax(tgt[idx] / T, 1), reduction="batchmean") * T * T
        return F.mse_loss(rep.float(), tgt[idx])

    def _memref_weights(self):
        """w_c = clamp(1 - my_loss_c / ref_c, 0, 1): 0 when I learn class c no faster than the reference, ->1 when my
        loss on c is near 0 while the reference is not (memorizing). ref = federation's mean loss on c (fed) or my own
        mean loss over all classes (local control). Uses last round's statistics."""
        if not self.memref_stats:
            return None
        mine = {c: s / n for c, (n, s) in self.memref_stats.items()}
        if self.memref_mode == "local":
            m = sum(mine.values()) / len(mine)
            ref = {c: m for c in mine}
        else:
            if not self.memref_global:
                return None
            ref = self.memref_global
        w = torch.zeros(self.num_classes, device=self.device)
        for c, l in mine.items():
            if c in ref and ref[c] > 0:
                w[c] = min(max(1.0 - l / ref[c], 0.0), 1.0)
        self._kd_acc = {**(getattr(self, "_kd_acc", None) or {}), "memref_w": float(w[list(mine)].mean())}
        return w

    def _make_synthetic(self, src, trainloader, init=None):
        """Invert `src` (the received FedAvg'd common model, or my own private backbone for peer exchange) into
        inputs of my own classes, drawn with my label frequencies. Teacher logits are kept for distillation."""
        sync = (lambda: torch.cuda.synchronize()) if self.device.type == "cuda" else (lambda: None)
        sync(); t0 = time.time()
        teacher = copy.deepcopy(src).eval().requires_grad_(False)
        sync(); t1 = time.time()
        if getattr(self, "_labels_cache", None) is None:
            self._labels_cache = torch.cat([y.view(-1) for _, y in trainloader.dataset])
        labels = self._labels_cache
        self.local_labels = labels.unique()
        shape = trainloader.dataset[0][0].shape
        ys = labels[torch.randint(len(labels), (self.inv_cfg["n"],))].to(self.device)
        steps, x0 = self.inv_cfg["steps"], None
        if init is not None:  # relay: refine received images (their labels) for a few steps instead of starting from noise
            if self.inv_fast:  # one pass: fresh noise images + received images, all for `inv_relay` steps
                x0 = torch.cat([torch.randn((len(ys),) + tuple(shape), device=self.device), init[0].to(self.device)])
                ys = torch.cat([ys, init[1].to(self.device)])
            else:
                x0, ys = init[0], init[1].to(self.device)
            steps = self.inv_relay
        if getattr(self, "_x_range", None) is None:  # per-channel range of my real inputs
            xr = torch.stack([x for x, _ in trainloader.dataset]).transpose(0, 1).flatten(1)
            self._x_range = tuple(v.view(1, -1, 1, 1).to(self.device) for v in (xr.min(1).values, xr.max(1).values))
        lo, hi = self._x_range
        anchors = None
        if self.inv_anchor and self.semantic_anchors is not None:
            anchors = self.semantic_anchors[: self.num_classes].to(self.device).float()
        with torch.enable_grad():
            xs = synthesize(teacher, ys, shape, steps=steps, bn_w=self.inv_cfg["bn_w"], lo=lo, hi=hi,
                            anchors=anchors, jitter=self.inv_cfg["jitter"], first_bn=self.inv_cfg["first_bn"],
                            init=x0, chunk=self.inv_cfg["chunk"])
        sync(); t2 = time.time()
        with torch.no_grad():
            reps = teacher.base(xs)
            logits = teacher.head(reps)
        sync()
        self._t_inv = {"t_copy": t1 - t0, "t_inv": t2 - t1, "t_post": time.time() - t2}
        self._syn_acc = float(logits.argmax(1).eq(ys).float().mean())
        if self.inv_cfg["conf"] > 0:  # send only images my own backbone confidently assigns to their class
            ok = F.softmax(logits.float(), 1).gather(1, ys.view(-1, 1)).view(-1) >= self.inv_cfg["conf"]
            if ok.any():
                xs, ys, logits, reps = xs[ok], ys[ok], logits[ok], reps[ok]
        if self.inv_deep and self._deep_stages:
            cache = []
            hooks = deep_attach(teacher.base[0], self._deep_stages, cache)
            try:
                with torch.no_grad():
                    teacher.base(xs)
                    zd = torch.stack([p(f).float() for p, f in zip(self._deep_proj, cache[:len(self._deep_stages)])], 1)
                    zd = torch.cat([zd, zd[:, -1:].expand(-1, 2 - zd.shape[1], -1)], 1) if zd.shape[1] < 2 else zd  # FedAvgCNN has 1 stage: pad to 2 slots
            finally:
                for h in hooks:
                    h.remove()
            return xs, ys, logits, reps, zd  # zd: [N, stages, D] in the shared (anchor-aligned) stage coordinates
        return xs, ys, logits, reps

    def _synthetic_loss(self, model, batch):
        xs, ys, lt, rp = self._syn[:4]
        k = max(1, int(batch * self.syn_frac))
        idx = (torch.randperm(len(ys), device=ys.device)[:k] if self.inv_train > 0 and len(ys) >= k
               else torch.randint(len(ys), (k,), device=ys.device))
        x = xs[idx]
        if self.inv_train > 0:
            x = x.detach().clone().requires_grad_(True)
            lo, hi = self._x_range
            step = self.inv_train

            def refine(g, idx=idx, x0=x):  # descend MY loss on these images = invert toward their class under my backbone
                with torch.no_grad():
                    xs[idx] = torch.max(torch.min(x0.detach() - step * g.sign(), hi), lo)
            x.register_hook(refine)
        x = self._augment(x) if self.augment else x
        n0 = len(self._deep_cache)
        with frozen_bn(model):
            rep = model.base(x)
            out = model.head(rep)
        if self.syn_filter > 0:  # quality gate: keep images whose label my model already ranks in its top-k
            with torch.no_grad():
                keep = out.topk(self.syn_filter, 1).indices.eq(ys[idx].view(-1, 1)).any(1)
            self._kd_acc = {**(getattr(self, "_kd_acc", None) or {}), "syn_keep": float(keep.float().mean())}
            if not keep.any():
                return out.sum() * 0.0
            out, rep, idx = out[keep], rep[keep], idx[keep]
        base = self.loss(out, ys[idx]) + self.cal_anchors_loss(rep, model, ys[idx])[0]
        if self.inv_anchor:
            # the producer's representation of this image, read in the shared anchor space (not its personal head)
            extra = F.mse_loss(rep.float(), rp[idx].float())
            S = len(self._deep_stages or [])
            if len(self._syn) > 4 and S and len(self._deep_cache) - n0 >= S:
                zd = self._syn[4][idx]
                feats = self._deep_cache[n0:n0 + S]  # this synthetic forward's stage outputs (deep hooks are attached)
                extra = extra + sum(F.mse_loss(p(f).float(), zd[:, k]) for k, (p, f) in enumerate(zip(self._deep_proj, feats)))
            return base + extra
        T = self.kd_temp
        kd = F.kl_div(F.log_softmax(out.float() / T, 1), F.softmax(lt[idx].float() / T, 1), reduction="batchmean") * T * T
        return base + kd

    def _mutual_distill_loss(self, rep, output, x, y, cm, out_priv=None):
        """Private model <-> shared small model (FedAvg'd by everyone, so it has seen all clients' data):
        both learn the task + the shared anchors, and distill each other on logits (and, by default, in the
        shared anchor space). Returns the sum of both models' extra losses."""
        rep_c = cm.base(x)
        out_c = cm.head(rep_c)
        T = self.kd_temp
        kl = lambda student, teacher: F.kl_div(F.log_softmax(student.float() / T, 1),
                                               F.softmax(teacher.detach().float() / T, 1), reduction="batchmean") * T * T
        common_task = self.loss(out_c, y) + self.cm_anchor * self.cal_anchors_loss(rep_c, cm, y)[0]
        self._kd_acc = {"cm_train_acc": float(out_c.argmax(1).eq(y).float().mean())}
        if self.cm_from_priv > 0 and out_priv is not None:  # bidirectional (FML/FedAPEN): the twin also learns from the private model
            from_priv = F.kl_div(F.log_softmax(out_c.float(), 1), F.softmax(out_priv.detach().float(), 1), reduction="batchmean")
            common_task = common_task + self.cm_from_priv * from_priv
            self._kd_acc["cm_from_priv_kl"] = float(from_priv)
        if self.cm_ntd > 0 and self._teacher is not None:  # twin keeps the received twin's knowledge of absent classes
            with torch.no_grad():
                out_t = self._teacher.head(self._teacher.base(x))
            common_task = common_task + self.cm_ntd * ntd_loss(out_c, out_t, y)
        if self.kd_weight == 0:  # residual mode: no distillation term, skip the extra teacher forward
            return common_task
        if self._teacher is not None:
            # private learns from the frozen received model; the common copy still learns from the private (FML direction)
            with torch.no_grad():
                rep_t = self._teacher.base(x)
                out_t = self._teacher.head(rep_t)
            self._kd_acc["teacher_train_acc"] = float(out_t.argmax(1).eq(y).float().mean())
            distill = kl(output, out_t) + kl(out_c, output)
            if self.kd_feat:
                distill = distill + F.mse_loss(rep.float(), rep_t.float()) + F.mse_loss(rep_c.float(), rep.detach().float())
            return common_task + self.kd_weight * distill
        distill = kl(output, out_c) + kl(out_c, output)
        if self.kd_feat:
            distill = distill + F.mse_loss(rep.float(), rep_c.detach().float()) + F.mse_loss(rep_c.float(), rep.detach().float())
        return common_task + self.kd_weight * distill

    def _stat_view_loss(self, rep, y, model):
        if self.second_view == "proto_shift":
            # view = move each feature along how its class looks on another client (cross-client prototype deviation)
            if not self.proto_devs:
                return rep.new_zeros(())
            shift = torch.zeros_like(rep)
            for c in y.unique().tolist():
                devs = self.proto_devs.get(int(c))
                if devs is None or devs.shape[0] < 2:
                    continue
                idx = (y == c).nonzero().view(-1)
                pick = torch.randint(devs.shape[0], (idx.numel(),))
                shift[idx] = devs[pick].to(rep.device, rep.dtype)
            if len(self.global_var) >= self.num_classes:
                var_mean = torch.stack([self.global_var[c] for c in range(self.num_classes)]).to(rep.device, rep.dtype)[y].mean(1, keepdim=True)
                target = (var_mean.clamp_min(0) * rep.shape[1]).sqrt()  # norm of the matching isotropic (VN) noise
                n = shift.norm(dim=1, keepdim=True)
                self.view_norm_log.append((float(n.mean()), float(target.mean())))  # raw cross-client deviation vs VN noise
                if self.shift_fixed > 0:  # magnitude tied to the fixed anchors, so it never shrinks with feature collapse
                    shift = torch.where(n > 0, shift / n.clamp_min(1e-8) * (self.shift_fixed * float(self.global_gap)), shift)
                elif self.shift_norm:
                    shift = torch.where(n > 0, shift / n.clamp_min(1e-8) * target, shift)
            rep2 = rep + self.stat_view_scale * shift
            logits1, logits2 = model.head(rep), model.head(rep2)
            consist = F.kl_div(F.log_softmax(logits2.float(), 1), F.softmax(logits1.detach().float(), 1), reduction="batchmean")
            return self.loss(logits2, y) + self.cal_anchors_loss(rep2, model, y)[0] + consist
        if self.second_view == "local":
            # control: this client's own per-class variance from its previous round (no other clients' information)
            own = getattr(self, "proto_stats", {}) or {}
            if not own:
                return rep.new_zeros(())
            zero = torch.zeros(rep.shape[1])
            table = [((own[c][2] / own[c][0]) - (own[c][1] / own[c][0]).square()).clamp_min(0) if c in own and own[c][0] > 1
                     else zero for c in range(self.num_classes)]
            var = torch.stack(table).to(rep.device, rep.dtype)[y]
        else:
            if len(self.global_var) < self.num_classes:
                return rep.new_zeros(())
            var = torch.stack([self.global_var[c] for c in range(self.num_classes)]).to(rep.device, rep.dtype)[y]
        if self.second_view == "noise_fixed":  # control for proto_shift + fixed: random direction, same constant norm
            d = torch.randn_like(rep)
            rep2 = rep + d / d.norm(dim=1, keepdim=True).clamp_min(1e-8) * (self.shift_fixed * float(self.global_gap))
            logits1, logits2 = model.head(rep), model.head(rep2)
            consist = F.kl_div(F.log_softmax(logits2.float(), 1), F.softmax(logits1.detach().float(), 1), reduction="batchmean")
            return self.loss(logits2, y) + self.cal_anchors_loss(rep2, model, y)[0] + consist
        if self.second_view in ("noise", "noise_kl"):
            var = var.mean(dim=1, keepdim=True).expand_as(var)  # same total energy, no class/dimension structure
        rep2 = rep + torch.randn_like(rep) * (self.stat_view_scale * var.clamp_min(0)).sqrt()
        logits1, logits2 = model.head(rep), model.head(rep2)
        if self.second_view == "stat_ce":  # ablation: shared-variance view with CE only (no anchor binding, no KL)
            return self.loss(logits2, y)
        if self.second_view == "noise_kl":  # anchor-free: noise view, CE + prediction consistency only
            consist = F.kl_div(F.log_softmax(logits2.float(), 1), F.softmax(logits1.detach().float(), 1), reduction="batchmean")
            return self.loss(logits2, y) + consist
        consist = F.kl_div(F.log_softmax(logits2.float(), 1), F.softmax(logits1.detach().float(), 1), reduction="batchmean")
        return self.loss(logits2, y) + self.cal_anchors_loss(rep2, model, y)[0] + consist

    @torch.no_grad()
    def _prototype_soft_targets(self):
        """C x C target table: row y = (1-beta) one-hot(y) + beta softmax(-d(P_y, P_j) / (mean d * T)) over the
        global prototypes (collaborative class similarity). None until every class has a global prototype."""
        protos = getattr(self, "global_protos", None)
        if not protos or len(protos) < self.num_classes:
            return None
        P = torch.stack([torch.as_tensor(protos[c], dtype=torch.float32).view(-1) for c in range(self.num_classes)]).to(self.device)
        d = torch.cdist(P, P)
        sim = torch.softmax(-d / (d.mean() * self.soft_temp), dim=1)
        eye = torch.eye(self.num_classes, device=self.device)
        return (1 - self.soft_label) * eye + self.soft_label * sim

    def _anchor_mixup_loss(self, rep, y, model):
        """Mix each feature with the shared anchor of a random other class; labels mixed with the same weight.
        Anchors are identical on every client, so the head learns globally consistent boundaries."""
        anchors = self.semantic_anchors[: self.num_classes].to(device=rep.device, dtype=rep.dtype)
        other = (y + torch.randint(1, self.num_classes, y.shape, device=y.device)) % self.num_classes
        lam = torch.distributions.Beta(self.anchor_mixup, self.anchor_mixup).sample(y.shape).to(rep.device)
        lam = torch.maximum(lam, 1 - lam).view(-1, 1)  # keep the sample's own class dominant
        logits = model.head(lam * rep + (1 - lam) * anchors.index_select(0, other))
        nll = F.log_softmax(logits.float(), dim=1)
        lam = lam.view(-1).float()
        return -(lam * nll.gather(1, y.view(-1, 1)).view(-1) + (1 - lam) * nll.gather(1, other.view(-1, 1)).view(-1)).mean()

    def set_semantic_anchors(self, semantic_anchors):
        if semantic_anchors is None:
            self.semantic_anchors = None
            return
        self.semantic_anchors = torch.as_tensor(
            semantic_anchors,
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()
        if getattr(self, "private_anchors", False):
            # control: each client sees its own fixed random rotation of the anchors (same geometry, nothing shared)
            if getattr(self, "_anchor_rot", None) is None:
                g = torch.Generator().manual_seed(10007 + int(self.id))
                d = self.semantic_anchors.shape[1]
                self._anchor_rot = torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(self.device)
            self.semantic_anchors = self.semantic_anchors @ self._anchor_rot

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
        # warmup: the first K local rounds use the fixed (global-anchor) margin, then the chosen rule
        reduce = max if self.train_time_cost["num_rounds"] < self.gap_warmup else self.gap_reduce
        gap = self.gap_scale * max(reduce(float(self.local_gap), float(self.global_gap)), 0.0)
        if gap > 0:
            one_hot = F.one_hot(y.long(), self.num_classes).to(device=feats.device, dtype=dist.dtype)
            dist = dist + one_hot * gap
        return self.loss(-dist, y.long())

    @torch.no_grad()
    def memorization_probe(self, max_n=1000):
        """Does private-vs-twin disagreement measure memorization? Members = my train samples, non-members = my test
        samples. Scores: conf = p_private(y|x) (standard loss-based membership signal); gap_head = p_private - p_twin
        (twin's own head); gap_anchor = p_private - p_twin_anchor (twin representation vs fixed anchors, no head).
        Returns membership AUCs (0.5 = no signal); a memorization score should separate members better than conf."""
        cm, model = self.common_model, getattr(self, "model", None)
        if cm is None or model is None or self.semantic_anchors is None:
            return {}
        anchors = self.semantic_anchors[: self.num_classes].to(self.device).float()
        model.to(self.device).eval(); cm.to(self.device).eval()

        def scores(loader, limit=max_n):
            out = {"conf": [], "gap_head": [], "gap_anchor": []}
            n = 0
            for x, y in loader:
                x, y = self._move_input(x, y)
                pp = F.softmax(model.head(model.base(x)).float(), 1).gather(1, y.view(-1, 1)).view(-1)
                rt = cm.base(x).float()
                pt = F.softmax(cm.head(rt).float(), 1).gather(1, y.view(-1, 1)).view(-1)
                d = torch.cdist(rt, anchors).pow(2)
                pa = F.softmax(-d / d.mean(1, keepdim=True).clamp(min=1e-6) * 10.0, 1).gather(1, y.view(-1, 1)).view(-1)
                out["conf"].append(pp); out["gap_head"].append(pp - pt); out["gap_anchor"].append(pp - pa)
                n += y.numel()
                if n >= limit:
                    break
            return {k: torch.cat(v) for k, v in out.items()}

        dump = self.current_round in getattr(self, "mem_dump_rounds", ())
        if dump:  # full train set in shard order (no shuffle) so samples can be matched to external memorization scores
            from torch.utils.data import DataLoader
            mem = scores(DataLoader(self._read_client_data_cached(is_train=True), self.batch_size, shuffle=False),
                         limit=float("inf"))
        else:
            mem = scores(self.load_train_data(drop_last=False))
        non = scores(self.load_test_data())
        if dump:
            import os
            d = os.path.join("logs", "memdump", str(getattr(self, "run_goal", "run")))
            os.makedirs(d, exist_ok=True)
            torch.save({k: v.cpu() for k, v in mem.items()}, os.path.join(d, f"r{self.current_round}_c{self.id}.pt"))
        model.train(); cm.train()

        def auc(a, b):  # P(score_member > score_nonmember), Mann-Whitney
            s = torch.cat([a, b]); r = s.argsort().argsort().float() + 1
            return float((r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
        return {k: auc(mem[k], non[k]) for k in mem}

    @torch.no_grad()
    def _cm_split(self):
        """(train part, held-out part) of my training data for the memorization gate; fixed per client."""
        if self._split is None:
            data = self._read_client_data_cached(is_train=True)
            perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(self.id)).tolist()
            n = max(1, int(len(data) * self.cm_val))
            self._split = (Subset(data, perm[n:]), Subset(data, perm[:n]))
        return self._split

    def load_train_data(self, batch_size=None, drop_last=True):
        if getattr(self, "cm_val", 0.0) <= 0:
            return super().load_train_data(batch_size, drop_last)
        return DataLoader(self._cm_split()[0], batch_size or self.batch_size, drop_last=drop_last, shuffle=True,
                          pin_memory=str(self.device).startswith("cuda"))

    @torch.no_grad()
    def twin_gate_stats(self):
        """Received twin (shared term alone) on my held-out split and on as many of my training samples:
        (held-out correct, train correct, n). The difference is the shared term's memorization gap."""
        train, val = self._cm_split()
        cm = self.common_model.to(self.device).eval()

        def correct(ds):
            c = 0
            for x, y in DataLoader(ds, 256):
                x, y = self._move_input(x, y)
                c += int(cm(x).argmax(1).eq(y).sum())
            return c

        n = len(val)
        out = correct(val), correct(Subset(train, list(range(min(n, len(train)))))), n
        cm.train()
        return out

    def common_model_test(self):
        """Personalized test accuracy of the common model (twin) with its own head: (correct, samples)."""
        cm = self.common_model
        if cm is None:
            return 0, 0
        cm.to(self.device).eval()
        model = getattr(self, "model", None)
        if model is not None:
            model.to(self.device).eval()
        anchors = None if self.semantic_anchors is None else self.semantic_anchors[: self.num_classes].to(self.device).float()
        correct = n = 0
        fused = defaultdict(int)  # extra accuracies of combining private + twin at test time
        for x, y in self.load_test_data():
            x, y = self._move_input(x, y)
            rt = cm.base(x).float()
            lt = cm.head(rt).float()
            correct += int(lt.argmax(1).eq(y).sum())
            if model is not None:
                rp = model.base(x).float()
                lp = model.head(rp).float()
                fused["prob_avg"] += int((F.softmax(lp, 1) + F.softmax(lt, 1)).argmax(1).eq(y).sum())
                fused["logit_sum"] += int((lp + lt).argmax(1).eq(y).sum())  # the residual-expert prediction
                if anchors is not None:  # same fixed anchor space for both architectures: average, nearest anchor
                    fused["rep_avg_anchor"] += int(torch.cdist((rp + rt) / 2, anchors).argmin(1).eq(y).sum())
                    fused["twin_anchor"] += int(torch.cdist(rt, anchors).argmin(1).eq(y).sum())
            n += y.numel()
        cm.train()
        if model is not None:
            model.train()
        self.fused_correct = dict(fused)
        return correct, n

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
                    if self.eval_head:
                        pred = model.head(rep).argmax(dim=1)
                    else:
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
