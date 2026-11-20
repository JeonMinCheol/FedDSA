"""FedScout client.

Every client holds (i) its own heterogeneous model, the *residual learner*, and (ii) a small *scout* model of one shared
architecture whose backbone is FedAvg'd. The residual learner is trained on CE(frozen received scout logits + its own
logits), so it only has to correct what the shared scout gets wrong on local data. Prediction = scout + alpha * learner,
with one alpha chosen on the clients' held-out splits. The scout also
  - screens training samples: the learner trains only where the received scout's p(y) is below --scout_screen_tau
    (its residual gradient softmax(scout + learner) - onehot(y) is ~0 elsewhere), and the same confidence gates
    inference (the learner runs only where the scout's top-1 probability is low);
  - is frozen by the server's memorization gate (held-out plateau), after which sharing stops, its personal head is
    refit once and fixed (--scout_freeze_head) and its backbone features are cached (--scout_cache).
"""
import copy
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from flcore.clients.clientbase import Client, load_item, save_item
from utils.augment import augment_batch

ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)  # ascending: ties -> the smaller weight on the learner


class clientScout(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(self.random_seed)
        self.device = torch.device(args.device)
        # --keep_client_models_on_gpu false: park my learner and scout in host memory between my turns (100 clients'
        # HtFE-8 learners do not fit in one GPU); they move to the GPU only while I train or am evaluated
        self.offload = not bool(getattr(args, "keep_client_models_on_gpu", True))
        self.model = load_item(self.role, "model", self.save_folder_name).to("cpu" if self.offload else self.device)
        self.scout = None  # set by the server (same initial weights everywhere)
        self.lr_scale = float(args.scout_lr_scale)
        # scout projector/head lr = scout lr x this (8 matches the pool projector: 64 features repeated 8 times into a
        # 512 x C head act like one 64 x C head trained with an 8x larger step)
        self.head_lr_scale = float(args.scout_head_lr_scale)
        # while the scout is federated its (averaged) head is never confident on my data, so screening keeps ~95% of
        # the samples and the learner mostly learns my class prior; with this, the learner waits until the freeze
        self.learner_after_freeze = bool(args.scout_learner_after_freeze)
        self.wd = float(args.scout_wd)
        self.val = float(args.scout_val)
        self.residual = args.scout_residual  # p2: learner only through the sum; i1: + standalone CE + KL to the scout
        self.screen_tau = float(args.scout_screen_tau)
        self.eval_ft = bool(args.scout_eval_ft) and args.scout_share == "all"
        # while the scout's head is the averaged one (--scout_share all, before the freeze) it does not know my label
        # distribution - that is what the residual learner adds - so screen on scout + log(my class frequencies)
        self.screen_prior = bool(args.scout_screen_prior) and args.scout_share == "all"
        self._log_prior = None
        self.cache = bool(args.scout_cache)
        self.frozen = False  # backbone frozen by the server's gate
        self.head_frozen = False  # personal projector/head fixed too (--scout_freeze_head)
        self._split = None
        self._feat_cache = None
        self.latest_train_metrics = {}

    # ---- data -------------------------------------------------------------------------------------------------------
    def split(self):
        """(train part, held-out part) of my training data, fixed per client (gate + alpha selection)."""
        if self._split is None:
            data = self._read_client_data_cached(is_train=True)
            perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(self.id)).tolist()
            n = max(1, int(len(data) * self.val))
            self._split = (Subset(data, perm[n:]), Subset(data, perm[:n]))
        return self._split

    def load_train_data(self, batch_size=None, drop_last=True):
        return DataLoader(self.split()[0], batch_size or self.batch_size, drop_last=drop_last, shuffle=True,
                          pin_memory=str(self.device).startswith("cuda"))

    def _park(self):
        if self.offload:
            self.scout.to("cpu")
            self.model.to("cpu")

    def _to(self, x, y):
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long().view(-1)

    # ---- training ---------------------------------------------------------------------------------------------------
    def train(self):
        start = time.time()
        sc, model = self.scout.to(self.device), self.model.to(self.device)
        trainloader = self.load_train_data()
        fixed = self.frozen and self.head_frozen  # the scout does not change any more
        for n, p in sc.named_parameters():
            p.requires_grad_(not (self.frozen if n.startswith("base.0.") else fixed))
        sc.train()
        if self.frozen:
            sc.base[0].eval()  # frozen backbone keeps its BN statistics
        lr_s = self.learning_rate * self.lr_scale
        groups = [{"params": [p for n, p in sc.named_parameters() if p.requires_grad and n.startswith("base.0.")], "lr": lr_s},
                  {"params": [p for n, p in sc.named_parameters() if p.requires_grad and not n.startswith("base.0.")],
                   "lr": lr_s * self.head_lr_scale}]
        groups = [g for g in groups if g["params"]]
        opt_s = torch.optim.SGD(groups, lr=lr_s, weight_decay=self.wd) if groups else None
        opt = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        teacher = copy.deepcopy(sc).eval().requires_grad_(False)  # the received scout, before it fits my data
        if self.cache and self.frozen and not self.augment:  # cached features are only valid for unaugmented inputs
            trainloader = self._cached_loader(sc)
        model.train()
        sums = {"task_loss": 0.0, "train_acc": 0.0, "priv_frac": 0.0, "nan_skip": 0.0}
        nb = 0
        skip_l = self.learner_after_freeze and not self.frozen
        for _ in range(self.local_epochs):
            for x, y, *ix in trainloader:
                x, y = self._to(x, y)
                if self.augment:  # standard crop + flip, only for the augmentation-scope experiment (off by default)
                    x = augment_batch(x)
                f = self._feat_cache[ix[0].to(self._feat_cache.device)] if ix else None
                if skip_l:  # --scout_learner_after_freeze: only the scout trains (no teacher pass, no learner pass)
                    if fixed:
                        continue
                    out_s = sc.head(sc.base(x) if f is None else sc.base[1:](f))
                    loss_s = self.loss(out_s, y)
                    if not bool(torch.isfinite(loss_s)):
                        continue
                    opt_s.zero_grad(set_to_none=True)
                    loss_s.backward()
                    opt_s.step()
                    with torch.no_grad():
                        sums["task_loss"] += float(loss_s)
                        sums["train_acc"] += float(out_s.argmax(1).eq(y).float().mean())
                    nb += 1  # priv_frac stays 0: the learner did not run
                    continue
                with torch.no_grad():
                    t_logits = teacher.head(teacher.base(x) if f is None else teacher.base[1:](f)).float()
                keep = None
                if self.screen_tau > 0:
                    # scout screening: skip samples the received scout already gets right with confidence
                    g = t_logits + self.log_prior() if (self.screen_prior and not self.frozen) else t_logits
                    p_y = g.softmax(1).gather(1, y[:, None]).squeeze(1)
                    keep = p_y < self.screen_tau
                    if int(keep.sum()) < 2:  # BatchNorm in train mode needs >= 2 samples
                        keep[p_y.topk(2, largest=False).indices] = True
                xp, yp, tp = (x, y, t_logits) if keep is None else (x[keep], y[keep], t_logits[keep])
                out_priv = model(xp)
                out = out_priv + tp  # residual: the learner fixes what the frozen received scout gets wrong
                loss_l = self.loss(out, yp)
                if self.residual == "i1":  # learner also usable alone: its own CE + KD from the scout
                    loss_l = loss_l + self.loss(out_priv, yp) + F.kl_div(
                        F.log_softmax(out_priv.float(), 1), F.softmax(tp, 1), reduction="batchmean")
                # the scout trains on its own CE
                loss_s = None if fixed else self.loss(sc.head(sc.base(x) if f is None else sc.base[1:](f)), y)
                opt.zero_grad(set_to_none=True)
                if opt_s is not None:
                    opt_s.zero_grad(set_to_none=True)
                # a learner that diverged must not stop the scout (and vice versa); with both finite this is exactly
                # one backward of the sum. Divergence happens when screening leaves the learner a few samples per
                # batch (BatchNorm), i.e. when the scout memorized my data (no gate / no averaging ablations).
                ok_l, ok_s = bool(torch.isfinite(loss_l)), loss_s is not None and bool(torch.isfinite(loss_s))
                if not ok_l:
                    sums["nan_skip"] += 1.0
                if not (ok_l or ok_s):
                    continue
                ((loss_l if ok_l else 0) + (loss_s if ok_s else 0)).backward()
                if ok_l:
                    opt.step()
                if ok_s:
                    opt_s.step()
                loss = (loss_l if ok_l else 0) + (loss_s if ok_s else 0)
                with torch.no_grad():
                    sums["task_loss"] += float(loss)
                    sums["train_acc"] += float(out.argmax(1).eq(yp).float().mean())
                    sums["priv_frac"] += 1.0 if keep is None else float(keep.float().mean())
                nb += 1
        self.latest_train_metrics = {k: v / max(nb, 1) for k, v in sums.items()}
        save_item(model, self.role, "model", self.save_folder_name)
        self._park()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start

    def log_prior(self):
        if self._log_prior is None:
            ys = torch.as_tensor([int(y) for _, y in self.split()[0]])
            self._log_prior = torch.bincount(ys, minlength=self.num_classes).float().clamp_min(1).log().to(self.device)
        return self._log_prior

    def _cached_loader(self, sc):
        """Frozen backbone (eval BN) on unaugmented data: its features never change, so compute them once and skip the
        scout backbone (teacher and live copy) for the rest of training."""
        sub = self.split()[0]
        if self._feat_cache is None:
            with torch.no_grad():
                self._feat_cache = torch.cat([sc.base[0](self._to(xb, yb)[0]) for xb, yb in DataLoader(sub, 256)])
                ib = torch.arange(0, len(sub), max(1, len(sub) // 64))[:64]  # self-check: rows line up with samples
                live = sc.base[0](torch.stack([sub[i][0] for i in ib.tolist()]).to(self.device))
            # batch 256 vs 64 picks different cuDNN algorithms; with TF32 (Ampere, ariel) they differ by up to ~1.5e-3
            # (features up to ~2.5), so atol=1e-4 failed on every client. A misaligned row differs by O(1).
            assert torch.allclose(live, self._feat_cache[ib.to(live.device)], atol=1e-2), "feature cache misaligned"
            self._idx_ds = [(xb, yb, i) for i, (xb, yb) in enumerate(sub)]
        return DataLoader(self._idx_ds, self.batch_size, drop_last=True, shuffle=True,
                          pin_memory=str(self.device).startswith("cuda"))

    def realign(self, epochs, sc=None):
        """After the averaged backbone arrives (or is restored), fit my personal projector/head to it with the backbone
        and its BN fixed; otherwise the scout I evaluate and learn from is a new backbone under last round's head.
        sc: a copy to fit instead of my scout (evaluation-time personalization, --scout_eval_ft)."""
        own = sc is None
        sc = (self.scout if own else sc).to(self.device)
        sc.train()
        sc.base[0].eval()
        params = [p for n, p in sc.named_parameters() if not n.startswith("base.0.")]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.SGD(params, lr=self.learning_rate * self.lr_scale * self.head_lr_scale, weight_decay=self.wd)
        for _ in range(epochs):
            for x, y in self.load_train_data():
                x, y = self._to(x, y)
                with torch.no_grad():
                    f = sc.base[0](x)
                loss = self.loss(sc.head(sc.base[1:](f)).float(), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        if own:
            self._park()

    # ---- evaluation -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def gate_stats(self):
        """Received scout on my held-out split and on as many of my training samples: (held-out correct, train
        correct, n). The difference is the shared model's memorization gap."""
        train, val = self.split()
        sc = self.scout.to(self.device).eval()

        def correct(ds):
            c = 0
            for x, y in DataLoader(ds, 256):
                x, y = self._to(x, y)
                c += int(sc(x).argmax(1).eq(y).sum())
            return c

        n = len(val)
        out = correct(val), correct(Subset(train, list(range(min(n, len(train)))))), n
        sc.train()
        self._park()
        return out

    @torch.no_grad()
    def logits(self):
        """Scout and learner logits on my test set and held-out split (the server picks alpha and scores everything)."""
        sc = self.scout
        if self.eval_ft and not self.frozen:
            # head still averaged (--scout_share all): score the model I would deploy if training stopped now, i.e.
            # this scout with its projector/head fitted to my data for one epoch (a copy; training is unaffected)
            sc = copy.deepcopy(self.scout)
            with torch.enable_grad():
                self.realign(1, sc)
        sc, model = sc.to(self.device).eval(), self.model.to(self.device).eval()
        out = {}
        for tag, loader in (("test", self.load_test_data()), ("held", DataLoader(self.split()[1], 256)),
                            ("train", DataLoader(self.split()[0], 256))):  # train: fits the class-wise bias diagnostic
            lt, lp, ys = [], [], []
            for x, y in loader:
                x, y = self._to(x, y)
                lt.append(sc(x).float().cpu())
                lp.append(torch.nan_to_num(model(x).float(), nan=0.0, posinf=0.0, neginf=0.0).cpu())  # diverged learner -> no vote
                ys.append(y.cpu())
            out[tag] = (torch.cat(lt), torch.cat(lp), torch.cat(ys))
        out["log_prior"] = self.log_prior().cpu()  # diagnostic: is the learner just a local class-prior fix?
        sc.train()
        model.train()
        self._park()
        return out

    def test_metrics(self):  # the server scores the fused prediction (see FedScout.test_metrics)
        raise NotImplementedError

    def train_metrics(self):
        return 0.0, 1
