"""ETF-ProtoFL client (Yang et al., Expert Systems With Applications 333, 2027, 133968).

Reimplemented from the paper text (no public code):
  * backbone f_i -> trainable projector g_i to d = C dims (BaseHeadSplit with --projector linear)
  * fixed simplex-ETF classifier on L2-normalised features (Eq. 6-8)
  * loss = CE(ETF logits) + lambda * AMCL(P, global prototypes)            (Eq. 9-13)
    AMCL: sim(a, b) = (1 + cos(a, b)) / 2, s_cc = (sim - m) / tau, s_cj = sim / tau
  * unnormalised class-mean prototypes P are uploaded (Sec 4.1)
  * inference: cosine similarity to the global prototypes (Sec 4.3)
Assumption (not given in the paper): ETF logits are scaled by --etf_logit_scale
(default 1/tau); unscaled cosine logits in [-1, 1] barely train CE over 100 classes.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client, load_item, save_item


def simplex_etf(d, num_classes):
    """d x C matrix whose unit-norm columns have pairwise cosine -1/(C-1)."""
    if d < num_classes - 1:
        raise ValueError("ETF needs feature dim >= num_classes - 1")
    etf = torch.eye(num_classes) - 1.0 / num_classes
    etf = F.normalize(etf, dim=0)
    if d > num_classes:
        etf = torch.cat([etf, torch.zeros(d - num_classes, num_classes)], dim=0)
    return etf[:d]


class ETFHead(nn.Module):
    def __init__(self, d, num_classes, scale):
        super().__init__()
        self.register_buffer("etf", simplex_etf(d, num_classes))
        self.scale = float(scale)

    def forward(self, p):
        return self.scale * F.normalize(p, dim=1) @ self.etf


class clientETFProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.device = torch.device(args.device)
        self.lamda = float(args.lamda)
        self.margin = float(args.etf_margin)
        self.adaptive_margin = getattr(args, "etf_adaptive_margin", "off")
        self.current_margin = self.margin
        self.tau = float(args.etf_tau)
        scale = args.etf_logit_scale if args.etf_logit_scale > 0 else 1.0 / self.tau
        model = load_item(self.role, "model", self.save_folder_name)
        model.head = ETFHead(int(args.feature_dim), self.num_classes, scale)  # frozen: buffer only
        self.model = model.to(self.device)
        save_item(self.model, self.role, "model", self.save_folder_name)
        self.global_protos = None
        self.protos, self.proto_counts = {}, {}

    def _move(self, x, y):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long().view(-1)

    def _global_matrix(self):
        if not self.global_protos:
            return None, None
        labels = sorted(self.global_protos)
        mat = torch.stack([self.global_protos[c].to(self.device).float() for c in labels])
        return torch.tensor(labels, device=self.device), mat

    def local_angular_gap(self, centered=False):
        """Mean nearest-neighbour distance between own class prototypes, in AMCL's sim units ((1-cos)/2).
        centered: subtract the mean prototype first (ETF/neural-collapse geometry is defined on centred means)."""
        if len(self.protos) < 2:
            return None
        mat = torch.stack([self.protos[c].float() for c in sorted(self.protos)])
        if centered:
            mat = mat - mat.mean(dim=0, keepdim=True)
        mat = F.normalize(mat, dim=1)
        cos = (mat @ mat.t()).fill_diagonal_(-1.0)
        return float(((1.0 - cos.max(dim=1).values) / 2.0).mean())

    def amcl_loss(self, p, y, labels, protos):
        """Angular-margin prototype contrastive loss over the classes that have a global prototype."""
        lookup = torch.full((self.num_classes,), -1, device=self.device, dtype=torch.long)
        lookup[labels] = torch.arange(labels.numel(), device=self.device)
        target = lookup[y]
        keep = target >= 0
        if not bool(keep.any()):
            return p.new_zeros(())
        sim = 0.5 * (1.0 + F.normalize(p[keep].float(), dim=1) @ F.normalize(protos, dim=1).t())
        logits = sim / self.tau
        logits = logits - (self.current_margin / self.tau) * F.one_hot(target[keep], labels.numel()).float()
        return F.cross_entropy(logits, target[keep])

    def train(self):
        start = time.time()
        model = self.model
        model.train()
        optimizer = torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr=self.learning_rate)
        labels, protos = self._global_matrix()
        gap = (self.local_angular_gap(centered=self.adaptive_margin == "min_centered")
               if self.adaptive_margin in ("min", "min_centered") else None)
        # adaptive: margin follows what this client actually separates (last round's own prototypes)
        self.current_margin = self.margin if gap is None else min(self.margin, gap)
        proto_sum = torch.zeros(self.num_classes, model.head.etf.shape[0], device=self.device)
        proto_cnt = torch.zeros(self.num_classes, device=self.device)
        for _ in range(self.local_epochs):
            for x, y in self.load_train_data():
                x, y = self._move(x, y)
                p = model.base(x)
                loss = self.loss(model.head(p), y)
                if protos is not None and self.lamda > 0:
                    loss = loss + self.lamda * self.amcl_loss(p, y, labels, protos)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    proto_sum.index_add_(0, y, p.detach().float())
                    proto_cnt.index_add_(0, y, torch.ones_like(y, dtype=proto_cnt.dtype))
        present = torch.nonzero(proto_cnt > 0).view(-1).tolist()
        self.protos = {c: (proto_sum[c] / proto_cnt[c]).detach() for c in present}
        self.proto_counts = {c: float(proto_cnt[c]) for c in present}
        # separability of this round's prototypes; used by the server's gap-weighted aggregation
        self.upload_gap = self.local_angular_gap(centered=True)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start

    @torch.no_grad()
    def test_metrics(self):
        labels, protos = self._global_matrix()
        if protos is None:
            return 0, 1e-5, 0
        self.model.eval()
        correct = total = head_correct = 0
        protos_n = F.normalize(protos, dim=1)
        for x, y in self.load_test_data():
            x, y = self._move(x, y)
            p = self.model.base(x).float()
            cos = F.normalize(p, dim=1) @ protos_n.t()
            correct += int((labels[cos.argmax(dim=1)] == y).sum())
            head_correct += int((self.model.head(p).argmax(dim=1) == y).sum())  # diagnostic only
            total += y.numel()
        self.last_head_correct = head_correct
        return correct, total, 0

    @torch.no_grad()
    def train_metrics(self):
        self.model.eval()
        losses = num = 0
        for x, y in self.load_train_data():
            x, y = self._move(x, y)
            losses += float(self.loss(self.model.head(self.model.base(x)), y)) * y.numel()
            num += y.numel()
        return losses, num
