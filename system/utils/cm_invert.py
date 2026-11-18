"""Client-side model inversion of the FedAvg'd small common model: synthesize inputs of this client's classes that
the aggregated model (and its aggregated BatchNorm statistics, i.e. all clients' feature statistics) recognizes,
then let the private heterogeneous backbone learn from them. The synthetic inputs carry other clients' knowledge
into the private backbone's INPUT side, which logit/feature distillation on local images cannot do."""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


def synthesize(teacher, labels, shape, steps=30, lr=0.1, bn_w=1.0, tv_w=1e-4, chunk=128, lo=None, hi=None,
               anchors=None, jitter=0, first_bn=1.0, init=None):
    """DeepInversion-style synthesis. teacher: eval-mode module with .base/.head. Returns detached inputs,
    kept inside the real data range [lo, hi] (per-channel tensors) so they cannot blow up the private model.
    anchors [C, D]: if given, the target is "teacher.base(x) reaches the fixed anchor of y" (architecture-free
    target in the shared anchor space) instead of cross-entropy on the teacher's own personal head.
    jitter > 0: random shift (up to `jitter` px) + horizontal flip of x at every step (DeepInversion); first_bn
    multiplies the first BN layer's statistic loss (low-level statistics matter most for image realism)."""
    teacher.eval()
    bns = [m for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)]
    out = []
    for i in range(0, len(labels), chunk):
        y = labels[i:i + chunk]
        x = (torch.randn((len(y),) + tuple(shape), device=y.device) if init is None
             else init[i:i + chunk].detach().clone().to(y.device)).requires_grad_(True)  # init: refine given images
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            bn_losses = []
            hooks = [m.register_forward_hook(
                lambda m, inp, o, s=(first_bn if j == 0 else 1.0): bn_losses.append(s * (
                    (inp[0].mean((0, 2, 3)) - m.running_mean).norm()
                    + (inp[0].var((0, 2, 3), unbiased=False) - m.running_var).norm()))) for j, m in enumerate(bns)]
            xi = x
            if jitter > 0:
                dx, dy = (int(v) for v in torch.randint(-jitter, jitter + 1, (2,)))
                xi = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
                if torch.rand(()) < 0.5:
                    xi = xi.flip(3)
            try:
                rep = teacher.base(xi)
                logits = teacher.head(rep)
            finally:
                for h in hooks:
                    h.remove()
            tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
            if anchors is None:
                target = F.cross_entropy(logits, y)
            else:  # squared distance to the class anchor, normalized so that rep = 0 gives 1
                a = anchors.index_select(0, y)
                target = (rep.float() - a).pow(2).sum(1).mean() / a.pow(2).sum(1).mean()
            loss = target + bn_w * sum(bn_losses) + tv_w * tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if lo is not None:
                with torch.no_grad():
                    x.data = torch.max(torch.min(x.data, hi), lo)
        out.append(x.detach())
    return torch.cat(out)


@contextlib.contextmanager
def frozen_bn(model):
    """Forward synthetic data with batch statistics (stable early in training, unlike running stats) but restore
    the BN buffers afterwards, so synthetic data never shifts the private model's test-time statistics."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bns]
    try:
        yield
    finally:
        for m, (rm, rv, nb) in zip(bns, saved):  # rebind, not copy_: the updated tensors are saved for backward
            m.running_mean, m.running_var, m.num_batches_tracked = rm, rv, nb


if __name__ == "__main__":  # self-check on a tiny BN model: synthesis lowers CE and keeps shapes
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(8, 5)
    t = T().eval()
    y = torch.randint(5, (16,))
    x0 = torch.randn(16, 3, 32, 32)
    ce0 = F.cross_entropy(t.head(t.base(x0)), y).item()
    xs = synthesize(t, y, (3, 32, 32), steps=50)
    ce1 = F.cross_entropy(t.head(t.base(xs)), y).item()
    assert xs.shape == (16, 3, 32, 32) and ce1 < ce0, (ce0, ce1)
    print(f"ok ce {ce0:.3f} -> {ce1:.3f}")
