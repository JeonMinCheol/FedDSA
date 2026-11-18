import torch
import torch.nn.functional as F


def augment_batch(x):
    """Standard image augmentation on a GPU batch: random crop (reflect-pad H/8) + horizontal flip."""
    n, c, h, w = x.shape
    pad = max(1, h // 8)
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i.view(n, 1, 1, 1) + torch.arange(h, device=x.device).view(1, 1, h, 1)).expand(n, c, h, padded.shape[3])
    out = torch.gather(padded, 2, rows)
    cols = (j.view(n, 1, 1, 1) + torch.arange(w, device=x.device).view(1, 1, 1, w)).expand(n, c, h, w)
    out = torch.gather(out, 3, cols)
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), out.flip(3), out)


def mix_batch(x, y, mode, alpha=1.0):
    """Mixup / CutMix on a GPU batch. Returns (mixed x, partner labels, lam); the loss is
    lam * L(y) + (1 - lam) * L(partner labels)."""
    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], y[perm], lam
    h, w = x.shape[2:]
    rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
    cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, h), max(cx - rw // 2, 0), min(cx + rw // 2, w)
    out = x.clone()
    out[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)  # exact pasted area
    return out, y[perm], lam


if __name__ == "__main__":
    x, y = torch.randn(8, 3, 32, 32), torch.arange(8)
    for m in ("mixup", "cutmix"):
        xm, yb, lam = mix_batch(x, y, m)
        assert xm.shape == x.shape and 0 <= lam <= 1 and yb.shape == y.shape
    print("ok")
