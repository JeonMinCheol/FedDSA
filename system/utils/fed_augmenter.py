"""Federated adversarial augmenter: a tiny image-to-image network, identical in every client (so it can be FedAvg'd
across heterogeneous backbones), placed IN FRONT of the backbone. It perturbs real local images within an eps
budget to raise the backbone's loss (through a gradient-reversal layer), and the backbone learns to resist.
Averaged over clients, it learns perturbations that many different backbones are weak to."""
import torch
import torch.nn as nn


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class Augmenter(nn.Module):
    def __init__(self, ch=3, width=16, eps=0.25, mode="pixel", lf=4, color=0.2):
        super().__init__()
        self.eps, self.mode, self.lf, self.color = eps, mode, lf, color
        self.net = nn.Sequential(nn.Conv2d(ch + 1, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, ch, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)  # starts as the identity map (delta = 0)
        nn.init.zeros_(self.net[-1].bias)
        if mode == "semantic":  # per-image, per-channel contrast/brightness from pooled features
            self.color_head = nn.Linear(ch + 1, 2 * ch)
            nn.init.zeros_(self.color_head.weight)
            nn.init.zeros_(self.color_head.bias)

    def forward(self, x, lo=None, hi=None):
        """pixel: x + eps * tanh(net([x, z])). semantic: low-frequency delta (net output average-pooled by `lf`
        and upsampled back) + bounded per-channel contrast/brightness, i.e. lighting/colour/blob changes instead of
        high-frequency adversarial noise. z ~ N(0,1) makes views stochastic. The change passes through a
        gradient-reversal layer: one backward pass makes the backbone descend and the augmenter ascend the loss."""
        z = torch.randn_like(x[:, :1])
        inp = torch.cat([x, z], 1)
        d = self.net(inp)
        if self.mode == "semantic":
            d = nn.functional.interpolate(nn.functional.avg_pool2d(d, self.lf), size=x.shape[2:], mode="bilinear",
                                          align_corners=False)
            sb = self.color * torch.tanh(self.color_head(inp.mean((2, 3))))
            s, b = sb[:, : x.shape[1], None, None], sb[:, x.shape[1]:, None, None]
            delta = x * s + b + self.eps * torch.tanh(d)
        else:
            delta = self.eps * torch.tanh(d)
        out = x + _Reverse.apply(delta)
        if lo is not None:
            out = torch.max(torch.min(out, hi), lo)
        return out


if __name__ == "__main__":  # self-check: identity at init, augmenter gets ascending gradients
    a = Augmenter()
    x = torch.randn(4, 3, 32, 32)
    assert torch.allclose(a(x), x)
    lin = nn.Linear(3 * 32 * 32, 2)
    y = torch.tensor([0, 1, 0, 1])
    a.net[-1].bias.data.fill_(0.1)
    loss = nn.functional.cross_entropy(lin(a(x).flatten(1)), y)
    loss.backward()
    w = lin.weight.grad.clone()
    # reversed: stepping the augmenter along -grad must INCREASE the loss
    with torch.no_grad():
        for p in a.parameters():
            if p.grad is not None:
                p -= 0.5 * p.grad
    torch.manual_seed(0)
    print("ok", float(loss), "grad on backbone:", float(w.abs().sum()) > 0)
