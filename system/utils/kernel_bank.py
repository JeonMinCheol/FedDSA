"""Shared kernel bank: every conv weight of any backbone is assembled from B x B x k x k chunks, each a linear
combination of M shared bank kernels (one bank per kernel size). The bank has the same shape in every
architecture, so the server can FedAvg it across heterogeneous clients; the per-layer coefficients stay personal."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BankConv2d(nn.Module):
    def __init__(self, conv, bank_owner, key, chunk, m):
        super().__init__()
        self.out_ch, self.in_g = conv.weight.shape[:2]
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
        self.key, self.chunk = key, chunk
        self._owner = [bank_owner]  # plain list: not a submodule, so the bank is stored/averaged once per model
        n_o, n_i = math.ceil(self.out_ch / chunk), math.ceil(self.in_g / chunk)
        fan_in = self.in_g * conv.weight.shape[2] * conv.weight.shape[3]
        # bank kernels have unit L2 norm (entries ~ 1/sqrt(D), D = B*B*kh*kw), so weight std = sqrt(M/D) * coef std;
        # match Kaiming std sqrt(2/fan_in). Unit-norm (not unit-RMS) keeps the weight-space step ~ lr, as in a plain conv.
        d = chunk * chunk * conv.weight.shape[2] * conv.weight.shape[3]
        self.coef = nn.Parameter(torch.randn(n_o, n_i, m) * math.sqrt(2.0 / fan_in) * math.sqrt(d / m))
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone())

    def weight(self):
        bank = self._owner[0].banks[self.key]  # [M, B, B, k, k]
        # unit-norm kernels: the bank only carries directions, the personal coefs carry scale
        bank = bank * torch.rsqrt(bank.pow(2).flatten(1).sum(1) + 1e-8).view(-1, 1, 1, 1, 1)
        w = torch.einsum("oim,mabxy->oaibxy", self.coef, bank)
        n_o, B, n_i, _, kh, kw = w.shape
        w = w.reshape(n_o * B, n_i * B, kh, kw)
        return w[: self.out_ch, : self.in_g]

    def forward(self, x):
        return F.conv2d(x, self.weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class KernelBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.banks = nn.ParameterDict()


def convert(backbone, m=64, chunk=16, seed=0):
    """Replaces every nn.Conv2d in `backbone` by a BankConv2d. Returns the KernelBank (attach it to the model so it
    is trained and saved with it). Bank init uses a fixed seed so all clients start from the same bank."""
    owner = KernelBank()
    g = torch.Generator().manual_seed(seed)
    convs = [(n, c) for n, c in backbone.named_modules() if isinstance(c, nn.Conv2d)]
    for name, conv in convs:
        kh, kw = conv.weight.shape[2:]
        key = f"k{kh}x{kw}"
        if key not in owner.banks:
            owner.banks[key] = nn.Parameter(torch.randn(m, chunk, chunk, kh, kw, generator=g))
        parent = backbone.get_submodule(name.rsplit(".", 1)[0]) if "." in name else backbone
        setattr(parent, name.rsplit(".", 1)[-1], BankConv2d(conv, owner, key, chunk, m).to(conv.weight.device))
    # one bank kernel is reused by thousands of chunks: average (not sum) its gradient over those uses
    uses = {}
    for mod in backbone.modules():
        if isinstance(mod, BankConv2d):
            uses[mod.key] = uses.get(mod.key, 0) + mod.coef.shape[0] * mod.coef.shape[1]
    for key, p in owner.banks.items():
        p.register_hook(lambda g, n=uses[key]: g / n)
    return owner


if __name__ == "__main__":  # self-check: shapes, gradients reach bank and coefs, init scale
    import torchvision
    net = torchvision.models.mobilenet_v2(num_classes=10)
    ref = net.features[1].conv[0][0].weight.std().item()
    bank = convert(net, m=32)
    out = net(torch.randn(2, 3, 32, 32))
    out.sum().backward()
    assert all(p.grad is not None for p in bank.banks.values())
    assert all(m.coef.grad is not None for m in net.modules() if isinstance(m, BankConv2d))
    w = net.features[1].conv[0][0].weight()
    print("ok", out.shape, sorted(bank.banks), f"std ref {ref:.3f} bank {w.std().item():.3f}")
