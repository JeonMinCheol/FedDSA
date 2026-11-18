"""Checks for --fedsa_frozen_head: one fixed random head, identical on every client and model, untouched by SGD."""
import torch
import torch.nn as nn

from flcore.clients.clientsa import fix_head


def test_same_head_everywhere_and_never_trained():
    a, b = nn.Linear(512, 100), nn.Linear(512, 100)  # different default inits (two clients / twin vs private)
    fix_head(a)
    fix_head(b)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert not a.weight.requires_grad and not a.bias.requires_grad
    w = a.weight.clone()
    base = nn.Linear(8, 512)
    opt = torch.optim.SGD(list(base.parameters()) + list(a.parameters()), lr=0.1, weight_decay=1e-3)
    loss = nn.functional.cross_entropy(a(base(torch.randn(4, 8))), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    opt.step()
    assert torch.equal(a.weight, w)  # head fixed; the base still trains
    assert base.weight.grad is not None
    fix_head(a, seed=0)  # idempotent: a second call does not re-draw
    assert torch.equal(a.weight, w)


if __name__ == "__main__":
    test_same_head_everywhere_and_never_trained()
    print("ok")
