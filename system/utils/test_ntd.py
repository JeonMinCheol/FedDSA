"""Checks for the twin's FedNTD term: zero when student = teacher, no gradient on the true-class logit."""
import torch

from flcore.clients.clientsa import ntd_loss


def test_ntd():
    y = torch.tensor([0, 3])
    t = torch.randn(2, 5)
    assert ntd_loss(t.clone(), t, y).abs() < 1e-6
    s = torch.randn(2, 5, requires_grad=True)
    ntd_loss(s, t, y).backward()
    assert s.grad[0, 0] == 0 and s.grad[1, 3] == 0 and s.grad.abs().sum() > 0
    s2 = t.clone()
    s2[0, 0] += 5.0  # changing only the true-class logit leaves the loss unchanged
    assert ntd_loss(s2, t, y).abs() < 1e-6


if __name__ == "__main__":
    test_ntd()
    print("ok")
