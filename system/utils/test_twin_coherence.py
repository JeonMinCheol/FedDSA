"""Checks for the twin coherence diagnostic: identical updates -> rho 1, orthogonal equal updates -> rho 1/n."""
import torch

from flcore.servers.serversa import twin_coherence


def test_coherence():
    prev = {"base.0.w": torch.zeros(4, dtype=torch.float64)}
    same = [({"base.0.w": torch.tensor([1.0, 2.0, 0.0, 0.0])}, 0.5)] * 2
    rho, mcos, _ = twin_coherence(same, prev)
    assert abs(rho - 1) < 1e-9 and abs(mcos - 1) < 1e-9
    orth = [({"base.0.w": torch.eye(4)[i] * 3}, 0.25) for i in range(4)]
    rho, mcos, norm = twin_coherence(orth, prev)
    assert abs(rho - 0.25) < 1e-9 and abs(mcos) < 1e-9 and abs(norm - 3) < 1e-9


if __name__ == "__main__":
    test_coherence()
    print("ok")
