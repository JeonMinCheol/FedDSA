"""Checks for the twin server step: server_lr scales the averaged step; agree masks disagreeing coordinates."""
import torch

from flcore.servers.serversa import twin_step


def test_twin_step():
    prev = {"base.0.w": torch.zeros(3, dtype=torch.float64)}
    states = [({"base.0.w": torch.tensor([1.0, 1.0, -1.0])}, 0.5), ({"base.0.w": torch.tensor([1.0, -1.0, -3.0])}, 0.5)]
    mean = lambda: {"base.0.w": sum(w * st["base.0.w"] for st, w in states)}  # FedAvg = [1, 0, -2]
    avg = mean()
    assert twin_step(states, avg, prev, server_lr=2.0) == 1.0
    assert torch.allclose(avg["base.0.w"], torch.tensor([2.0, 0.0, -4.0]))
    avg = mean()
    kept = twin_step(states, avg, prev, agree=1.0)  # coordinate 1: the clients disagree in sign -> masked
    assert abs(kept - 2 / 3) < 1e-9
    assert torch.allclose(avg["base.0.w"], torch.tensor([1.0, 0.0, -2.0]))
    states[1] = ({"base.0.w": torch.tensor([1.0, -3.0, -3.0])}, 0.5)
    avg = mean()  # [1, -1, -2]; coordinate 1 still disagrees -> stays at prev (0)
    twin_step(states, avg, prev, agree=1.0)
    assert torch.allclose(avg["base.0.w"], torch.tensor([1.0, 0.0, -2.0]))


if __name__ == "__main__":
    test_twin_step()
    print("ok")
