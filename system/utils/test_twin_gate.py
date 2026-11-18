"""Checks for the shared-term memorization gate (serversa.gate_common_model)."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from flcore.servers.serversa import FedSA


class _Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))  # base.0 = shared backbone, base.1 = personal
        self.head = nn.Linear(2, 2)


def test_gate_restores_best_and_freezes():
    acc = iter([0.30, 0.50, 0.45, 0.44, 0.43])  # best at round 1, patience 2 -> freeze at round 3
    twin = _Twin()
    client = SimpleNamespace(common_model=twin, cm_frozen=False, twin_gate_stats=lambda: (round(next(acc) * 100), 0, 100))
    server = SimpleNamespace(clients=[client], args=SimpleNamespace(fedsa_cm_gate=2, fedsa_cm_val=0.05))
    best = None
    for rnd in range(5):
        if rnd == 1:
            FedSA.gate_common_model(server, rnd)
            best = twin.base[0].weight.detach().clone()
        else:
            FedSA.gate_common_model(server, rnd)
        with torch.no_grad():
            if not client.cm_frozen:
                twin.base[0].weight.add_(1.0)  # local training keeps moving the backbone
                twin.base[1].weight.add_(1.0)
    assert client.cm_frozen and server._gate_round == 1
    assert torch.allclose(twin.base[0].weight, best + 0.0)  # backbone restored to the round-1 snapshot
