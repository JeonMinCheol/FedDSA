"""Checks for the width-nested twin (WidthResNet18 + HeteroFL-style averaging in serversa)."""
import copy
from types import SimpleNamespace

import torch

from flcore.servers.serversa import FedSA, _lead
from flcore.trainmodel.models import BaseHeadSplit, WidthResNet18


def _model(width):
    args = SimpleNamespace(feature_dim=512, num_classes=100, fedsa_projector="linear", algorithm="FedSA",
                           models=[f"WidthResNet18(width={width}, num_classes=args.num_classes)"])
    return BaseHeadSplit(args, 0)


def _server(models):
    full = _model(1.0)
    for m in models:
        m.load_state_dict({k: _lead(full.state_dict()[k], v) for k, v in m.state_dict().items()})
    clients = [SimpleNamespace(common_model=m, train_samples=10) for m in models]
    return SimpleNamespace(cm_global=full, selected_clients=clients, clients=clients,
                           args=SimpleNamespace(fedsa_cm_base_only=True))


def test_slices_forward():
    x = torch.randn(2, 3, 32, 32)
    for w in (0.25, 0.5, 1.0):
        assert _model(w)(x).shape == (2, 100)
    assert WidthResNet18(0.5).conv1.weight.shape == (32, 3, 7, 7)


def test_full_width_equals_plain_average():
    torch.manual_seed(0)
    s = _server([_model(1.0), _model(1.0)])
    for m in (s.clients[0].common_model, s.clients[1].common_model):
        for p in m.parameters():
            p.data.normal_()
    want = {k: (s.clients[0].common_model.state_dict()[k].float() + s.clients[1].common_model.state_dict()[k].float()) / 2
            for k in s.cm_global.state_dict() if k.startswith("base.0.")}
    FedSA.average_nested_common_models(s)
    for k, v in want.items():
        if v.is_floating_point() and "num_batches" not in k:
            assert torch.allclose(s.clients[0].common_model.state_dict()[k].float(), v, atol=1e-5), k


def test_uncovered_channels_keep_value():
    s = _server([_model(0.25), _model(0.5)])
    before = copy.deepcopy(s.cm_global.state_dict())
    for m in (s.clients[0].common_model, s.clients[1].common_model):
        for p in m.parameters():
            p.data.add_(1.0)
    FedSA.average_nested_common_models(s)
    k = "base.0.conv1.weight"  # full 64 channels; widths cover 16 and 32
    after = s.cm_global.state_dict()[k]
    assert torch.allclose(after[32:], before[k][32:])  # nobody trains channels 32..63
    assert torch.allclose(after[:16], before[k][:16] + 1.0)  # covered by both, both moved by +1
