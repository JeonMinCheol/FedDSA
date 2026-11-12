"""Isolate registered nn.Dropout; functional dropout is outside this helper."""
import hashlib

import torch
from torch import nn


class IsolatedDropout(nn.Dropout):
    def reset(self, seed):
        self._seed = int(seed)
        self._calls = {}

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x.mul_(0) if self.inplace else x * 0
        device = str(x.device)
        call = self._calls.get(device, 0)
        self._calls[device] = call + 1
        # Store integer counters, not CUDA generators, in model checkpoints.
        generator = torch.Generator(device=x.device).manual_seed(
            (self._seed + call) % (2**63 - 1)
        )
        mask = torch.rand(x.shape, device=x.device, generator=generator) >= self.p
        return x.mul_(mask).div_(1 - self.p) if self.inplace else x * mask / (1 - self.p)


def prepare_dropout(model, seed, client_id, round_idx):
    """Reset a private stream per registered dropout for this client's round.

    Spatial/alpha and custom dropout modules are rejected; direct functional
    stochastic operations must be audited separately for new backbones.
    """
    unsupported = (nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                   nn.AlphaDropout, nn.FeatureAlphaDropout)
    prepared = {}
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if isinstance(module, unsupported):
            raise TypeError(f"Unsupported stochastic dropout module: {name}")
        if not isinstance(module, nn.Dropout):
            continue
        if type(module) not in (nn.Dropout, IsolatedDropout):
            raise TypeError(f"Unsupported custom dropout module: {name}")
        original_id = id(module)
        parent_name, _, child_name = name.rpartition(".")
        if original_id in prepared:
            setattr(model.get_submodule(parent_name), child_name, prepared[original_id])
            continue
        if not isinstance(module, IsolatedDropout):
            if not name:
                raise TypeError("prepare_dropout requires a model containing dropout modules")
            replacement = IsolatedDropout(module.p, module.inplace)
            replacement.train(module.training)
            setattr(model.get_submodule(parent_name), child_name, replacement)
            module = replacement
        key = f"{int(seed)}:{int(client_id)}:{int(round_idx)}:{name}".encode()
        module.reset(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little"))
        prepared[original_id] = module
