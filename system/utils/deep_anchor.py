"""Multi-depth fixed-anchor supervision: pick intermediate stages of any backbone and project them into the
shared anchor space, so the fixed-anchor constraint also reaches the layers where memorization happens."""
import torch
import torch.nn as nn


def _stage_candidates(backbone):
    """Parameterized modules in forward-hook firing order: top-level children; if there are fewer than 4,
    long Sequentials (e.g. MobileNetV2.features) are expanded one level."""
    has_params = lambda m: any(True for _ in m.parameters())
    top = [(n, c) for n, c in backbone.named_children() if has_params(c)]
    if len(top) >= 4:  # already stage-structured (ResNet layer1..4, GoogLeNet inception blocks)
        return top
    mods = []
    for name, child in top:
        kids = list(child.named_children())
        if isinstance(child, nn.Sequential) and len(kids) >= 4:
            mods += [(f"{name}.{k}", m) for k, m in kids]
        else:
            mods.append((name, child))
    return [(n, m) for n, m in mods if has_params(m)]


@torch.no_grad()
def find_stages(backbone, x, fracs=(0.5, 0.75)):
    """Returns [(module_name, channels)] at the given fractions of the 4D-output stage list."""
    seen = []
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: seen.append((n, o)))
             for n, m in _stage_candidates(backbone)]
    was_training = backbone.training
    backbone.eval()
    try:
        backbone(x[:2])
    finally:
        backbone.train(was_training)
        for h in hooks:
            h.remove()
    order, out = [], {}
    for n, o in seen:  # first firing per module, 4D tensors only
        if n not in out and torch.is_tensor(o) and o.dim() == 4:
            out[n] = o.shape[1]
            order.append(n)
    picks = sorted({min(int(f * len(order)), len(order) - 1) for f in fracs})
    return [(order[i], int(out[order[i]])) for i in picks]


def make_projectors(stages, dim):
    return nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(c, dim)) for _, c in stages)


def attach(backbone, stages, cache):
    """Forward hooks that append each picked stage's output to `cache` (in forward order). Returns handles."""
    named = dict(backbone.named_modules())
    return [named[n].register_forward_hook(lambda m, i, o: cache.append(o)) for n, _ in stages]
