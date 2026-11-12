"""Opt-in next-local raw MCL gap policy for virtual candidate banks.

The fresh local-prototype gap is cached data. The bank's global gap is the
mean of each class's nearest-other distance, as in the actual local trainer.
This only changes the server candidate oracle; it does not update local gaps.
"""
import math

import torch

from utils.mcl_gap import _safe_norm, _validate


def _constant_gap(value, name):
    if torch.is_tensor(value):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(name+' must be a finite nonnegative scalar')
        value = value.detach().item()
    if isinstance(value, bool):
        raise ValueError(name+' must be a finite nonnegative scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name+' must be a finite nonnegative scalar') from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(name+' must be a finite nonnegative scalar')
    return result


def resolve_next_gap(anchors, old_gap, context=None):
    """Return the unchanged old scalar, or a bank-differentiable future gap.

    Context requires ``next_local_gap`` and ``mode`` (min/max). Both local and
    global gaps must be positive even in max mode, matching mcl_gap_value.
    Call inside the B3 gradient-enabled scope to retain the global derivative.
    """
    if context is None:
        return float(old_gap)
    if not isinstance(context, dict):
        raise ValueError('next_gap_context must be a dictionary')
    if 'next_local_gap' not in context or 'mode' not in context:
        raise ValueError('next_gap_context requires next_local_gap and mode')
    local = _constant_gap(context['next_local_gap'], 'next_local_gap')
    mode = context['mode']
    if mode not in ('min', 'max'):
        raise ValueError('next-gap mode must be min or max')
    _validate(anchors, old_gap)
    bank = anchors.double()
    if bank.size(0) == 1 or local <= 0:
        return (bank.sum()*0).to(anchors)
    if torch.is_grad_enabled() and bank.requires_grad:
        distances = _safe_norm(bank[:, None, :]-bank[None, :, :])
    else:
        distances = torch.cdist(bank, bank, p=2,
            compute_mode='donot_use_mm_for_euclid_dist')
    diagonal = torch.eye(bank.size(0), dtype=torch.bool, device=bank.device)
    nearest = distances.masked_fill(diagonal, float('inf')).min(dim=1).values
    if not bool(torch.isfinite(nearest).all()):
        raise ValueError('Next-gap anchor distances must remain finite')
    global_gap = nearest.mean()
    if bool(global_gap.detach() <= 0):
        return (global_gap*0).to(anchors)
    local_gap = global_gap.new_tensor(local)
    result = (torch.minimum(local_gap, global_gap) if mode == 'min'
              else torch.maximum(local_gap, global_gap))
    return result.to(anchors)


def candidate_next_gap_kwargs(client, cache):
    """Read a fresh post-local cache only for the explicitly enabled policy."""
    if not bool(getattr(client, 'candidate_next_gap', False)):
        return {}
    if not isinstance(cache, dict) or cache.get('candidate_next_gap') is not True:
        raise ValueError('Enabled next-gap policy requires its fresh candidate cache')
    if 'next_local_gap' not in cache or 'next_gap_mode' not in cache:
        raise ValueError('Next-gap candidate cache is incomplete')
    local = _constant_gap(cache['next_local_gap'], 'next_local_gap')
    mode = cache['next_gap_mode']
    if mode not in ('min', 'max'):
        raise ValueError('next_gap_mode must be min or max')
    return {'next_gap_context': {'next_local_gap': local, 'mode': mode}}
