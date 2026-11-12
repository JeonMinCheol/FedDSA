"""Numerically strict, class-specific feasible Euclidean MCL margins."""
import math

import torch
import torch.nn.functional as F


def _validate(anchors, base_gap):
    if (not torch.is_tensor(anchors) or anchors.ndim != 2 or anchors.size(0) < 1
            or anchors.size(1) < 1 or not anchors.is_floating_point()
            or not bool(torch.isfinite(anchors).all())):
        raise ValueError('Finite floating class-by-feature anchors required')
    if torch.is_tensor(base_gap):
        if (base_gap.ndim != 0 or not base_gap.is_floating_point()
                or not bool(torch.isfinite(base_gap.detach()))
                or bool(base_gap.detach() < 0)):
            raise ValueError('A finite nonnegative scalar MCL gap is required')
    elif not math.isfinite(float(base_gap)) or float(base_gap) < 0:
        raise ValueError('A finite nonnegative scalar MCL gap is required')


def _safe_norm(values):
    # sqrt at zero has undefined higher derivatives. Clamp before sqrt, then
    # select exact zero, so duplicate anchors and diagonal entries stay finite.
    square = values.square().sum(dim=-1)
    safe = square.clamp_min(torch.finfo(values.dtype).tiny).sqrt()
    return torch.where(square > 0, safe, torch.zeros_like(safe))


def _mcl_gap_values(anchors, base_gap, class_cap=False):
    _validate(anchors, base_gap)
    bank = anchors.double()
    if bank.size(0) == 1:
        nearest = bank.sum(dim=1)*0
    else:
        if torch.is_grad_enabled() and bank.requires_grad:
            distances = _safe_norm(bank[:, None, :]-bank[None, :, :])
        else:
            # Frozen banks need no higher derivatives; avoid materializing a
            # classes-by-classes-by-feature double tensor for every batch.
            distances = torch.cdist(bank, bank, p=2,
                compute_mode='donot_use_mm_for_euclid_dist')
        diagonal = torch.eye(bank.size(0), dtype=torch.bool, device=bank.device)
        nearest = distances.masked_fill(diagonal, float('inf')).min(dim=1).values
    magnitude = _safe_norm(bank)
    if not bool(torch.isfinite(nearest).all()) or not bool(torch.isfinite(magnitude).all()):
        raise ValueError('Anchor-distance computation must remain finite')
    scale = torch.maximum(torch.ones_like(nearest), torch.maximum(magnitude, nearest))
    slack = 16*torch.finfo(anchors.dtype).eps*scale
    # Optional policy-generated scalar tensors retain the derivative of the
    # bank-dependent raw gap; ordinary scalar callers keep the legacy path.
    raw = (base_gap.to(nearest).expand_as(nearest) if torch.is_tensor(base_gap)
           else torch.full_like(nearest, float(base_gap)))
    applied = torch.minimum(raw, (nearest-slack).clamp_min(0)) if class_cap else raw
    return applied.to(anchors), nearest, slack


def classwise_mcl_gap(anchors, base_gap, class_cap=False):
    """Return one margin per class, retaining anchor gradients when capped.

    The numeric slack uses the original bank dtype. No statistical margin or
    freely tuned multiplier is introduced. Duplicate/single-class banks cap at
    zero; an original zero margin remains exactly zero.
    """
    _validate(anchors, base_gap)
    if not class_cap:
        if torch.is_tensor(base_gap):
            return base_gap.to(anchors).expand(anchors.size(0))
        return anchors.new_full((anchors.size(0),), float(base_gap))
    return _mcl_gap_values(anchors, base_gap, True)[0]


def add_mcl_gap(distances, labels, anchors, base_gap, class_cap=False):
    """Add a margin only to each sample's true-class distance.

    Disabled mode reproduces the old scalar expression, including returning
    the original distances object at zero gap.
    """
    _validate(anchors, base_gap)
    if (distances.ndim != 2 or distances.size(1) != anchors.size(0)
            or labels.ndim != 1 or labels.numel() != distances.size(0)):
        raise ValueError('One true label per distance row and matching classes required')
    if not torch.is_tensor(base_gap) and float(base_gap) <= 0:
        return distances
    one_hot = F.one_hot(labels, anchors.size(0)).to(device=distances.device, dtype=distances.dtype)
    if not class_cap:
        if torch.is_tensor(base_gap):
            return distances+one_hot*base_gap.to(distances)
        return distances+one_hot*float(base_gap)
    gaps = classwise_mcl_gap(anchors, base_gap, True).to(distances)
    return distances+one_hot*gaps.index_select(0, labels)[:, None]


@torch.no_grad()
def sample_gap_metrics(anchors, base_gap, labels, class_cap=False):
    """Unweighted sample means for the actual batch, independent of agreement."""
    applied, nearest, _ = _mcl_gap_values(anchors, base_gap, class_cap)
    used = applied.double().index_select(0, labels)
    distance = nearest.index_select(0, labels)
    raw = float(base_gap)
    return {'mcl_gap_raw': used.new_tensor(raw),
        'mcl_gap_applied_mean': used.mean(), 'mcl_gap_applied_min': used.min(),
        'mcl_gap_applied_max': used.max(),
        'mcl_gap_capped_fraction': (used < raw).double().mean(),
        'mcl_gap_infeasible_before_fraction': (raw > distance).double().mean(),
        'mcl_gap_infeasible_after_fraction': (used > distance).double().mean()}


@torch.no_grad()
def gap_summary(anchors, base_gap, class_cap=False, detailed=False):
    """JSON audit over classes; no query/test outcomes enter the cap."""
    applied, nearest, slack = _mcl_gap_values(anchors, base_gap, class_cap)
    values = applied.double();raw = float(base_gap)
    result = {'class_cap_enabled': bool(class_cap), 'raw_gap': raw,
        'applied_gap_mean': float(values.mean()), 'applied_gap_min': float(values.min()),
        'applied_gap_max': float(values.max()),
        'capped_classes': int((values < raw).sum()),
        'infeasible_before': int((raw > nearest).sum()),
        'infeasible_after': int((values > nearest).sum()),
        'positive_non_strict_after': int(((values > 0) & (values >= nearest)).sum()),
        'capped_fraction': float((values < raw).double().mean()),
        'infeasible_before_fraction': float((raw > nearest).double().mean()),
        'infeasible_after_fraction': float((values > nearest).double().mean()),
        'max_excess_before': float((raw-nearest).clamp_min(0).max()),
        'max_excess_after': float((values-nearest).clamp_min(0).max()),
        'nearest_min': float(nearest.min()), 'nearest_mean': float(nearest.mean()),
        'nearest_max': float(nearest.max()),
        'numeric_slack_min': float(slack.min()), 'numeric_slack_max': float(slack.max()),
        'scope': 'Necessary Euclidean anchor-distance feasibility only;not a guarantee of feature attainability or accuracy.'}
    if detailed:
        result.update(nearest_class_distance=nearest.cpu().tolist(),
            applied_gap=values.cpu().tolist(), numeric_slack=slack.cpu().tolist())
    return result
