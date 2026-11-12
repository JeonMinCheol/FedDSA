"""Experimental local evidence for RAAR beta; not enabled in existing runs."""
import math

import torch

from utils.candidate_update_trust import (
    anchor_coordinate_direction_scores,
    virtual_query_loss,
)
from utils.raar_prior_refinement import _strength


def _validate_prior(prior):
    if (prior.ndim != 1 or not prior.numel() or not prior.is_floating_point()
            or not bool(torch.isfinite(prior).all())
            or bool((prior < 0).any()) or bool((prior > 1).any())):
        raise ValueError('A finite original RAAR beta within [0, 1] is required')


@torch.no_grad()
def propose_marginal_raar_beta(prior_beta, estimates, max_weight=1.5):
    """Use local evidence to change beta, with separate extrapolation evidence.

    The three estimates must be evaluated at the original per-class beta,
    the exact EMA candidate (beta=1), and the midpoint beyond that candidate.
    Remote adverse evidence only blocks extrapolation: it must not reduce a
    low current beta where the local effect is helpful or unknown. As before,
    finite paired-query validation is required before applying a proposal.
    """
    _validate_prior(prior_beta)
    if len(estimates) != 3 or not math.isfinite(max_weight) or max_weight <= 1:
        raise ValueError('Three point estimates and a maximum above one are required')
    prior = prior_beta.double()
    favorable, adverse = [], []
    for estimate in estimates:
        def value(name, default=None):
            result = estimate.get(name, default)
            if not isinstance(result, torch.Tensor) or result.shape != prior.shape:
                raise ValueError(f'{name} needs one tensor value per class')
            result = result.to(prior)
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f'{name} must be finite')
            return result
        mean = value('mean')
        error = value('standard_error')
        count = value('effective_clients')
        downside = value('negative_mass', torch.zeros_like(prior))
        if bool((error < 0).any() or (count < 0).any() or (downside < 0).any()):
            raise ValueError('Uncertainty, support and downside must be nonnegative')
        positive, negative = _strength(mean, error, downside, count)
        if 'current_mean' in estimate:
            now_mean = value('current_mean')
            now_error = value('current_standard_error')
            now_downside = value('current_negative_mass', torch.zeros_like(prior))
            if bool((now_error < 0).any() or (now_downside < 0).any()):
                raise ValueError('Current uncertainty and downside must be nonnegative')
            now_positive, now_negative = _strength(now_mean, now_error, now_downside, count)
            positive = torch.minimum(positive, now_positive)
            negative = torch.maximum(negative, now_negative)
        favorable.append(positive)
        adverse.append(negative)
    base = prior + (1 - prior) * favorable[0] - prior * adverse[0]
    extra = (float(max_weight) - 1) * favorable[0] * favorable[1] * favorable[2]
    beta = base + extra
    unchanged = (favorable[0] == 0) & (adverse[0] == 0)
    beta = torch.where(unchanged, prior, beta)
    return {'beta': beta.to(prior_beta),
            'favorable_evidence': torch.stack(favorable),
            'adverse_evidence': torch.stack(adverse),
            'summary': {
                'mode': 'raar_geometry_prior_refinement',
                'proposal_rule': 'local_marginal_at_original_raar_beta',
                'application_reference': 'slow_stable_anchor',
                'uncertain_evidence_preserves_original_beta': True,
                'remote_harm_only_blocks_extrapolation': True,
                'local_positive_alone_maximum_beta': 1.,
                'class_mean_normalization': False,
                'coefficient_one_is_fixed_ema': True,
                'coefficient_zero_is_stable_anchor': True,
                'stable_return_retained': True,
                'maximum_beta': float(max_weight),
                'mean_beta': float(beta.mean()),
                'raised_classes': int((beta > prior + 1e-8).sum()),
                'lowered_classes': int((beta < prior - 1e-8).sum()),
                'unchanged_classes': int(unchanged.sum()),
                'mean_extra_interval_increment': float(extra.mean()),
                'score_path_required': 'original_RAAR_bank_then_EMA_boundary_and_extra',
                'proposal_only': True,
                'finite_independent_validation_required': True,
                'calibrated_probability': False}}


def marginal_raar_direction_scores(support_features, support_labels, query_features,
                                   query_labels, projector, head, stable, candidate,
                                   prior_beta, inner_lr, mse_weight, mcl_weight,
                                   mcl_gap, max_weight=1.5, return_gradients=False,
                                   mcl_gap_class_cap=False,next_gap_context=None):
    """Measure class derivatives at the actual original RAAR bank.

    All three points retain the full bank coupling of MCL. Two query halves
    estimate conditional query noise with the same normalization as the fixed
    path estimator. Agreement is absent; the supplied MCL gap is retained.
    Frozen model parameters and all RNG states remain unchanged.
    """
    _validate_prior(prior_beta)
    if (stable.ndim != 2 or stable.shape != candidate.shape
            or stable.shape[0] != prior_beta.numel()
            or not stable.is_floating_point() or stable.dtype != candidate.dtype
            or stable.device != candidate.device):
        raise ValueError('Matching floating class-by-feature anchor banks are required')
    if not all(bool(torch.isfinite(v).all()) for v in (stable, candidate)):
        raise FloatingPointError('Marginal RAAR scoring received nonfinite anchors')
    if not math.isfinite(max_weight) or max_weight <= 1:
        raise ValueError('Extrapolation needs a maximum beta above one')
    if query_labels.numel() < 2:
        raise ValueError('Two query observations are required for conditional noise')
    for name, number in [('inner_lr', inner_lr), ('mse_weight', mse_weight),
                         ('mcl_weight', mcl_weight), ('mcl_gap', mcl_gap)]:
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    direction = candidate.detach().double() - stable.detach().double()
    norm = direction.norm(dim=1)
    coefficients = torch.stack((prior_beta.to(stable), torch.ones_like(prior_beta).to(stable),
                               torch.full_like(prior_beta, 1 + (max_weight - 1) / 2).to(stable)))
    split = query_labels.numel() // 2
    fraction = split / query_labels.numel()
    scores, variances, cosines, relative_norms, point_gradients, losses = [], [], [], [], [], []
    for point in coefficients:
        weights = point[:, None]
        bank = stable.detach() + weights * (candidate.detach() - stable.detach())
        bank = torch.where(weights == 1, candidate.detach(), bank)
        gradients, half_losses = [], []
        for features, labels in ((query_features[:split], query_labels[:split]),
                                 (query_features[split:], query_labels[split:])):
            with torch.enable_grad():
                measured = bank.detach().clone().requires_grad_(True)
                loss = virtual_query_loss(support_features, support_labels, features, labels,
                                          projector, head, measured, inner_lr, mse_weight,
                                          mcl_weight, mcl_gap, create_graph=True,
                                          mcl_gap_class_cap=mcl_gap_class_cap,
                                          **({'next_gap_context':next_gap_context} if next_gap_context is not None else {}))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Marginal virtual query loss is nonfinite')
                gradient = torch.autograd.grad(loss, measured, allow_unused=True)[0]
                gradient = torch.zeros_like(measured) if gradient is None else gradient.detach()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError('Marginal virtual query gradient is nonfinite')
            gradients.append(gradient.double())
            half_losses.append(float(loss.detach()))
        first, second = gradients
        combined = fraction * first + (1 - fraction) * second
        scale = (fraction * first.square().sum() + (1 - fraction) * second.square().sum()).sqrt()
        denominator = scale * norm
        valid = denominator > 0
        gain = -(combined * direction).sum(dim=1)
        difference = -((first - second) * direction).sum(dim=1)
        score = torch.zeros_like(norm)
        sensitivity = torch.zeros_like(norm)
        score[valid] = gain[valid] / denominator[valid]
        sensitivity[valid] = difference[valid] / denominator[valid]
        auxiliary = anchor_coordinate_direction_scores(combined, direction)
        scores.append(score.clamp(-1, 1))
        variances.append(fraction * (1 - fraction) * sensitivity.square())
        cosines.append(auxiliary['class_cosine'])
        relative_norms.append(combined.norm(dim=1) / scale.clamp_min(1e-300))
        point_gradients.append(combined)
        losses.append(fraction * half_losses[0] + (1 - fraction) * half_losses[1])
    result = {
        'segment_scores': torch.stack(scores),
        'segment_valid': (norm > 0).expand(3, -1).clone(),
        'segment_query_noise_variance': torch.stack(variances),
        'segment_class_cosines': torch.stack(cosines),
        'segment_relative_gradient_norms': torch.stack(relative_norms),
        'point_coefficients': coefficients.detach(),
        'point_losses': losses,
        'score_locations': ['original_RAAR_bank', 'exact_EMA_candidate', 'beyond_EMA_midpoint'],
        'quadrature_nodes': 0,
        'query_samples': int(query_labels.numel()),
        'mcl_gap': float(mcl_gap),
        'mcl_gap_class_cap':bool(mcl_gap_class_cap),
        'margin_recomputed_for_each_bank':bool(mcl_gap_class_cap),
        'gradient_through_class_cap':bool(mcl_gap_class_cap),
        'agreement_in_virtual_update': False,
        'score_normalization': 'per_client_query_half_gradient_rms',
        'query_noise_scope': 'Two query halves conditional on one virtual support batch',
        'support_noise_separately_estimated': False,
        'within_client_gradient_magnitude_preserved': True,
    }
    if return_gradients:
        result['point_gradients'] = torch.stack(point_gradients)
    if next_gap_context is not None:
        from utils.raar_next_gap import resolve_next_gap
        result['next_gap_point_audit']={
            'next_local_gap':next_gap_context['next_local_gap'],'mode':next_gap_context['mode'],
            'raw_gaps':[float(resolve_next_gap(torch.where(point[:,None]==1,candidate.detach(),
                stable.detach()+point[:,None]*(candidate.detach()-stable.detach())),
                mcl_gap,next_gap_context)) for point in coefficients],
            'bank_dependent_global_gap_gradient':True,'latest_local_prototypes':True,
            'same_context_for_both_query_halves':True}
    return result
