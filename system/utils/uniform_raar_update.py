"""Uniform RAAR direction with the accepted same-state mean movement budget.

The classwise proposal supplies the budget only. One common net-step scale
acts on both forward and slow-reference return components; it is not beta.
"""
import torch

from utils.raar_prior_refinement import apply_raar_beta


@torch.no_grad()
def apply_uniform_raar(current, candidate, stable, raw_beta, observed, gamma, raw_updated):
    if observed.dtype != torch.bool or observed.shape != raw_beta.shape:
        raise ValueError("An observation mask is required")
    beta = raw_beta.clone()
    indices = observed.nonzero(as_tuple=False).flatten()
    if indices.numel():
        beta[indices] = raw_beta[indices].double().mean().to(beta)
    base, _ = apply_raar_beta(current, candidate, stable, beta, gamma)
    target = (raw_updated.double() - current.double()).norm(dim=1).mean()
    direction = base.double() - current.double()
    mean = direction.norm(dim=1).mean()
    scale = 1.0
    fallback = None
    if float(target) == 0.0:
        updated = current.clone()
        scale = 0.0
    else:
        if float(mean) == 0.0:
            direction = candidate.double() - current.double()
            fallback = "fixed_ema_candidate"
            mean = direction.norm(dim=1).mean()
            if float(mean) == 0.0:
                direction = stable.double() - current.double()
                fallback = "slow_reference"
                mean = direction.norm(dim=1).mean()
            if float(mean) == 0.0:
                raise RuntimeError("No class-independent direction can match a positive RAAR movement")
        scale = float(target / mean)
        updated = (current.double() + scale * direction).to(current)
        for _ in range(3):
            actual_mean = (updated.double() - current.double()).norm(dim=1).mean()
            if float(actual_mean) == 0.0:
                raise RuntimeError("Positive matched movement rounded to zero")
            scale *= float(target / actual_mean)
            updated = (current.double() + scale * direction).to(current)
    actual = (updated.double() - current.double()).norm(dim=1)
    error = abs(float(actual.mean() - target))
    tolerance = 8 * torch.finfo(current.dtype).eps * max(
        1.0, float(current.double().norm(dim=1).mean()), float(target))
    if error > tolerance:
        raise RuntimeError("Final physical mean L2 movement matching failed")
    return updated, beta, {
        "allocation": "uniform_mean_matched",
        "raw_final_beta": raw_beta.cpu().tolist(),
        "controlled_base_beta": beta.cpu().tolist(),
        "observed": observed.cpu().tolist(),
        "reference_budget": "same_state_normal_RAAR_final_mean_L2_in_own_trajectory",
        "target_mean_l2": float(target),
        "actual_mean_l2": float(actual.mean()),
        "actual_step_l2": actual.cpu().tolist(),
        "mean_l2_match_abs_error": error,
        "mean_l2_match_tolerance": tolerance,
        "global_net_step_scale": scale,
        "scale_is_beta": False,
        "stable_return_in_base_update": True,
        "both_forward_and_return_scaled_together": True,
        "degenerate_direction_fallback": fallback,
        "unobserved_beta_preserved": True,
        "uses_test_data": False,
    }
