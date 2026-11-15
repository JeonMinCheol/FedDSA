"""Sign-preserving CAU gates using raw EMA and client-consistency shrinkage."""
import hashlib
import math

import torch


@torch.no_grad()
def raw_cau_weights(raw, noise, effective_clients, tau, alpha=.25):
    if raw.ndim != 1 or raw.shape != noise.shape or raw.shape != effective_clients.shape:
        raise ValueError("Aligned class vectors required")
    if not math.isfinite(tau) or not tau > 0 or not 0 < alpha < 1:
        raise ValueError("Positive finite utility scale and bounded gate strength required")
    if (not all(bool(torch.isfinite(v).all()) for v in (raw, noise, effective_clients))
            or bool((noise < 0).any()) or bool((effective_clients < 0).any())):
        raise ValueError("Finite evidence and nonnegative support/dispersion required")
    support = effective_clients / (effective_clients + 1.)
    consistency = raw.abs() / (raw.abs() + noise + 1e-12)
    shrinkage = support * consistency
    weight = 1. + alpha * shrinkage * torch.tanh(raw / tau)
    return weight.clamp(1. - alpha, 1. + alpha), shrinkage


@torch.no_grad()
def update_raw_gate_statistics(server, count_sum, client_counts, client_std):
    """Update dispersion EMA and current-cohort Kish effective client counts."""
    if not hasattr(server, "raw_gate_noise_ema"):
        server.raw_gate_noise_ema = torch.zeros_like(server.raw_class_utility)
    valid = (count_sum > 0).to(server.device)
    server.raw_gate_noise_ema[valid] = (
        (1. - server.utility_ema) * server.raw_gate_noise_ema[valid]
        + server.utility_ema * client_std.to(server.device)[valid]
    )
    server.raw_gate_effective_clients = (
        count_sum.square() / client_counts.square().sum(dim=0).clamp_min(1.)
    ).to(server.device)


@torch.no_grad()
def raw_cau_payload(server, policy, tau):
    # There is no raw observation before the first broadcast in the main code.
    raw = getattr(server, "raw_class_utility", None)
    raw = raw.detach() if raw is not None else torch.zeros_like(server.class_utility)
    noise = getattr(server, "raw_gate_noise_ema", torch.zeros_like(raw)).to(raw)
    support = getattr(server, "raw_gate_effective_clients", torch.zeros_like(raw)).to(raw)
    weight, shrinkage = raw_cau_weights(raw, noise, support, tau)
    sources = torch.arange(raw.numel(), device=raw.device)
    if policy == "raw_shuffled":
        valid = (support > 0).nonzero(as_tuple=False).flatten()
        key = f"raw_cau_gate:{int(server.random_seed)}:{int(server.current_round)}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(valid.numel(), generator=generator).to(raw.device)
        sources[valid] = valid[order]
    elif policy != "raw":
        raise ValueError("Unknown raw CAU gate policy")
    return {
        "weight": weight[sources].clone(), "utility": raw[sources].clone(),
        "shrinkage": shrinkage[sources].clone(), "noise_ema": noise[sources].clone(),
        "effective_clients": support[sources].clone(), "source_class": sources.clone(),
    }
