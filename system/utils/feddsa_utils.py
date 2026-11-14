from collections import defaultdict
import itertools
import math

import torch
import torch.nn.functional as F


def configure_utility_module(args):
    """Configure the adopted RAAR without changing the local CAU/PCW switches."""
    gamma = getattr(args, "anchor_candidate_mix", None)
    legacy_gamma = getattr(args, "moving_anchor_ema", None)
    if gamma is not None and legacy_gamma is not None and gamma != legacy_gamma:
        raise ValueError("anchor_candidate_mix and moving_anchor_ema name the same prototype fraction")
    gamma = float(gamma if gamma is not None else legacy_gamma if legacy_gamma is not None else 0.5)
    momentum = float(getattr(args, "anchor_momentum", 0.9999))
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Prototype fraction gamma must be in [0,1]")
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("Slow reference anchor momentum must be in [0,1]")
    trust = getattr(args, "fast_anchor_proto_trust", "auto")
    if trust not in {"off", "auto"}:
        raise ValueError("Use the adopted RAAR (auto) or disable it (off)")
    if getattr(args, "reliability_weighted_aggregation", False):
        raise ValueError("Class-count prototype aggregation cannot combine reliability weighting")
    args.anchor_candidate_mix = args.moving_anchor_ema = gamma
    args.anchor_momentum = momentum
    args.raar_enabled = (trust == "auto" and not getattr(args, "disable_prototype_trust", False)
                         and not getattr(args, "fixed_anchor", False))
    args.candidate_next_gap = args.candidate_gain_enabled = args.raar_enabled
    args.prototype_count_aggregation = True
    args.mcl_gap_class_cap = True
    args.prototype_trust_source = "candidate_gain"
    args.candidate_trust_reference = "geometry_prior"
    args.candidate_trust_scoring = "prior_marginal"
    args.candidate_trust_noise_model = "query_split_cohort"
    args.candidate_trust_max_weight = 1.5
    args.utility_trust_penalty = 0.25
    args.utility_trust_temperature = 0.001
    if args.raar_enabled:
        validate_candidate_next_gap(args)
    return args


def validate_candidate_next_gap(args):
    """NextGap is a server policy; preserve the local CAU and SGD gap states."""
    if getattr(args, "algorithm", "FedDSA") != "FedDSA":
        raise ValueError("NextGap requires FedDSA")
    if (getattr(args, "new_anchor_mode", "none") != "none"
            or getattr(args, "semantic_gate_mode", "off") != "off"
            or any(bool(getattr(args, key, False)) for key in (
                "dual_disagreement_gate", "freeze_semantic_anchor", "scalar_reliability_state",
                "reliability_weighted_aggregation", "use_amp"))):
        raise ValueError("NextGap RAAR requires ordinary single anchors and full precision")
    if getattr(args, "mcl_variant", "fedsa") != "fedsa":
        raise ValueError("NextGap RAAR requires distance-softmax MCL")
    if getattr(args, "mcl_gap_mode", "min") not in {"min", "max"}:
        raise ValueError("MCL gap mode must be min or max")
    ema = float(getattr(args, "utility_ema", 0.25))
    if not math.isfinite(ema) or not 0 < ema <= 1:
        raise ValueError("RAAR evidence EMA must be in (0,1]")
    return args


def _mean(values):
    return float(sum(values) / len(values)) if values else math.nan


@torch.no_grad()
def normalized_anchor_drift(anchors, initial_anchors):
    """Per-class anchor drift normalized by the nearest initial class distance."""
    anchors = torch.as_tensor(anchors, dtype=torch.float32)
    initial_anchors = torch.as_tensor(initial_anchors, dtype=torch.float32)
    distances = torch.cdist(initial_anchors, initial_anchors, p=2)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    drift = (anchors - initial_anchors).norm(p=2, dim=1)
    return torch.where(nearest > 1e-12, drift / nearest, torch.full_like(drift, torch.nan))


@torch.no_grad()
def prototype_anchor_metrics(records, adaptive_anchors, stable_anchors, active_mask=None):
    """Compute class-balanced prototype/anchor diagnostics before server updates."""
    adaptive_anchors = torch.as_tensor(adaptive_anchors, dtype=torch.float32)
    stable_anchors = torch.as_tensor(stable_anchors, dtype=torch.float32)
    adaptive_anchors = torch.nan_to_num(adaptive_anchors, nan=0.0, posinf=0.0, neginf=0.0)
    stable_anchors = torch.nan_to_num(stable_anchors, nan=0.0, posinf=0.0, neginf=0.0)
    adaptive_norm = F.normalize(adaptive_anchors, dim=1)
    stable_norm = F.normalize(stable_anchors, dim=1)

    values = defaultdict(lambda: defaultdict(list))
    prototypes = defaultdict(list)
    weighted_prototypes = defaultdict(list)
    for protos, counts in records:
        for label, proto in protos.items():
            label = int(label)
            if label < 0 or label >= adaptive_norm.size(0):
                continue
            proto = torch.as_tensor(proto, dtype=torch.float32).view(-1)
            if proto.numel() != adaptive_norm.size(1):
                continue
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            proto_norm = F.normalize(proto, dim=0)
            similarity = proto_norm @ adaptive_norm.t()
            if active_mask is not None:
                mask = torch.as_tensor(active_mask, dtype=torch.bool)
                similarity = similarity.masked_fill(~mask, -torch.inf)
            own = similarity[label]
            if similarity.numel() > 1:
                negative = similarity.clone()
                negative[label] = -torch.inf
                margin = own - negative.max()
                retrieval = float(similarity.argmax().item() == label)
            else:
                margin = torch.tensor(0.0)
                retrieval = 1.0
            values[label]["retrieval"].append(retrieval)
            values[label]["margin"].append(float(margin.item()))
            values[label]["alignment"].append(float(own.item()))
            weight = max(int(counts.get(label, 1)), 1)
            prototypes[label].append(proto)
            weighted_prototypes[label].append((proto, weight))

    by_class = {}
    for label, class_values in values.items():
        by_class[label] = {
            "proto_retrieval_accuracy": _mean(class_values["retrieval"]),
            "proto_anchor_margin": _mean(class_values["margin"]),
            "positive_alignment_cosine": _mean(class_values["alignment"]),
        }

    dispersion = []
    centers = {}
    for label, rows in weighted_prototypes.items():
        total = sum(weight for _, weight in rows)
        center = sum((proto * weight for proto, weight in rows), torch.zeros_like(rows[0][0]))
        center = F.normalize(center / max(total, 1), dim=0)
        centers[label] = center
        dispersion.append(
            _mean([float(1.0 - (F.normalize(proto, dim=0) @ center).item()) for proto in prototypes[label]])
        )

    relational_error = []
    for left, right in itertools.combinations(sorted(centers), 2):
        prototype_distance = 1.0 - float(centers[left] @ centers[right])
        stable_distance = 1.0 - float(stable_norm[left] @ stable_norm[right])
        relational_error.append(abs(prototype_distance - stable_distance))

    return {
        "proto_retrieval_accuracy": _mean(
            [row["proto_retrieval_accuracy"] for row in by_class.values()]
        ),
        "proto_anchor_margin": _mean([row["proto_anchor_margin"] for row in by_class.values()]),
        "positive_alignment_cosine": _mean(
            [row["positive_alignment_cosine"] for row in by_class.values()]
        ),
        "prototype_dispersion": _mean(dispersion),
        "relational_distance_error": _mean(relational_error),
        "by_class": by_class,
    }


def _demo():
    anchors = torch.eye(2)
    assert normalized_anchor_drift(anchors, anchors).eq(0).all()
    metrics = prototype_anchor_metrics(
        [({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}, {0: 2, 1: 2})],
        anchors,
        anchors,
    )
    assert metrics["proto_retrieval_accuracy"] == 1.0
    assert metrics["proto_anchor_margin"] > 0.9
    assert metrics["prototype_dispersion"] == 0.0
    assert metrics["relational_distance_error"] == 0.0


if __name__ == "__main__":
    _demo()


@torch.no_grad()
def aggregate_count_prototypes(records, feat_dim):
    if type(feat_dim) is not int or feat_dim <= 0:
        raise ValueError('Positive integer feature dimension required')
    groups = {};clients = set()
    for record in records:
        client = record['client'];protos = record['protos'];counts = record['counts']
        if type(client) is not int or client < 0 or client in clients:
            raise ValueError('Each client ID must be a distinct nonnegative integer')
        clients.add(client)
        if set(protos) != set(counts):
            raise ValueError('Every prototype needs its actual class sample count')
        for label, proto in protos.items():
            count = counts[label]
            if (type(label) is not int or label < 0 or type(count) is not int or count <= 0
                    or not torch.is_tensor(proto) or proto.ndim != 1 or proto.numel() != feat_dim
                    or not proto.is_floating_point() or not bool(torch.isfinite(proto).all())):
                raise ValueError('Invalid label, positive count or finite feature vector')
            groups.setdefault(label, []).append((client, count, proto))
    weighted = {};scatter = {};support = {};diagnostics = {
        'aggregation': 'actual_class_sample_count',
        'estimand': 'pooled-example centroid across observed client class means',
        'disagreement': 'count-weighted mean squared L2 residual',
        'support': 'Kish effective number of contributing clients',
        'class_mean_normalization': False, 'uniform_prototypes': {}, 'by_class': {}}
    for label, entries in groups.items():
        reference = entries[0][2]
        if any(p.dtype != reference.dtype or p.device != reference.device for _, _, p in entries):
            raise ValueError('Class prototype dtype/device must match')
        values = torch.stack([p for _, _, p in entries]).double()
        counts = torch.tensor([n for _, n, _ in entries], dtype=torch.float64, device=values.device)
        weights = counts / counts.sum()
        center = (weights[:, None] * values).sum(dim=0)
        uniform = values.mean(dim=0)
        disagreement = float((weights * (values-center).square().sum(dim=1)).sum())
        effective = float(1 / weights.square().sum())
        weighted[label] = center.to(reference).detach().clone()
        scatter[label] = disagreement;support[label] = effective
        scale = math.sqrt(disagreement)
        influences = []
        for index, (client, count, _) in enumerate(entries):
            other_mass = 1-float(weights[index])
            leave_one_out = ((center-weights[index]*values[index])/other_mass
                             if other_mass > 0 else center)
            shift = float((center-leave_one_out).norm())
            influences.append({'client': client, 'class_sample_count': count,
                'weight': float(weights[index]), 'leave_one_client_out_shift_l2': shift,
                'relative_leave_one_client_out_shift': shift/max(scale, 1e-12)})
        diagnostics['uniform_prototypes'][str(label)] = uniform.cpu().tolist()
        diagnostics['by_class'][str(label)] = {
            'client_ids': [client for client, _, _ in entries],
            'class_sample_counts': [n for _, n, _ in entries],
            'total_class_samples': int(counts.sum()), 'observed_clients': len(entries),
            'max_weight': float(weights.max()), 'effective_support': effective,
            'weighted_disagreement': disagreement,
            'uniform_to_weighted_center_l2': float((uniform-center).norm()),
            'weighted_center': center.cpu().tolist(),
            'uniform_center': uniform.cpu().tolist(), 'client_influence': influences}
    return weighted, scatter, support, diagnostics
