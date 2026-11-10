from collections import defaultdict
import itertools
import math

import torch
import torch.nn.functional as F


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
