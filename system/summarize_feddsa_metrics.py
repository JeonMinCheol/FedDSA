#!/usr/bin/env python
"""Summarize persistent FedDSA metric JSONL logs.

The script is deliberately independent of the training process.  It can be
run while jobs are still writing logs and will summarize the evaluations that
are present; after round 250 it gives the fixed-round and Last-50 numbers used
for the final comparison.
"""

import argparse
import glob
import json
import math
from pathlib import Path


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _get(row, path):
    value = row
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _finite(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(
        x_var * y_var
    )


METRIC_PATHS = {
    "accuracy": "test_accuracy",
    "PAR": "mapping.proto_retrieval_accuracy",
    "PAM": "mapping.proto_anchor_margin",
    "agreement_head": "train.anchor_agree_head_acc",
    "both_wrong": "train.anchor_agree_both_wrong",
    "dual_disagreement": "train.dual_disagreement_rate",
    "dual_uncertainty": "train.dual_uncertainty_mean",
    "boundary_loss": "train.boundary_loss",
    "boundary_active_rate": "train.boundary_active_rate",
    "boundary_margin": "train.boundary_margin",
    "class_boundary_loss": "train.class_boundary_loss",
    "class_boundary_active_rate": "train.class_boundary_active_rate",
    "class_boundary_mean": "new_anchor.class_boundary_mean",
    "class_boundary_step": "new_anchor.class_boundary_step",
    "class_boundary_observed_margin": "new_anchor.class_boundary_observed_margin",
    "open_world_active_classes": "open_world.active_classes",
    "open_world_new_classes": "open_world.new_classes_this_round",
    "unknown_auroc": "open_world_eval.unknown_auroc",
    "unknown_recall": "open_world_eval.unknown_recall",
    "known_accept_rate": "open_world_eval.known_accept_rate",
    "known_accuracy": "open_world_eval.known_accuracy",
    "aggregation_reliability_mean": "reliability_aggregation.mean_weight",
    "aggregation_reliability_min": "reliability_aggregation.min_weight",
    "prototype_utility_mean": "prototype_utility.mean",
    "prototype_utility_positive_rate": "prototype_utility.positive_rate",
    "mcl_tail": "train.mcl_tail_frac",
    "mcl_sample_margin": "train.mcl_sample_margin",
    "semantic_adaptive_distance": "semantic_adaptive_distance",
    "adaptive_stable_distance": "adaptive_stable_distance",
    "rank": "geometry.rank",
    "adaptive_rank": "geometry.adaptive_rank",
    "semantic_initial_drift": "geometry.semantic_initial_drift",
    "stable_initial_drift": "geometry.stable_initial_drift",
    "adaptive_initial_drift": "geometry.adaptive_initial_drift",
    "volume_ratio": "geometry.volume_ratio",
    "adaptive_volume_ratio": "geometry.adaptive_volume_ratio",
    "meta_grad_norm": "new_anchor.meta_grad_norm",
    "meta_step_norm": "new_anchor.meta_step_norm",
    "discriminative_proto_acc": "new_anchor.discriminative_proto_acc",
    "discriminative_grad_norm": "new_anchor.discriminative_grad_norm",
    "discriminative_step_norm": "new_anchor.discriminative_step_norm",
    "role_cosine": "role_separation.semantic_adaptive_cosine",
    "role_rank_gap": "role_separation.adaptive_minus_semantic_rank",
}


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and _finite(row.get("round")) is not None:
                rows.append(row)
    return sorted(rows, key=lambda row: int(row["round"]))


def summarize(path, last_round=250, last_n=5):
    rows = load_rows(path)
    if not rows:
        return {"file": str(path), "rounds": [], "error": "no valid rows"}

    observed_last = max(int(row["round"]) for row in rows)
    tail = [row for row in rows if int(row["round"]) >= max(0, observed_last - 50)]
    # If a run is incomplete, last_n evaluation points are the most useful
    # provisional tail; completed 250-round runs retain the true Last-50 set.
    if observed_last < last_round and len(tail) > last_n:
        tail = rows[-last_n:]

    fixed = next((row for row in rows if int(row["round"]) == last_round), None)
    latest = rows[-1]
    values = {
        name: [_get(row, row_path) for row in rows]
        for name, row_path in METRIC_PATHS.items()
    }
    correlations = {
        name: _pearson(values["accuracy"], series)
        for name, series in values.items()
        if name != "accuracy"
    }

    def point_summary(point_rows):
        return {
            name: _mean([_get(row, row_path) for row in point_rows])
            for name, row_path in METRIC_PATHS.items()
        }

    return {
        "file": str(path),
        "mode": latest.get("new_anchor_mode"),
        "anchor_mode": latest.get("anchor_mode"),
        "seed": latest.get("seed"),
        "rounds": [int(row["round"]) for row in rows],
        "observed_last_round": observed_last,
        "latest": point_summary([latest]),
        "fixed_round": point_summary([fixed]) if fixed else None,
        "last50_or_provisional_tail": point_summary(tail),
        "accuracy_correlations": correlations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL metric files; defaults to system/logs/feddsa_metrics_*.jsonl",
    )
    parser.add_argument("--output", default="results/feddsa_metric_summary.json")
    parser.add_argument("--last-round", type=int, default=250)
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    if not paths:
        paths = [Path(path) for path in glob.glob("system/logs/feddsa_metrics_*.jsonl")]
    summaries = [
        summarize(path, last_round=args.last_round) for path in sorted(paths) if path.exists()
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
