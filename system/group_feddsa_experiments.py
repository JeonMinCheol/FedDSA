#!/usr/bin/env python3
"""Summarize real log files stored together for each FedDSA experiment."""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

from summarize_feddsa_metrics import load_rows


CONFIG = Path(__file__).resolve().parents[1] / "logs/experiment_groups/groups.json"
LATE_ROUNDS = range(450, 501, 10)


def run_summary(run):
    if run.get("evaluation_source") == "stdout":
        path = Path(run["stdout"])
        rows = []
        if path.exists():
            current_round = None
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = re.search(r"Round number:\s*(\d+)", line)
                    if match:
                        current_round = int(match.group(1))
                    match = re.search(r"Averaged Test Accuracy:\s*([0-9.]+)", line)
                    if match and current_round is not None:
                        rows.append({"round": current_round, "test_accuracy": float(match.group(1)), "seed": run["seed"]})
                        current_round = None
    else:
        path = Path(run["metrics"])
        rows = load_rows(path) if path.exists() else []
    accuracies = {}
    for row in rows:
        round_id, accuracy = row["round"], row.get("test_accuracy")
        if row.get("seed") != run["seed"] or round_id in accuracies:
            raise ValueError(f"Mixed seed or duplicate round in {path}: {round_id}")
        if not isinstance(round_id, int) or not isinstance(accuracy, (float, int)):
            raise ValueError(f"Invalid evaluation in {path}: {round_id}")
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid accuracy in {path}: {accuracy}")
        accuracies[round_id] = accuracy
    latest = max(accuracies, default=None)
    return {
        **run,
        "evaluation_count": len(accuracies),
        "latest_round": latest,
        "latest_accuracy": accuracies.get(latest),
        "round500_accuracy": accuracies.get(500),
        "last50_accuracy": mean(accuracies[r] for r in LATE_ROUNDS)
        if all(r in accuracies for r in LATE_ROUNDS) else None,
    }


def group_summary(group):
    runs = [run_summary(run) for run in group["runs"]]
    if not runs or len({run["seed"] for run in runs}) != len(runs):
        raise ValueError(f"Missing or repeated seeds in {group['id']}")
    result = {**group, "runs": runs}
    for key in ("round500_accuracy", "last50_accuracy"):
        values = [run[key] for run in runs]
        result[key] = mean(values) if all(v is not None for v in values) else None
    return result


def percent(value):
    return "—" if value is None else f"{100 * value:.4f}%"


def refresh(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    summaries = [group_summary(group) for group in config["groups"]]
    names = [group["id"] for group in summaries]
    aliases = config.get("aliases", {})
    for name in [*names, *aliases]:
        if not name or any(not (c.isalnum() or c in "_-") for c in name):
            raise ValueError(f"Invalid group name: {name}")
    references = [(r["job_id"], r["seed"], r["stdout"]) for g in summaries for r in g["runs"]]
    if len(set(names)) != len(names) or len(set(references)) != len(references):
        raise ValueError("Repeated group or training run; use a shared reference alias")
    if any(alias in names or target not in names for alias, target in aliases.items()):
        raise ValueError("Invalid shared reference alias")

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    index = [
        "# 실험 그룹", "", f"요약 갱신: {timestamp}", "", config["note"], "",
        "그룹별 폴더에 실제 로그 파일을 모았다. 대기 작업의 로그는 실행 시작 후 생성된다.",
        "학습 진행은 각 그룹의 seed0.out·seed1.out에서 확인한다. controller.out은 부모 관리 로그다.",
        "학습 출력·정확도는 10라운드마다, 상세 게이트 로그는 매 라운드 기록된다. 이 요약 표는 아래 갱신 명령 실행 시 업데이트된다.",
        "평균 정확도는 모든 예정 시드의 해당 평가가 있을 때만 표시하며, 단위는 %다.",
        "기존 3시드 그룹과 현재 2시드 그룹의 평균은 시드 구성이 다르다.", "",
        "| 설정 그룹 | 시드 | 작업 ID | 평가된 라운드 | 500라운드 평균 | 450–500 평균 |",
        "|---|---|---|---|---|---|",
    ]
    for group in summaries:
        folder = output / group["id"]
        folder.mkdir(exist_ok=True)
        readme = [
            f"# {group['label']}", "", group["protocol"], "",
            f"요약 갱신: {timestamp}. 이 폴더의 시드별 파일은 실제 원본 로그다.", "",
            "| 시드 | 작업 ID | 평가된 라운드 | 500라운드 | 450–500 평균 | 로그 |",
            "|---|---|---|---|---|---|",
        ]
        for run in group["runs"]:
            seed = run["seed"]
            progress = run["latest_round"] if run["latest_round"] is not None else "미시작"
            links = []
            for key, suffix in (("stdout", "out"), ("metrics", "metrics.jsonl"),
                                ("gate_metrics", "gates.jsonl")):
                if run.get(key):
                    artifact = Path(run[key])
                    if artifact.is_symlink():
                        raise ValueError(f"Expected a real log file: {artifact}")
                    name = artifact.relative_to(folder)
                    links.append(f"[{key}]({name})")
            readme.append(
                f"| {seed} | {run['job_id']} | {progress} | "
                f"{percent(run['round500_accuracy'])} | {percent(run['last50_accuracy'])} | "
                + " · ".join(links) + " |"
            )
        readme += ["", f"전체 시드 500라운드 평균: {percent(group['round500_accuracy'])}",
                   f"전체 시드 450–500 평균: {percent(group['last50_accuracy'])}", "",
                   "설정과 원본 경로: [summary.json](summary.json)"]
        (folder / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        (folder / "summary.json").write_text(
            json.dumps({"updated_at": timestamp, **group}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        seeds = ", ".join(str(r["seed"]) for r in group["runs"])
        jobs = ", ".join(str(j) for j in dict.fromkeys(r["job_id"] for r in group["runs"]))
        progress = ", ".join(f"s{r['seed']}: " + (str(r["latest_round"])
                             if r["latest_round"] is not None else "미시작")
                             for r in group["runs"])
        index.append(f"| [{group['label']}]({group['id']}/README.md) | {seeds} | {jobs} | "
                     f"{progress} | {percent(group['round500_accuracy'])} | "
                     f"{percent(group['last50_accuracy'])} |")
    for alias, target in aliases.items():
        index += ["", f"공통 기준: [{alias}]({target}/README.md) (실제 파일은 {target} 한 곳에 저장)."]
    if (output / "older_runs/README.md").exists():
        index += ["", "[이전 추가 실험·통합 보고서](older_runs/README.md)"]
    index += ["", "요약 다시 갱신: `python system/group_feddsa_experiments.py` (저장소 루트에서 실행)"]
    (output / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return summaries


if __name__ == "__main__":
    groups = refresh()
    print(f"Refreshed {len(groups)} experiment groups in {CONFIG.parent}")
