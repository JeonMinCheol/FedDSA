"""What does the heterogeneous learner cost on top of the same scout, and what does it buy? (external review, item 15)
and the cascade's accuracy-latency trade-off (item 14), from the FedScout logs and the device measurements.
- accuracy: scout alone vs fused at the last evaluation (seed mean), i.e. the learner's contribution
- training cost of the learner stage only (TFLOPs per client from flops.json, Pi 5 / Orin hours from device CSVs)
- cascade: accuracy and expected per-sample latency = scout + (fraction the scout is unsure of) x private, per threshold
usage: python learner_cost.py > ../docs/figs/learner_cost_cascade.md"""
import csv
import glob
import json
import re

import numpy as np

FL = json.load(open("../docs/figs/data/flops.json"))
ARCH = ["cnn", "gnet", "mnv2", "r18", "r34", "r50", "r101", "r152"]
DEV = {"Pi 5": ("../docs/figs/data/device/wall_pi_{d}_wall.csv", "../docs/figs/data/device/wall_pi_{d}_big_wall.csv"),
       "Orin Nano": ("../docs/figs/data/device/orin_{d}.csv",)}
RUNS = [("Cifar100", "lafsa8", 10, "cifar"), ("Flowers102", "lafsa8L50", 50, "flowers")]


def last_eval(path):
    m = re.findall(r"ScoutEval alpha=\S+ fused=([0-9.]+) scout=([0-9.]+).*?((?:cascade[0-9.]+=[0-9.]+ run[0-9.]+=[0-9.]+ ?)+)", open(path).read())
    fu, sc, cas = m[-1]
    taus = {float(t): (float(a), float(r)) for t, a, t2, r in re.findall(r"cascade([0-9.]+)=([0-9.]+) run([0-9.]+)=([0-9.]+)", cas)}
    return 100 * float(fu), 100 * float(sc), taus


def device(dname, d):
    ms = {}
    for pat in DEV[dname]:
        for p in glob.glob(pat.format(d=d)):
            for r in csv.DictReader(open(p)):
                ms[(r["arch"], r["workload"])] = float(r["ms_per_step"]) / int(r["batch"])  # per sample
    return ms


print("# Learner add-on cost vs gain, and cascade accuracy-latency\n")
print("## Learner on top of the same scout (seed mean, last evaluation)\n")
print("| dataset (local epochs) | scout alone | fused | learner gain | learner-stage training, TFLOPs/client | Pi 5 learner stage, h/client | Orin learner stage, h/client |")
print("|---|---|---|---|---|---|---|")
for d, key, E, dk in RUNS:
    ev = [last_eval(p) for p in sorted(glob.glob(f"logs/{key}_{d}_s[0-2].out"))]
    fu, sc = np.mean([e[0] for e in ev]), np.mean([e[1] for e in ev])
    f = FL[d]
    n = np.array(f["n_train"])
    P = np.array([f["private_native_train"][i % 8] for i in range(len(n))])
    tfl = E * np.mean(0.95 * n * P * 0.30) / 1e12  # screening keeps ~30% of the samples for the learner
    hours = {}
    for dn in DEV:
        ms = device(dn, dk)
        hours[dn] = np.mean([E * 0.95 * n[i] * ms.get((ARCH[i % 8], "learner_post_k30"), np.nan) for i in range(len(n))]) / 3.6e6
    print(f"| {d} ({E}) | {sc:.2f} | {fu:.2f} | {fu - sc:+.2f} | {tfl:.1f} | {hours['Pi 5']:.2f} | {hours['Orin Nano']:.2f} |")

print("\n## Cascade: run the learner only where the scout's top-1 probability is below tau (last evaluation, seed mean)\n")
print("Latency per sample = scout + run fraction x private (batch 1, mean over the 20 clients' architectures).\n")
print("| dataset | tau | accuracy | learner run fraction | Pi 5 latency, ms | Orin latency, ms |")
print("|---|---|---|---|---|---|")
for d, key, E, dk in RUNS:
    ev = [last_eval(p) for p in sorted(glob.glob(f"logs/{key}_{d}_s[0-2].out"))]
    lat = {}
    for dn in DEV:
        ms = device(dn, dk)
        sc_ms = ms[("-", "infer_scout")]
        pv_ms = np.mean([ms[(ARCH[i % 8], "infer_private")] for i in range(20)])
        lat[dn] = (sc_ms, pv_ms)
    rows = [("scout only", np.mean([e[1] for e in ev]), 0.0)]
    for t in sorted(ev[0][2]):
        rows.append((f"{t:g}", 100 * np.mean([e[2][t][0] for e in ev]), np.mean([e[2][t][1] for e in ev])))
    rows.append(("always (fused)", np.mean([e[0] for e in ev]), 1.0))
    for t, acc, run in rows:
        print(f"| {d} | {t} | {acc:.2f} | {run:.2f} | " + " | ".join(f"{lat[dn][0] + run * lat[dn][1]:.1f}" for dn in DEV) + " |")
