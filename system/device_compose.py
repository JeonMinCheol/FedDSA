"""Compose per-step device measurements into whole-run device training time / energy per client (CIFAR-100 split,
HtFE-img-8: client i runs arch i % 8). Methods (500 rounds, 1 local epoch):
  FedScout 2-stage: 490 x scout_train + E_LOCAL (= 50, final) x learner_post_k30
  every-round learner (500 rounds): 490 x every_round_k85 + 10 x learner_post_k30
  FedAPEN: 500 x fedapen        private-only baselines (LG-FedAvg, FedProto, FedTGP, FedSA): 500 x local
per epoch = n_i / B steps. Round wall-clock = slowest client (straggler). usage: python device_compose.py ../docs/figs/data/device/orin_cifar.csv Cifar100"""
import csv, json, sys
import numpy as np
E_LOCAL = 50  # local learner epochs after the R490 freeze (final config, 11-20)
*paths, d = sys.argv[1:]
rows = [r for p in paths for r in csv.DictReader(open(p))]
ms = {(r["arch"], r["workload"]): float(r["ms_per_step"]) for r in rows}
J = {(r["arch"], r["workload"]): float(r["energy_J_per_step"]) for r in rows}
B = int(rows[0]["batch"]) if rows[0]["workload"] != "infer_scout" else 32
ARCH = ["cnn", "gnet", "mnv2", "r18", "r34", "r50", "r101", "r152"]  # main.py HtFE-img-8 order
n = json.load(open("../docs/figs/data/flops.json"))[d]["n_train"]
clients = [(ARCH[i % 8], n[i]) for i in range(len(n)) if ("-" if False else ARCH[i % 8], "local") in ms]
print(f"{paths[0].split('/')[-1]}: {len(clients)}/{len(n)} clients have measurements (archs: {sorted({a for a, _ in clients})})")
def total(a, k, which):
    t = ms if which == "s" else J
    s = lambda w, arch=a: t[(arch, w)] if (arch, w) in t else t[("-", w)]
    steps = k / B
    return {"FedScout 2-stage": steps * (490 * s("scout_train") + E_LOCAL * s("learner_post_k30")),
            "every-round learner": steps * (490 * s("every_round_k85") + 10 * s("learner_post_k30")),
            "FedAPEN": steps * 500 * s("fedapen"),
            "private-only (LG/Proto/TGP/SA)": steps * 500 * s("local")}
meth = list(total(*clients[0], "s"))
tt = {m: [total(a, k, "s")[m] / 3.6e6 for a, k in clients] for m in meth}  # hours
ee = {m: [total(a, k, "e")[m] / 1e3 if not np.isnan(total(a, k, "e")[m]) else np.nan for a, k in clients] for m in meth}  # kJ
rt = {m: max(tt[m]) for m in meth}
print(f"  {'method':32s} {'mean h/client':>13s} {'slowest client h':>16s} {'mean kJ/client':>15s} {'vs FedScout (mean time)':>24s}")
for m in meth:
    print(f"  {m:32s} {np.mean(tt[m]):13.2f} {rt[m]:16.2f} {np.nanmean(ee[m]):15.1f} {np.mean(tt[m]) / np.mean(tt['FedScout 2-stage']):23.1f}x")
