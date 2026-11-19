"""Accuracy vs cumulative communication (MB per client, upload + download) for FedScout runs and HtFL baselines.

FedScout: curves from the logs (seed-mean accuracy at each eval round vs the seed-mean cumulative comm before that
round; comm_MB in the log is all clients' up + down, so per client = comm_MB / num_clients).
Baselines: only the 500R result is on this server (docs/twin_handoff.md, docs/baselines_progress.md), so each is one
point at its 500R total, from its per-round payload (d = 512 feature dim, C classes, k = mean classes per client):
  FedProto / FedTGP / FedSA : up k*d prototypes, down C*d global prototypes / anchors
  FedGH                     : up k*d prototypes, down the global head (C*d + C)
  LG-FedAvg                 : up + down the head (C*d + C)
  FedRG                     : up k*d prototypes, down k*d aggregated representations
  FedAPEN                   : up + down the shared model (CIFARResNet-<depth> with a C-class fc)
  pFedMoE                   : up + down the shared c8 extractor (no head), 78064 state floats like FedScout's count
Curve baselines (pFedMoE, FedSA, FedAPEN) are seed TSVs (round, acc %) under docs/figs/data/, exported on the node that ran them:
  python comm_vs_acc.py --export <run log> ../docs/figs/data/pfedmoe_c8_<dataset>_s<seed>.tsv
usage: python comm_vs_acc.py [--out ../docs/figs/comm_vs_acc_draft.png]"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D, ROUNDS, NC = 512, 500, 20
DATA = {"Cifar100": (100, 46.0), "Flowers102": (102, 31.5), "TinyImagenet": (200, 88.8)}  # C, mean classes/client
APEN_PARAMS = {("c20", 100): 278324, ("c20", 102): 278454, ("c20", 200): 284824,
               ("c8", 100): 83892, ("c8", 102): 84022, ("c8", 200): 90392}
BASE = {  # 500R, seed-mean curve max (docs/twin_handoff.md section 4; FedAPEN c20 from docs/baselines_progress.md)
    "LG-FedAvg": (40.64, 53.80, 23.64), "FedGH": (39.83, 53.06, 23.19), "FedProto": (38.24, 49.88, 18.24),
    "FedTGP": (44.51, 55.60, 23.20), "FedSA": (44.89, 55.59, 23.88), "FedRG": (41.98, 50.62, 24.15),  # FedSA C/F re-aggregated (progress log), Tiny = handoff
    "FedAPEN (c20)": (56.95, 68.53, 48.31)}
OURS = [  # label, log glob per dataset (seeds), color
    ("FedScout c20, personal head", "logs/m500_{d}_FedScout_s[0-2].out", "#2a78d6"),
    ("FedScout c20, share-all", {"Cifar100": "logs/abl_shareall_Cifar100_s[0-2].out",
                                 "Flowers102": "logs/m500_Flowers102_FedScoutSA_s[0-2].out"}, "#eb6834"),
    ("FedScout c8, share-all (seed 0)", "logs/c8late_{d}_s0.out", "#1baf7a"),
]
CURVE_BASE = [  # label, seed TSV glob, per-round floats per client (up + down), color
    ("pFedMoE (c8)", "../docs/figs/data/pfedmoe_c8_{d}_s[0-2].tsv", 2 * 78064, "#8a5cc7"),
    # per-round floats as a method name = per_round_floats(name, dataset); exported on ariel from the baseline logs
    ("FedSA", "../docs/figs/data/fedsa_{d}_s[0-2].tsv", "FedSA", "#8a8984"),
    ("FedAPEN (c20)", "../docs/figs/data/fedapen_c20_{d}_s[0-2].tsv", "FedAPEN (c20)", "#c98a2e"),
    ("FedAPEN (c8)", "../docs/figs/data/fedapen_c8_{d}_s[0-2].tsv", "FedAPEN (c8)", "#d4542c"),
]


def per_round_floats(method, d):
    C, k = DATA[d]
    head = C * D + C
    return {"FedProto": (k + C) * D, "FedTGP": (k + C) * D, "FedSA": (k + C) * D, "FedGH": k * D + head,
            "LG-FedAvg": 2 * head, "FedRG": 2 * k * D, "FedAPEN (c20)": 2 * APEN_PARAMS[("c20", C)],
            "FedAPEN (c8)": 2 * APEN_PARAMS[("c8", C)]}[method]


def mb(floats):
    return 4 * floats / 1e6


def curve(path):
    """[(round, acc %, cumulative MB per client before that round)] from one FedScout log."""
    txt = open(path).read()
    comm = {int(r): float(v) / NC for r, v in re.findall(r"\[Round (\d+)\] ScoutTrain .*?comm_MB=([0-9.]+)", txt)}
    acc = [(int(r), 100 * float(a)) for r, a in
           re.findall(r"Round number: (\d+)-+\n(?:.*\n)*?Averaged Test Accuracy: ([0-9.]+)", txt)]
    return [(r, a, comm.get(r - 1, 0.0)) for r, a in acc]


def tsv_curve(path, floats):
    """Same as curve() for a baseline: eval at round r comes after r rounds of the fixed per-round payload."""
    rows = [line.split("\t") for line in open(path).read().splitlines() if line]
    return [(int(r), float(a), int(r) * mb(floats)) for r, a in rows]


def seed_mean(paths, load=curve):
    cs = [load(p) for p in paths]
    n = min(len(c) for c in cs)
    if n == 0:
        return None
    rounds = [c[0] for c in cs[0][:n]]
    a = np.mean([[x[1] for x in c[:n]] for c in cs], 0)
    m = np.mean([[x[2] for x in c[:n]] for c in cs], 0)
    return rounds, a, m, len(cs)


def best_within(a, m, budget):
    ok = [x for x, y in zip(a, m) if y <= budget * 1.01]
    return max(ok) if ok else None


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../docs/figs/comm_vs_acc_draft.png")
ap.add_argument("--export", nargs=2, metavar=("LOG", "TSV"), help="write a baseline log's (round, acc %%) curve and exit")
args = ap.parse_args()
if args.export:
    txt = open(args.export[0]).read()
    assert "Best accuracy." in txt, "run not finished"
    pts = re.findall(r"Round number: (\d+)-+\n(?:.*\n)*?Averaged Test Accuracy: ([0-9.]+)", txt)
    os.makedirs(os.path.dirname(args.export[1]) or ".", exist_ok=True)
    open(args.export[1], "w").write("".join(f"{r}\t{100 * float(a):.2f}\n" for r, a in pts))
    raise SystemExit(f"wrote {len(pts)} rounds to {args.export[1]}")
os.makedirs(os.path.dirname(args.out), exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
rows = []
for j, (d, ax) in enumerate(zip(DATA, axes)):
    proto_budget = mb(per_round_floats("FedTGP", d)) * ROUNDS
    apen_budget = mb(per_round_floats("FedAPEN (c20)", d)) * ROUNDS
    for label, pat, color in OURS:
        pat = pat.get(d) if isinstance(pat, dict) else pat.format(d=d)
        paths = sorted(p for p in glob.glob(pat or "") if open(p).read().count("ScoutTrain") > 0)
        res = seed_mean(paths) if paths else None
        if res is None:
            continue
        r, a, m, n = res
        ax.plot(m[1:], a[1:], color=color, lw=2, label=f"{label} (n={n})")  # round 0 has no comm (log axis)
        rows.append((d, f"{label} (n={n}, R{r[-1]})", best_within(a, m, proto_budget), best_within(a, m, apen_budget),
                     float(a.max()), float(m[-1])))
    curved = set()  # baselines drawn as curves: their 500R point is not drawn again
    for label, pat, floats, color in CURVE_BASE:
        paths = sorted(glob.glob(pat.format(d=d)))
        if not paths:
            continue
        floats = per_round_floats(floats, d) if isinstance(floats, str) else floats
        curved.add(label)
        r, a, m, n = seed_mean(paths, lambda p: tsv_curve(p, floats))
        ax.plot(m[1:], a[1:], color=color, lw=1.6, ls="--", label=f"{label} (n={n})")
        rows.append((d, f"{label} (n={n}, R{r[-1]})", best_within(a, m, proto_budget), best_within(a, m, apen_budget),
                     float(a.max()), float(m[-1])))
    pts = sorted(((mb(per_round_floats(name, d)) * ROUNDS, vals[j], name) for name, vals in BASE.items() if name not in curved), key=lambda t: t[1])
    gap, last = 2.4, -1e9  # stacked labels: at least `gap` accuracy points apart, leader line to the dot
    for x, y, name in pts:
        ax.scatter([x], [y], s=36, color="#8a8984", zorder=3, edgecolors="#fcfcfb", linewidths=1.5)
        ly = max(y, last + gap)
        last = ly
        ax.annotate(name, (x, y), xytext=(x * 1.6, ly), fontsize=8, color="#52514e", va="center",
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6))
        rows.append((d, name + " (500R)", y if x <= proto_budget * 1.01 else None, y if x <= apen_budget * 1.01 else None,  # 1%: FedGH = prototype + C floats
                     y, x))
    ax.axvline(proto_budget, color="#c3c2b7", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(8, 4000)
    ax.text(proto_budget, ax.get_ylim()[1], "prototype methods, 500R ", fontsize=7, color="#52514e", rotation=90, va="top", ha="right")
    ax.set_title({"Cifar100": "CIFAR-100", "Flowers102": "Flowers102", "TinyImagenet": "Tiny-ImageNet"}[d], fontsize=11, color="#0b0b0b")
    ax.set_xlabel("cumulative communication per client (MB, up + down)", fontsize=9, color="#52514e")
    ax.grid(True, which="major", color="#ebebea", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
axes[0].set_ylabel("test accuracy (%)", fontsize=9, color="#52514e")
leg = {}  # union over panels: a curve may exist for only some datasets (FedAPEN c8: Flowers so far)
for a in axes:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        leg.setdefault(ll.split(" (n=")[0], hh)
fig.legend(list(leg.values()), list(leg), fontsize=9, frameon=False, loc="lower center", ncol=4)
fig.patch.set_facecolor("#fcfcfb")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
print(f"saved {args.out}")

fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
print("| dataset | method | best within prototype-500R budget | best within FedAPEN-c20-500R budget | best overall | MB used |")
print("|---|---|---|---|---|---|")
for d, name, b1, b2, best, used in rows:
    print(f"| {d} | {name} | {fmt(b1)} | {fmt(b2)} | {best:.2f} | {used:.0f} |")
print("budgets (MB per client): " + ", ".join(
    f"{d}: prototype {mb(per_round_floats('FedTGP', d)) * ROUNDS:.0f}, FedAPEN-c20 {mb(per_round_floats('FedAPEN (c20)', d)) * ROUNDS:.0f}"
    for d in DATA))
