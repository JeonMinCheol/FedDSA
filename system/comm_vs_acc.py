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
Shared-model methods are drawn at the paper's capacity only (c8). Curves are seed means; the band is +-1 seed std
where at least two seeds reached that round (partial runs count up to the round they reached).
--x compute: the x axis is cumulative local training compute per client (TFLOPs; evaluation and server-side work
excluded), from FLOPs per sample measured once per model in docs/figs/data/flops.json (system/flops_table.py:
train = forward + backward through autograd, forward = eval forward). Per local sample and round:
  LG-FedAvg, FedProto, FedSA: train(private);  FedTGP, FedGH, FedRG: train(private) + forward(private) (prototype pass)
  FedAPEN, pFedMoE: train(private) + train(shared c8)
  FedScout: forward(scout) (teacher) + train(scout) + forward(scout backbone) (head realign, personal head only)
            + train(private) x priv_frac of that round (screening); after the freeze only the learner term
  shortcut: priv_frac is logged as the clients' average, so every architecture gets the same fraction; log it per
  client if architectures turn out to be screened very differently.
usage: python comm_vs_acc.py [--x comm|compute] [--out PNG]"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D, ROUNDS, NC = 512, 500, 20
DATA = {"Cifar100": (100, 46.0), "Flowers102": (102, 31.5), "TinyImagenet": (200, 88.8)}  # C, mean classes/client
APEN_PARAMS = {("c8", 100): 83892, ("c8", 102): 84022, ("c8", 200): 90392}
BASE = {  # 500R, seed-mean curve max (docs/twin_handoff.md section 4); drawn only where no curve TSV exists
    "LG-FedAvg": (40.64, 53.80, 23.64), "FedGH": (39.83, 53.06, 23.19), "FedProto": (38.24, 49.88, 18.24),
    "FedTGP": (44.51, 55.60, 23.20), "FedSA": (44.89, 55.59, 23.88), "FedRG": (41.98, 50.62, 24.15)}  # FedSA C/F re-aggregated (progress log), Tiny = handoff
# colors: the validated categorical palette in fixed order (dataviz palette.md); ours solid, baselines dashed
OURS = [  # label, log glob per dataset (seeds), color
    ("FedScout c8 (personal head)", "logs/main_c8_{d}_s[0-2].out", "#2a78d6"),
    ("FedScout c8, share-all (appendix)", "logs/c8sa40_{d}_s0.out", "#eb6834"),
    ("FedScout c8, personal head, patience 40", "logs/c8bb40_{d}_s0.out", "#1baf7a"),  # interim until main_c8 Flowers
]
CURVE_BASE = [  # label, seed TSV glob, per-round floats per client (up + down), color
    # per-round floats as a method name = per_round_floats(name, dataset); exported on ariel / from the user's logs
    ("FedAPEN (c8)", "../docs/figs/data/fedapen_c8_{d}_s[0-2].tsv", "FedAPEN (c8)", "#eda100"),
    ("pFedMoE (c8)", "../docs/figs/data/pfedmoe_c8_{d}_s[0-2].tsv", "pFedMoE (c8)", "#e87ba4"),
    ("FedSA", "../docs/figs/data/fedsa_{d}_s[0-2].tsv", "FedSA", "#008300"),
    ("FedTGP", "../docs/figs/data/fedtgp_{d}_s[0-2].tsv", "FedTGP", "#6250d6"),  # user's log (-t 3, seed arg 0)
    ("FedRG", "../docs/figs/data/fedrg_{d}_s[0-2].tsv", "FedRG", "#e34948"),  # user's log (-t 3, seed arg 0)
    # palette has 8 hues: the weaker HtFL baselines share one gray and carry a direct label at the curve end
    ("LG-FedAvg", "../docs/figs/data/lg-fedavg_{d}_s[0-2].tsv", "LG-FedAvg", None),
    ("FedGH", "../docs/figs/data/fedgh_{d}_s[0-2].tsv", "FedGH", None),
    ("FedProto", "../docs/figs/data/fedproto_{d}_s[0-2].tsv", "FedProto", None),
]
OTHER = "other HtFL (labelled)"


def per_round_floats(method, d):
    C, k = DATA[d]
    head = C * D + C
    return {"FedProto": (k + C) * D, "FedTGP": (k + C) * D, "FedSA": (k + C) * D, "FedGH": k * D + head,
            "LG-FedAvg": 2 * head, "FedRG": 2 * k * D, "FedAPEN (c8)": 2 * APEN_PARAMS[("c8", C)],
            "pFedMoE (c8)": 2 * 78064}[method]


def train_tflops(method, d):
    """Mean over clients of one round of local training compute (TFLOPs)."""
    f = FL[d]
    proto_pass = method in ("FedTGP", "FedGH", "FedRG")
    shared = {"FedAPEN (c8)": f["apen_shared_c8_train"], "pFedMoE (c8)": f["pfedmoe_shared_c8_train"]}.get(method, 0)
    return np.mean([(f["private_pool_train"][i % 8] + proto_pass * f["private_pool"][i % 8] + shared) * n
                    for i, n in enumerate(f["n_train"])]) / 1e12


def scout_tflops(d, pf, realign, frozen):
    f = FL[d]
    return np.mean([0.95 * n * ((0 if frozen else f["scout"] + f["scout_train"] + realign * f["scout_backbone"])
                                + f["private_linear_train"][i % 8] * pf) for i, n in enumerate(f["n_train"])]) / 1e12


def cost(method, d):
    """One round of the chosen x quantity for a baseline: MB (up + down) or TFLOPs of local training, per client."""
    return mb(per_round_floats(method, d)) if X == "comm" else train_tflops(method, d)


def mb(floats):
    return 4 * floats / 1e6


def curve(path, d):
    """[(round, acc %, cumulative x per client before that round)] from one FedScout log."""
    txt = open(path).read()
    tr = re.findall(r"\[Round (\d+)\] ScoutTrain .*?priv_frac=([0-9.]+).*?comm_MB=([0-9.]+) frozen=(\d)", txt)
    if X == "comm":
        cum = {int(r): float(c) / NC for r, _, c, _ in tr}
    else:
        head = re.search(r"FedScout: .*", txt).group()
        realign = int("(backbone)" in head and "realign 0" not in head)
        cum, tot = {}, 0.0
        for r, pf, _, fz in tr:
            tot += scout_tflops(d, float(pf), realign, fz == "1")
            cum[int(r)] = tot
    acc = [(int(r), 100 * float(a)) for r, a in
           re.findall(r"Round number: (\d+)-+\n(?:.*\n)*?Averaged Test Accuracy: ([0-9.]+)", txt)]
    return [(r, a, cum.get(r - 1, 0.0)) for r, a in acc]


def tsv_curve(path, per_round):
    """Same as curve() for a baseline: eval at round r comes after r rounds of the fixed per-round cost."""
    rows = [line.split("\t") for line in open(path).read().splitlines() if line]
    return [(int(r), float(a), int(r) * per_round) for r, a in rows]


def seed_mean(paths, load=curve):
    """Per eval round: mean acc, mean MB and seed std over the seeds that reached it (std nan with one seed)."""
    cs = [c for c in map(load, paths) if c]
    if not cs:
        return None
    by = {}
    for c in cs:
        for r, a, m in c:
            by.setdefault(r, []).append((a, m))
    rounds = sorted(by)
    a = np.array([np.mean([x[0] for x in by[r]]) for r in rounds])
    m = np.array([np.mean([x[1] for x in by[r]]) for r in rounds])
    sd = np.array([np.std([x[0] for x in by[r]], ddof=1) if len(by[r]) > 1 else np.nan for r in rounds])
    return rounds, a, m, sd, len(cs)


def best_within(a, m, budget):
    ok = [x for x, y in zip(a, m) if y <= budget * 1.01]
    return max(ok) if ok else None


ap = argparse.ArgumentParser()
ap.add_argument("--x", choices=["comm", "compute"], default="comm")
ap.add_argument("--out", default=None, help="default ../docs/figs/<x>_vs_acc_draft.png")
ap.add_argument("--export", nargs=2, metavar=("LOG", "TSV"), help="write a baseline log's (round, acc %%) curve and exit")
args = ap.parse_args()
if args.export:
    txt = open(args.export[0]).read()
    assert "Best accuracy." in txt, "run not finished"
    pts = re.findall(r"Round number: (\d+)-+\n(?:.*\n)*?Averaged Test Accuracy: ([0-9.]+)", txt)
    os.makedirs(os.path.dirname(args.export[1]) or ".", exist_ok=True)
    open(args.export[1], "w").write("".join(f"{r}\t{100 * float(a):.2f}\n" for r, a in pts))
    raise SystemExit(f"wrote {len(pts)} rounds to {args.export[1]}")
X = args.x
args.out = args.out or f"../docs/figs/{X}_vs_acc_draft.png"
FL = json.load(open("../docs/figs/data/flops.json")) if X == "compute" else None
UNIT = "MB" if X == "comm" else "TFLOPs"
os.makedirs(os.path.dirname(args.out), exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
rows = []
for j, (d, ax) in enumerate(zip(DATA, axes)):
    proto_budget = cost("FedTGP", d) * ROUNDS
    apen_budget = cost("FedAPEN (c8)", d) * ROUNDS
    for label, pat, color in OURS:
        pat = pat.get(d) if isinstance(pat, dict) else pat.format(d=d)
        paths = sorted(p for p in glob.glob(pat or "") if open(p).read().count("ScoutTrain") > 0)
        res = seed_mean(paths, lambda p: curve(p, d)) if paths else None
        if res is None:
            continue
        r, a, m, sd, n = res
        ax.plot(m[1:], a[1:], color=color, lw=2.2, label=f"{label} (n={n})")  # round 0 has no comm (log axis)
        ax.fill_between(m[1:], (a - sd)[1:], (a + sd)[1:], color=color, alpha=0.15, lw=0)
        rows.append((d, f"{label} (n={n}, R{r[-1]})", best_within(a, m, proto_budget), best_within(a, m, apen_budget),
                     float(a.max()), float(m[-1])))
    curved = set()  # baselines drawn as curves: their 500R point is not drawn again
    for label, pat, floats, color in CURVE_BASE:
        paths = sorted(glob.glob(pat.format(d=d)))
        if not paths:
            continue
        per_round = cost(floats, d)
        curved.add(label)
        r, a, m, sd, n = seed_mean(paths, lambda p: tsv_curve(p, per_round))
        gray = color is None
        color = "#8a8984" if gray else color
        ax.plot(m[1:], a[1:], color=color, lw=1.0 if gray else 1.6, ls="--", label=OTHER if gray else f"{label} (n={n})")
        ax.fill_between(m[1:], (a - sd)[1:], (a + sd)[1:], color=color, alpha=0.10 if gray else 0.15, lw=0)
        if gray:
            ax.annotate(label, (m[-1], a[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color="#52514e", va="center")
        rows.append((d, f"{label} (n={n}, R{r[-1]})", best_within(a, m, proto_budget), best_within(a, m, apen_budget),
                     float(a.max()), float(m[-1])))
    pts = sorted(((cost(name, d) * ROUNDS, vals[j], name) for name, vals in BASE.items() if name not in curved), key=lambda t: t[1])
    gap, last = 2.4, -1e9  # stacked labels: at least `gap` accuracy points apart, leader line to the dot
    for x, y, name in pts:
        ax.scatter([x], [y], s=36, color="#8a8984", zorder=3, edgecolors="#fcfcfb", linewidths=1.5)
        ly = max(y, last + gap)
        last = ly
        ax.annotate(name, (x, y), xytext=(x * 1.6, ly), fontsize=8, color="#52514e", va="center",
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6))
        ax.set_ylim(top=max(ax.get_ylim()[1], ly + gap))  # keep stacked labels inside the panel
        rows.append((d, name + " (500R)", y if x <= proto_budget * 1.01 else None, y if x <= apen_budget * 1.01 else None,  # 1%: FedGH = prototype + C floats
                     y, x))
    ax.axvline(proto_budget, color="#c3c2b7", lw=1, ls="--")
    ax.set_xscale("log")
    if X == "comm":
        ax.set_xlim(2, 1500)
    ax.text(proto_budget, ax.get_ylim()[1], "prototype methods, 500R ", fontsize=7, color="#52514e", rotation=90, va="top", ha="right")
    ax.set_title({"Cifar100": "CIFAR-100", "Flowers102": "Flowers102", "TinyImagenet": "Tiny-ImageNet"}[d], fontsize=11, color="#0b0b0b")
    ax.set_xlabel("cumulative communication per client (MB, up + down)" if X == "comm" else
                  "cumulative local training compute per client (TFLOPs)", fontsize=9, color="#52514e")
    ax.grid(True, which="major", color="#ebebea", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
axes[0].set_ylabel("test accuracy (%)", fontsize=9, color="#52514e")
leg = {}  # union over panels: a curve may exist for only some datasets (FedAPEN c8: Flowers so far)
for a in axes:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        leg.setdefault(ll.split(" (n=")[0], hh)
order = [x[0] for x in OURS] + [x[0] for x in CURVE_BASE if x[3]] + [OTHER]  # ours, colored baselines, gray group
names = [x for x in order if x in leg]
fig.legend([leg[x] for x in names], names, fontsize=9, frameon=False, loc="lower center", ncol=4)
fig.patch.set_facecolor("#fcfcfb")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
print(f"saved {args.out}")

fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
print(f"| dataset | method | best within prototype-500R budget | best within FedAPEN-c8-500R budget | best overall | {UNIT} used |")
print("|---|---|---|---|---|---|")
for d, name, b1, b2, best, used in rows:
    print(f"| {d} | {name} | {fmt(b1)} | {fmt(b2)} | {best:.2f} | {used:.1f} |")
print(f"budgets ({UNIT} per client): " + ", ".join(
    f"{d}: prototype {cost('FedTGP', d) * ROUNDS:.1f}, FedAPEN-c8 {cost('FedAPEN (c8)', d) * ROUNDS:.1f}" for d in DATA))
