"""Paper main results (outline agreed 11-20 with the external review): Table 1 (training vs inference contribution of the
private model), Figure 1 (when the private model's gain survives), Table 2 (FedScout accuracy and training cost).
Reads whatever runs have finished; missing cells print as "running". Last evaluation is the primary metric.
usage: python paper_tables.py   -> ../docs/figs/paper_table1.md, paper_table2.md, paper_fig1.png"""
import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "../docs/figs"
# FedAPEN c8 (512 x C head, coupled training) diagnostics at R500 from the ariel runs (docs/baselines_progress.md),
# seed 0 / 1 / 2: shared alone, shared + log prior, learned ensemble (main-table curve, last evaluation)
ARIEL_APEN = {"Cifar100": {"shared": [32.81, 33.40, 34.22], "shared_prior": [60.69, 61.15, 61.57], "ensemble": 54.83},
              "TinyImagenet": {"shared": [23.63, 23.59, 23.28], "shared_prior": [48.71, 48.38, 47.91], "ensemble": 41.84}}


def done(path):
    return "Best accuracy" in open(path).read()


def last_diag(path, tag):
    """last APENDiag / FMLDiag line of a finished run -> {key: %}"""
    lines = [l for l in open(path) if l.startswith(tag)]
    if not lines:
        return None
    return {k: 100 * float(v) for k, v in re.findall(r"([a-z_+]+)=([0-9.]+)", lines[-1].split("|")[0]) if "w_private" not in k}


def mean_sd(v):
    v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v), len(v)) if v else None


def fmt(ms):
    return "running" if ms is None else f"{ms[0]:.2f} ± {ms[1]:.2f} (n={ms[2]})"


# ---------------- Table 1 ----------------
rows = []
for d in ("Cifar100", "TinyImagenet"):
    ce = [last_diag(p, "APENDiag ") for p in sorted(glob.glob(f"logs/apence_{d}_s[0-2].out")) if done(p)]
    ce_sp = mean_sd([x["shared+prior"] for x in ce if x]) if ce else None
    a = ARIEL_APEN[d]
    rows.append((f"FedAPEN / {d}", fmt(ce_sp), fmt(mean_sd(a["shared"])), fmt(mean_sd(a["shared_prior"])), f"{a['ensemble']:.2f} (n=3)",
                 ce_sp, mean_sd(a["shared_prior"]), a["ensemble"]))
fml = [last_diag(p, "FMLDiag ") for p in sorted(glob.glob("logs/fml_Cifar100_s[0-2].out")) if done(p)]
fmlce = [last_diag(p, "FMLDiag ") for p in sorted(glob.glob("logs/fmlce_Cifar100_s[0-2].out")) if done(p)]
f_ce = mean_sd([x["shared_prior"] for x in fmlce if x]) if fmlce else None
f_sp = mean_sd([x["shared_prior"] for x in fml if x]) if fml else None
f_sh = mean_sd([x["shared"] for x in fml if x]) if fml else None
f_pv = mean_sd([x["private"] for x in fml if x]) if fml else None
rows.append(("FML / Cifar100 (its prediction = private model)", fmt(f_ce), fmt(f_sh), fmt(f_sp), fmt(f_pv), f_ce, f_sp, f_pv[0] if f_pv else None))
with open(f"{OUT}/paper_table1.md", "w") as f:
    f.write("# Table 1. Training vs inference contribution of the private model (last evaluation, seed mean ± std)\n\n")
    f.write("| method / data | CE-only shared + prior | coupled shared alone | coupled shared + prior | method's own prediction (FedAPEN: learned ensemble; FML: private) | training effect (coupled − CE-only, shared + prior) | inference-rule effect (own prediction − coupled shared + prior) |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for name, c1, c2, c3, c4, ce_sp, sp, own in rows:
        tr = f"{sp[0] - ce_sp[0]:+.2f}" if (ce_sp and sp) else "running"
        inf = f"{own - sp[0]:+.2f}" if (own is not None and sp) else "running"
        f.write(f"| {name} | {c1} | {c2} | {c3} | {c4} | {tr} | {inf} |\n")
    f.write("\nFedAPEN coupled rows: ariel runs (512 x C head, paper lr), R500. Tuned FedAPEN replaces them once the "
            "HP-fairness grid picks a configuration. FML uses the paper's alpha = beta = 0.5.\n")

# ---------------- Figure 1 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
INK, MUTED = "#1f2328", "#59636e"
style = {"Cifar100": ("#2a78d6", "label shift (Dir 0.1)"), "Cifar100_iid_fs": ("#eb6834", "synthetic feature shift (IID labels)")}
for key, ls, lab in (("apenswap", "-", "512×C head"), ("answap", "--", "64×C head")):
    for d, (col, dlab) in style.items():
        p = f"logs/{key}_{d}_s0.out"
        if not glob.glob(p) or not done(p):
            continue
        pts = re.findall(r"APENSwap shared@R(\d+) private@R\d+: .*?margin=([+-][0-9.]+).*?shared_nll_val=([0-9.]+)", open(p).read())
        if pts:
            x = [float(n) for _, _, n in pts]
            y = [float(m) for _, m, _ in pts]
            ax1.plot(x, y, ls, color=col, marker="o", ms=4, lw=1.6, label=f"{dlab}, {lab}")
ax1.axhline(0, color=MUTED, lw=1)
ax1.invert_xaxis()  # better shared model to the right
ax1.set_xlabel("shared model validation NLL (better →)", color=MUTED)
ax1.set_ylabel("ensemble − (shared + prior), %p", color=MUTED)
ax1.set_title("final private model × shared checkpoints (R50–R500)", color=INK, fontsize=10, loc="left")
if any(not l.get_label().startswith("_") for l in ax1.lines):  # the zero line has no label
    ax1.legend(frameon=False, fontsize=7)
else:
    ax1.text(0.5, 0.5, "checkpoint-swap runs pending", ha="center", va="center", color=MUTED, transform=ax1.transAxes)
conds = [("label shift\n(CIFAR Dir 0.1)", "lafsa8L50_Cifar100"), ("label shift\n(Flowers)", "lafsa8L50_Flowers102"),
         ("label + feature shift\n(CIFAR)", "lafsa8L50_Cifar100_fs"), ("feature shift\n(CIFAR IID)", "lafsa8L50_Cifar100_iid_fs"),
         ("real domains\n(DomainNet)", "lafsa8L50_DomainNet10_fs")]
vals = []
for lab, key in conds:
    g = []
    for p in sorted(glob.glob(f"logs/{key}_s[0-2].out")):
        if done(p):
            m = re.findall(r"ScoutEval alpha=\S+ fused=([0-9.]+) scout=([0-9.]+)", open(p).read())
            g.append(100 * (float(m[-1][0]) - float(m[-1][1])))
    vals.append(g)
xs = np.arange(len(conds))
ax2.bar(xs, [np.mean(g) if g else 0 for g in vals], color="#1baf7a", width=0.6,
        yerr=[np.std(g) if len(g) > 1 else 0 for g in vals], capsize=3)
for x, g in zip(xs, vals):
    ax2.text(x, (np.mean(g) if g else 0) + 0.15, f"n={len(g)}" if g else "pending", ha="center", fontsize=7, color=MUTED)
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_xticks(xs, [c[0] for c in conds], fontsize=7)
ax2.set_ylabel("FedScout − scout alone (head FT), %p", color=MUTED)
ax2.set_title("learner gain on top of the personalized scout (490R + 50 local)", color=INK, fontsize=10, loc="left")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/paper_fig1.png", dpi=160)

# ---------------- Table 2 ----------------
FL = json.load(open("../docs/figs/data/flops.json"))


def scout_last(key, field):
    g = []
    for p in sorted(glob.glob(f"logs/{key}_s[0-2].out")):
        if done(p):
            m = re.findall(r"ScoutEval alpha=\S+ fused=([0-9.]+) scout=([0-9.]+)", open(p).read())
            g.append(100 * float(m[-1][0 if field == "fused" else 1]))
    return mean_sd(g)


def tsv_last(key, d):
    g = []
    for p in sorted(glob.glob(f"../docs/figs/data/{key}_{d}_s[0-2].tsv")):
        r = [l.split("\t") for l in open(p) if l.strip()]
        g.append(float(r[-1][1]))
    return mean_sd(g)


DS = ("Cifar100", "Flowers102", "TinyImagenet")
t2 = [("FedAPEN (paper lr; tuned version replaces it after the grid)", [fmt(tsv_last("fedapen_c8", d)) for d in DS]),
      ("FedAPEN + prior, same checkpoint", [fmt(mean_sd(ARIEL_APEN[d]["shared_prior"])) if d in ARIEL_APEN else "running" for d in DS]),
      ("scout alone (c8 FedAvg + head FT, no private model)", [fmt(scout_last(f"lafsa8L50_{d}", "scout")) for d in DS]),
      ("FedScout (490R + 50 local)", [fmt(scout_last(f"lafsa8L50_{d}", "fused")) for d in DS])]
c = FL["Cifar100"]
n = np.array(c["n_train"])
P = np.array([c["private_native_train"][i % 8] for i in range(len(n))])
Pp = np.array([c["private_pool_train"][i % 8] for i in range(len(n))])
apen = 500 * np.mean(n * (Pp + c["apen_shared_c8_train"])) / 1e12
scout = 490 * np.mean(0.95 * n * c["c8_native_train"]) / 1e12
learner = 50 * np.mean(0.95 * n * P * 0.30) / 1e12
with open(f"{OUT}/paper_table2.md", "w") as f:
    f.write("# Table 2. FedScout accuracy and training cost (last evaluation, seed mean ± std)\n\n")
    f.write("| method | " + " | ".join(DS) + " | training compute per client, CIFAR-100 (TFLOPs) |\n|---|" + "---|" * (len(DS) + 1) + "\n")
    costs = [f"{apen:.0f}", f"{apen:.0f} (same training)", f"{scout:.0f}", f"{scout:.0f} + {learner:.1f} (learner)"]
    for (name, cells), cost in zip(t2, costs):
        f.write(f"| {name} | " + " | ".join(cells) + f" | {cost} |\n")
    f.write("\nDevice line (estimate from per-step costs measured on the device): Raspberry Pi 5, CIFAR-100, per client: "
            "FedScout 3.6 h / 80 kJ vs FedAPEN 27.7 h / 888 kJ (wall power).\n")
print(open(f"{OUT}/paper_table1.md").read())
print(open(f"{OUT}/paper_table2.md").read())
print("wrote", f"{OUT}/paper_fig1.png")
