"""When does the private model help? FedAPEN diagnostics at every evaluation of one run: private, shared, shared + log
prior and the learned ensemble against the round, and the ensemble's margin over shared + prior against the shared
model's own accuracy (the native-head run flips sign as the shared model trains: R330 +2.4, R500 -1.5 on Dir 0.1).
usage: python apen_trajectory.py logs/an_Cifar100_s0.out [more logs] --out ../docs/figs/apen_trajectory.png"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse(path):
    txt = open(path).read()
    rows = []
    for r, block in re.findall(r"Round number: (\d+)-+\n(.*?)(?=-+Round number|\Z)", txt, re.S):
        d = re.search(r"APENDiag private=([0-9.]+) shared=([0-9.]+) shared\+prior=([0-9.]+)", block)
        e = re.search(r"Averaged Test Accuracy: ([0-9.]+)", block)
        if d and e:
            rows.append((int(r), *(100 * float(v) for v in d.groups()), 100 * float(e.group(1))))
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--out", default="../docs/figs/apen_trajectory.png")
a = ap.parse_args()
INK, MUTED = "#1f2328", "#59636e"
# one-hue sequential map for the round, its palest quarter cut off so the earliest points stay visible
ROUND_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("rounds", plt.get_cmap("Oranges")(np.linspace(0.3, 1.0, 256)))
COL = {"private": "#2a78d6", "shared": "#9a6b00", "shared + prior": "#1baf7a", "ensemble": "#eb6834"}
fig, axes = plt.subplots(len(a.logs), 2, figsize=(10.5, 3.3 * len(a.logs)), squeeze=False)
for row, path in zip(axes, a.logs):
    rs = parse(path)
    r, pv, sh, sp, en = zip(*rs)
    ax = row[0]
    for lab, ys in (("private", pv), ("shared", sh), ("shared + prior", sp), ("ensemble", en)):
        ax.plot(r, ys, color=COL[lab], lw=2, label=lab)
    ax.set_xlabel("round", color=MUTED)
    ax.set_ylabel("test accuracy (%)", color=MUTED)
    ax.set_title(path.split("/")[-1].replace(".out", ""), color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax2 = row[1]
    margin = [e - s for e, s in zip(en, sp)]
    ax2.axhline(0, color=MUTED, lw=1)
    # R0 (untrained models, shared ~1%) is left out of the margin panel: it is not a trained checkpoint and stretches the axes
    keep = [k for k, rr in enumerate(r) if rr > 0]
    # points, not a line: the shared accuracy is not monotone in the round (it wobbles late in training), so joining the
    # points in round order zigzags left and right; the round is the color instead (light = early, dark = late)
    sc = ax2.scatter([sh[k] for k in keep], [margin[k] for k in keep], c=[r[k] for k in keep], cmap=ROUND_CMAP,
                     vmin=0, vmax=max(r), s=16, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax2, pad=0.01)
    cb.set_label("round", color=MUTED)
    cb.outline.set_visible(False)
    ax2.set_xlabel("shared model alone, test accuracy (%)", color=MUTED)
    ax2.set_ylabel("ensemble − (shared + prior), %p", color=MUTED)
    ax2.set_title("ensemble margin vs shared-model quality (R10–R500)", color=INK, fontsize=10, loc="left")
    for x in (ax, ax2):
        x.spines[["top", "right"]].set_visible(False)
        x.grid(alpha=0.25, lw=0.6)
    flip = next((rr for rr, m in zip(r, margin) if rr > 50 and m < 0), None)
    print(f"{path}: margin at last eval {margin[-1]:+.2f} (R{r[-1]}), max {max(margin):+.2f} (R{r[margin.index(max(margin))]}), first sign flip below 0 after R50: R{flip}")
fig.tight_layout()
fig.savefig(a.out, dpi=160)
print("wrote", a.out)
