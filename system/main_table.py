"""Main table with the protocol the external review asked for: next to the seed-mean best of the test curve (our
protocol so far, optimistic for every method alike), the last evaluation (no checkpoint selection) and paired per-seed
differences against FedAPEN; FedScout rows split into scout alone (c8 FedAvg + head fine-tuning, no private model),
scout + log prior, and the fused prediction (scout + alpha * learner).
usage: python main_table.py > ../docs/figs/main_table.md"""
import glob
import re

import numpy as np

DS = ("Cifar100", "Flowers102", "TinyImagenet")
BASE = [("FedProto", "fedproto"), ("FedGH", "fedgh"), ("LG-FedAvg", "lg-fedavg"), ("FedRG", "fedrg"), ("FedTGP", "fedtgp"),
        ("FedSA", "fedsa"), ("FedMRL (c8)", "fedmrl_c8"), ("pFedMoE (c8)", "pfedmoe_c8"), ("FedAPEN (c8)", "fedapen_c8")]
OURS = [("FedScout, learner every round (500R)", "npsa8"), ("FedScout 2-stage, 490R + 10 local", "lafsa8"),
        ("FedScout 2-stage, 490R + 50 local (final)", "lafsa8L50")]


def tsv(path):
    d = {}
    for line in open(path):
        p = line.split("\t")
        try:
            d[int(float(p[0]))] = float(p[1])
        except (ValueError, IndexError):
            pass
    return d


def scout_log(path):
    """{round: {'fused', 'scout', 'scout_prior'}} in %, from one FedScout log (evaluations every 10 rounds)."""
    txt = open(path).read()
    out = {}
    for r, block in re.findall(r"Round number: (\d+)-+\n(.*?)(?=-+Round number|\Z)", txt, re.S):
        m = re.search(r"ScoutEval alpha=\S+ fused=([0-9.]+) scout=([0-9.]+) scout_prior=([0-9.]+)", block)
        if m:
            out[int(r)] = {"fused": 100 * float(m.group(1)), "scout": 100 * float(m.group(2)), "scout_prior": 100 * float(m.group(3))}
    return out


def summarize(curves):
    """curves: list of {round: acc} (one per seed) -> best of the seed mean, last common round, per-seed last values."""
    if len(curves) < 1:
        return None
    rs = sorted(set.intersection(*[set(c) for c in curves]))
    if not rs:
        return None
    m = np.array([[c[r] for c in curves] for r in rs])
    i = int(m.mean(1).argmax())
    return {"best": (m.mean(1)[i], m.std(1)[i], rs[i]), "last": (m[-1].mean(), m[-1].std(), rs[-1]), "last_seeds": m[-1], "n": len(curves)}


def fmt(s):
    if s is None:
        return "—"
    b, l = s["best"], s["last"]
    return f"{b[0]:.2f} ± {b[1]:.2f} / {l[0]:.2f} (R{l[2]}, n={s['n']})"


rows, apen_last = [], {}
for name, key in BASE:
    cells = []
    for d in DS:
        s = summarize([tsv(p) for p in sorted(glob.glob(f"../docs/figs/data/{key}_{d}_s[0-2].tsv"))])
        cells.append(fmt(s))
        if key == "fedapen_c8" and s and s["n"] == 3:
            apen_last[d] = s["last_seeds"]
    rows.append((name, cells))
paired = []
for name, key in OURS:
    for field, label in (("scout", "scout alone (c8 FedAvg + head FT)"), ("scout_prior", "scout + log prior"), ("fused", "fused (scout + α·learner)")):
        cells = []
        for d in DS:
            logs = [scout_log(p) for p in sorted(glob.glob(f"logs/{key}_{d}_s[0-2].out")) if "Best accuracy" in open(p).read()]
            s = summarize([{r: v[field] for r, v in lg.items()} for lg in logs]) if logs else None
            cells.append(fmt(s))
            if field == "fused" and s and s["n"] == 3 and d in apen_last:
                paired.append((name, d, s["last_seeds"] - apen_last[d]))
        rows.append((f"{name}: {label}", cells))

print("# Main table (seed-mean best ± std / last evaluation)\n")
print("Each cell: best of the seed-mean test curve ± std at that round / mean of the last evaluation (round, seeds). "
      "FedScout rows: scout alone = the shared c8 after FedAvg with its head fine-tuned on local data for one epoch (no "
      "private model); fused = scout + alpha * learner (alpha on held-out).\n")
print("| method | " + " | ".join(DS) + " |")
print("|---|" + "---|" * len(DS))
for name, cells in rows:
    print(f"| {name} | " + " | ".join(cells) + " |")
print("\n## Paired per-seed differences at the last evaluation (FedScout fused − FedAPEN, same seed and split)\n")
print("| FedScout config | dataset | seed 0 | seed 1 | seed 2 | mean |")
print("|---|---|---|---|---|---|")
for name, d, diff in paired:
    print(f"| {name} | {d} | " + " | ".join(f"{v:+.2f}" for v in diff) + f" | {diff.mean():+.2f} |")
