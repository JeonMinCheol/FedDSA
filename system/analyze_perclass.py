"""Analysis B: accuracy of FedAPEN's private / shared / shared+log-prior / ensemble predictions, split by how frequent
each class is in the client's own training data (terciles of the classes present locally, plus locally absent ones).
Reads logs/<goal>_perclass.json written with --apen_dump_perclass (last evaluation).
usage: python analyze_perclass.py logs/an_Cifar100_s0_perclass.json [more.json ...]"""
import json
import sys

import numpy as np

K = ("private", "shared", "shared_prior", "ensemble")


def bins(train_n):
    """class index -> bin name, per client: terciles of present classes by local train count, 'absent' otherwise."""
    present = sorted((c for c, n in enumerate(train_n) if n > 0), key=lambda c: -train_n[c])
    out = {c: "absent" for c, n in enumerate(train_n) if n == 0}
    for r, c in enumerate(present):
        out[c] = ("majority", "middle", "minority")[min(2, 3 * r // len(present))]
    return out


def table(path):
    clients = json.load(open(path))["clients"].values()
    tot = {b: {k: 0 for k in K + ("n",)} for b in ("majority", "middle", "minority", "absent", "all")}
    for cl in clients:
        for c, b in bins(cl["train_n"]).items():
            for k in K + ("n",):
                tot[b][k] += cl[k][c]
                tot["all"][k] += cl[k][c]
    print(path)
    print(f"  {'bin':9s} {'test n':>7s} " + " ".join(f"{k:>12s}" for k in K) + "   ens-(sh+pr)  priv-(sh+pr)")
    for b, t in tot.items():
        if t["n"]:
            a = {k: 100 * t[k] / t["n"] for k in K}
            print(f"  {b:9s} {t['n']:7d} " + " ".join(f"{a[k]:12.2f}" for k in K)
                  + f"   {a['ensemble'] - a['shared_prior']:+11.2f}  {a['private'] - a['shared_prior']:+12.2f}")
    keys = [k for k in next(iter(clients)) if k.startswith("tau")]  # stored as tau0.5, tau2.0, ...
    if keys:  # prior-correction strength sweep, all classes: 0 = shared, 1 = shared_prior
        acc = {0.0: tot["all"]["shared"], 1.0: tot["all"]["shared_prior"]}
        acc.update({float(k[3:]): sum(sum(cl[k]) for cl in clients) for k in keys})
        print("  tau sweep (g + tau*log prior): " + "  ".join(f"{t:g}:{100 * acc[t] / tot['all']['n']:.2f}" for t in sorted(acc)))
    return tot


if __name__ == "__main__":
    # self-check on a toy client: class 0 frequent, 1 middle, 2 rare, 3 absent
    assert bins([30, 20, 10, 0]) == {0: "majority", 1: "middle", 2: "minority", 3: "absent"}
    for p in sys.argv[1:]:
        table(p)
