"""Communication cost per method and accuracy vs. cumulative traffic.

Bytes are computed from what each method exchanges (fp32), not measured on the wire:
  prototypes (FedProto, FedTGP, FedSA): up = local classes x d, down = C x d
  FedGH, FedRG: up = local-class prototypes, down = shared head (d x C + C)
  LG-FedAvg: up = down = head (d x C + C)
  FML, FedAPEN, FedKD, FedMRL: up = down = shared model parameters (FedKD's SVD compression is ignored: upper bound)
  ours (twin): up = down = the client's twin backbone slice (base.0) + prototypes, and only until the gate freezes
               the twin ("TwinGate freeze" round in the log); afterwards prototypes only.

Usage: python utils/comm_cost.py --log logs/cNestG_s0.out --method twin [--widths 0.25,0.5,...] [--arch r18]
Prints, per evaluated round: cumulative MB per client (mean over clients, up + down) and test accuracy.
"""
import argparse
import json
import os
import re
from types import SimpleNamespace

MB = 2 ** 20


def local_classes(dataset):
    cfg = json.load(open(os.path.join("..", "dataset", dataset, "config.json")))
    return [len(s) for s in cfg["Size of samples for labels in clients"]], cfg["num_classes"]


def n_params(spec, args):
    from flcore.trainmodel.models import BaseHeadSplit
    a = SimpleNamespace(**vars(args))
    a.models = [spec]
    return sum(p.numel() for p in BaseHeadSplit(a, 0).base[0].parameters())


def per_round_bytes(method, dataset, d=512, widths=None, arch="r18", shared_params=None, sen_shape=None):
    """(bytes per client per round while the shared model is exchanged, bytes after a gate freeze) as lists over clients."""
    ks, C = local_classes(dataset)
    proto = [4 * (k * d + C * d) for k in ks]
    head = 4 * (d * C + C)
    if method in ("FedProto", "FedTGP", "FedSA"):
        return proto, proto
    if method in ("FedGH", "FedRG"):
        b = [4 * k * d + head for k in ks]
        return b, b
    if method == "LG-FedAvg":
        return [2 * head] * len(ks), [2 * head] * len(ks)
    if method in ("FML", "FedAPEN", "FedKD", "FedMRL"):
        return [2 * 4 * shared_params] * len(ks), [2 * 4 * shared_params] * len(ks)
    if method == "twin":
        args = SimpleNamespace(feature_dim=d, num_classes=C, fedsa_projector="linear", algorithm="FedSA", sen_shape=sen_shape)
        ws = [1.0] if not widths else widths
        if arch == "sres":
            make = lambda r: f"WidthSenResNet(args.sen_shape, args.num_classes, width={r})"
        elif arch in ("c20", "c56"):
            make = lambda r: f"CIFARResNet({int(arch[1:])}, width={r}, num_classes=args.num_classes)"
        else:
            make = lambda r: f"WidthResNet18(width={r}, num_classes=args.num_classes)"
        sizes = {r: n_params(make(r), args) for r in set(ws)}
        twin = [2 * 4 * sizes[ws[i % len(ws)]] for i in range(len(ks))]
        return [t + p for t, p in zip(twin, proto)], proto
    raise ValueError(method)


def parse_log(path):
    acc, freeze, r = {}, None, None
    for line in open(path, errors="ignore"):
        m = re.match(r"-+Round number: (\d+)", line)
        if m:
            r = int(m.group(1))
        m = re.match(r"Fused .*logit_sum=([\d.]+)", line)  # ours: the residual sum is the prediction
        if m and r is not None:
            acc[r] = float(m.group(1))
        m = re.match(r"Averaged Test Accuracy: ([\d.]+)", line)
        if m and r is not None and r not in acc:
            acc[r] = float(m.group(1))
        m = re.search(r"\[Round (\d+)\] TwinGate freeze", line)
        if m:
            freeze = int(m.group(1))
    return acc, freeze


def curve(acc, freeze, during, after):
    """[(round, cumulative MB per client, accuracy)] for the evaluated rounds."""
    md, ma = sum(during) / len(during) / MB, sum(after) / len(after) / MB
    out = []
    for r in sorted(acc):
        live = r if freeze is None else min(r, freeze)
        out.append((r, live * md + (r - live) * ma, acc[r]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", default="Cifar100")
    ap.add_argument("--widths", default="")
    ap.add_argument("--arch", default="r18")
    ap.add_argument("--shared_params", type=float, default=None, help="for FML/FedAPEN/FedKD/FedMRL")
    a = ap.parse_args()
    acc, freeze = parse_log(a.log)
    sen = (9, 1, 128) if a.dataset == "HAR" else (9, 3, 256) if a.dataset == "PAMAP2" else None
    during, after = per_round_bytes(a.method, a.dataset, widths=[float(w) for w in a.widths.split(",")] if a.widths else None,
                                    arch=a.arch, shared_params=a.shared_params, sen_shape=sen)
    print(f"# {a.log}: {sum(during) / len(during) / MB:.2f} MB/client/round while sharing, "
          f"{sum(after) / len(after) / MB:.3f} after freeze (freeze round: {freeze})")
    for r, mb, ac in curve(acc, freeze, during, after):
        print(f"R{r:<4d} {mb:10.1f} MB  acc {ac * 100:.2f}")
