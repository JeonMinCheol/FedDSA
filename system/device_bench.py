"""On-device cost of one local training step (Jetson Nano / Raspberry Pi / any CPU or GPU).

Standalone on purpose: copy this single file to the device. Needs only torch + torchvision
(Python 3.6 / torch 1.10 compatible, so it also runs in the Jetson Nano l4t-pytorch container).

Each workload is one SGD step on a synthetic CIFAR-shaped batch (3x32x32, 100 classes); timing does not depend on
pixel values. Workloads mirror the training code (system/flcore/clients/clientsa.py, final config):
  local         private fwd+bwd                              (= Local; also the device cost of FedProto/FedTGP)
  fedapen       private + shared twin, both fwd+bwd          (= FedAPEN/FML-style shared model)
  ours_pre      private on CE(teacher_twin + private) + twin fwd+bwd + frozen-teacher fwd   (before the gate freezes)
  ours_post     same after the freeze: twin backbone frozen (fwd only), projector/head still train
  ours_cache    after the freeze with cached twin features (no twin backbone at all)
  ours_gate_K   ours_pre where the private model only sees K% of the batch (twin-gated private training)
  realign       one head-realignment step (twin backbone fwd, projector/head fwd+bwd); runs once per round epoch
  twin_train    the twin alone, fwd+bwd

usage:
  python device_bench.py --device cuda --archs cnn,mnv2,gnet,r18 --batch 32 --steps 20 --out nano.csv
  python device_bench.py --device cpu  --archs cnn,mnv2 --batch 32 --steps 10 --out pi.csv
Energy: on Jetson the script reads tegrastats (POM_5V_IN / VDD_IN) automatically; on a Raspberry Pi use an external
USB power meter and align it with the printed start/end timestamps.
"""
import argparse
import csv
import re
import resource
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_CLASSES, FEAT_DIM = 100, 512  # CIFAR-100, FedSA projector width (-fd 512)


# ---- models (copied from system/flcore/trainmodel/models.py) ----
class _WBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.down is None else self.down(x)))


class CIFARResNet(nn.Module):
    def __init__(self, depth=20, width=1.0, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        c = [max(4, int(round(16 * width * m))) for m in (1, 2, 4)]
        self.conv1 = nn.Conv2d(3, c[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c[0])
        stages, cin = [], c[0]
        for i, cout in enumerate(c):
            stages.append(nn.Sequential(_WBlock(cin, cout, 1 if i == 0 else 2), *[_WBlock(cout, cout, 1) for _ in range(n - 1)]))
            cin = cout
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(c[2], num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class FedAvgCNN(nn.Module):
    def __init__(self, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.fc1(torch.flatten(self.conv2(self.conv1(x)), 1)))


def backbone(arch):
    """architecture with its classifier removed, and the classifier's input width"""
    tv = torchvision.models
    m = {"cnn": lambda: FedAvgCNN(NUM_CLASSES),
         # the repo's mobilenet_v2 is torchvision's topology with `fc` instead of `classifier`; same compute
         "mnv2": lambda: tv.mobilenet_v2(num_classes=NUM_CLASSES),
         "gnet": lambda: tv.googlenet(num_classes=NUM_CLASSES, aux_logits=False, init_weights=False),
         "r18": lambda: tv.resnet18(num_classes=NUM_CLASSES), "r34": lambda: tv.resnet34(num_classes=NUM_CLASSES),
         "r50": lambda: tv.resnet50(num_classes=NUM_CLASSES), "r101": lambda: tv.resnet101(num_classes=NUM_CLASSES),
         "r152": lambda: tv.resnet152(num_classes=NUM_CLASSES)}
    if arch[0] == "c" and arch[1:].isdigit():
        net = CIFARResNet(int(arch[1:]), num_classes=NUM_CLASSES)
    else:
        net = m[arch]()
    if hasattr(net, "classifier"):  # torchvision mobilenet_v2: Sequential(Dropout, Linear)
        dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
    else:
        dim = net.fc.in_features
        net.fc = nn.Identity()
    return net, dim


class Split(nn.Module):
    """BaseHeadSplit with the linear projector: backbone -> Linear(dim, 512) -> head Linear(512, classes)"""

    def __init__(self, arch):
        super().__init__()
        self.backbone, dim = backbone(arch)
        self.proj = nn.Linear(dim, FEAT_DIM)
        self.head = nn.Linear(FEAT_DIM, NUM_CLASSES)

    def top(self, f):
        return self.head(self.proj(torch.flatten(f, 1)))

    def forward(self, x):
        return self.top(self.backbone(x))


# ---- workloads: each is one optimizer step ----
def make_workloads(priv, twin, teacher, x, y, feat):
    ce = F.cross_entropy
    opt_p = torch.optim.SGD(priv.parameters(), lr=0.01)
    opt_t = torch.optim.SGD(twin.parameters(), lr=0.03)

    def step(loss, opts):
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()

    def local():
        step(ce(priv(x), y), [opt_p])

    def fedapen():
        lp, lt = priv(x), twin(x)
        w = 0.2  # learned ensemble weight; its cost is negligible
        step(ce(lp, y) + ce(lt, y) + ce((1 - w) * lt.detach() + w * lp, y), [opt_p, opt_t])

    def ours_pre(k=1.0):
        with torch.no_grad():
            t = teacher(x)
        m = max(2, int(round(k * x.shape[0])))
        step(ce(priv(x[:m]) + t[:m], y[:m]) + ce(twin(x), y), [opt_p, opt_t])

    def ours_post():
        with torch.no_grad():
            t = teacher(x)
            f = twin.backbone(x)  # frozen backbone (eval BN)
        step(ce(priv(x) + t, y) + ce(twin.top(f), y), [opt_p, opt_t])

    def ours_cache():
        with torch.no_grad():
            t = teacher.top(feat)
        step(ce(priv(x) + t, y) + ce(twin.top(feat), y), [opt_p, opt_t])

    def realign():
        with torch.no_grad():
            f = twin.backbone(x)
        step(ce(twin.top(f), y), [opt_t])

    def twin_train():
        step(ce(twin(x), y), [opt_t])

    w = {"local": local, "fedapen": fedapen, "ours_pre": ours_pre, "ours_post": ours_post, "ours_cache": ours_cache,
         "realign": realign, "twin_train": twin_train}
    for k in (60, 40, 20):
        w["ours_gate_%d" % k] = (lambda k=k: ours_pre(k / 100.0))
    return w


# ---- power (Jetson tegrastats) ----
class Tegrastats(object):
    def __init__(self):
        self.samples, self.proc = [], None
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, universal_newlines=True)
        except OSError:
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            m = re.search(r"(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?/", line)
            if m:
                self.samples.append((time.time(), int(m.group(1)) / 1000.0))

    def mean_watts(self, t0, t1):
        v = [w for t, w in self.samples if t0 <= t <= t1]
        return sum(v) / len(v) if v else float("nan")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--archs", default="cnn,mnv2,gnet,r18")
    ap.add_argument("--twin", default="c20")
    ap.add_argument("--workloads", default="local,fedapen,ours_pre,ours_post,ours_cache,ours_gate_60,ours_gate_40,realign,twin_train")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--out", default="device_bench.csv")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    x = torch.randn(a.batch, 3, 32, 32, device=dev)
    y = torch.randint(0, NUM_CLASSES, (a.batch,), device=dev)
    power = Tegrastats()
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)

    twin, teacher = Split(a.twin).to(dev).train(), Split(a.twin).to(dev).eval()
    with torch.no_grad():
        feat = twin.backbone.eval()(x).detach()
    twin.train()
    print("twin %s backbone params %d (upload = download = %.2f MB fp32 per round)"
          % (a.twin, sum(p.numel() for p in twin.backbone.parameters()),
             4 * sum(v.numel() for v in twin.backbone.state_dict().values() if v.is_floating_point()) / 1e6))
    rows = []
    for arch in a.archs.split(","):
        priv = Split(arch).to(dev).train()
        works = make_workloads(priv, twin, teacher, x, y, feat)
        for name in a.workloads.split(","):
            if name in ("twin_train", "realign") and rows and any(r["workload"] == name for r in rows):
                continue  # does not depend on the private architecture
            fn = works[name]
            try:
                for _ in range(a.warmup):
                    fn()
                sync()
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                times = []
                t0 = time.time()
                for _ in range(a.steps):
                    s = time.perf_counter()
                    fn()
                    sync()
                    times.append(time.perf_counter() - s)
                t1 = time.time()
            except RuntimeError as e:  # e.g. out of memory on a small device
                print("%s/%s failed: %s" % (arch, name, str(e).splitlines()[0]))
                if cuda:
                    torch.cuda.empty_cache()
                continue
            ms = 1000 * sum(times) / len(times)
            sd = (sum((1000 * t - ms) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            watts = power.mean_watts(t0, t1)
            r = {"device": a.device, "arch": arch if name not in ("twin_train", "realign") else "-", "twin": a.twin,
                 "workload": name, "batch": a.batch, "ms_per_step": round(ms, 1), "ms_std": round(sd, 1),
                 "samples_per_s": round(1000 * a.batch / ms, 1), "peak_mem_MB": round(mem, 1),
                 "avg_power_W": round(watts, 2), "energy_J_per_step": round(watts * ms / 1000, 3),
                 "t_start": round(t0, 1), "t_end": round(t1, 1)}
            rows.append(r)
            print("%-5s %-13s %8.1f ms/step (+-%.1f)  %7.1f samples/s  mem %7.1f MB  %5.2f W"
                  % (r["arch"], name, ms, sd, r["samples_per_s"], mem, watts))
        del priv
        if cuda:
            torch.cuda.empty_cache()
    power.stop()
    with open(a.out, "w") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
