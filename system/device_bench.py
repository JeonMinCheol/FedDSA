"""On-device cost of FedScout's local work and the baselines' (Raspberry Pi / Jetson Nano / any CPU or GPU),
plus ONNX export of the inference models for a Hailo NPU (inference only; Hailo cannot train).

Standalone on purpose: copy this single file to the device. Needs only torch + torchvision
(Python 3.6 / torch 1.10 compatible, so it also runs in the Jetson Nano l4t-pytorch container).

Final FedScout (11-20, docs/paper_story.md section 4): c8 scout and private learners without projector (each model's
own classifier), scout averaged for 490 rounds while the learner does not train, then the learner trains locally
(no communication) on the samples the frozen scout is unsure of. Training workloads (one SGD step on a synthetic
batch; timing does not depend on pixel values), mirroring system/flcore/clients/clientscout.py:
  scout_train        c8 scout fwd+bwd                    (FedScout rounds 1-490: the device's only federated work)
  learner_post_kK    frozen scout fwd (all samples) + learner fwd+bwd on K% of the batch (local stage after the freeze;
                     the logged screening fraction is about 30%, so K=30 is the representative one)
  every_round_kK     scout fwd (teacher) + learner fwd+bwd on K% + scout fwd+bwd (learner trained every round, npsa8)
  local              private fwd+bwd                     (per-round device cost of LG-FedAvg / FedProto / FedTGP / FedSA)
  fedapen            private + c8 fwd+bwd + ensemble CE  (FedAPEN)
Inference workloads (eval, no grad, --infer_batch, default 1 = latency):
  infer_scout, infer_private      cascade latency = infer_scout + (fraction the scout is unsure of) x infer_private
Total device training time of a run = epochs x (local samples / batch) x ms_per_step, e.g. FedScout
  490 x n/B x scout_train + E_local x n/B x learner_post_k30   vs   FedAPEN 500 x n/B x fedapen.

usage:
  python device_bench.py --device cuda --archs cnn,mnv2,gnet,r18 --size 32 --classes 100 --out nano_cifar.csv
  python device_bench.py --device cpu  --archs cnn,mnv2,gnet,r18 --size 32 --classes 100 --steps 10 --out pi_cifar.csv
  python device_bench.py --export_onnx onnx_cifar --size 32 --classes 100     (then compile each .onnx to .hef with the
      Hailo Dataflow Compiler on an x86 host and run `hailortcli benchmark <model>.hef` on the device)
--size 64 --classes 102 for Flowers102, --size 64 --classes 200 for TinyImagenet.
Energy: on Jetson the script reads tegrastats (POM_5V_IN / VDD_IN, board input) automatically; on a Raspberry Pi 5 it
sums the PMIC rails (`vcgencmd pmic_read_adc`, SoC + memory, a lower bound of wall power). For wall power use an
external USB meter and align it with the printed start/end timestamps.
"""
import argparse
import csv
import os
import re
import resource
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_CLASSES, SIZE = 100, 32  # set from --classes / --size in main()


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
    sp = ((SIZE - 4) // 2 - 4) // 2  # FedAvgCNN flatten size (main.fedavg_cnn_input_dim)
    m = {"cnn": lambda: FedAvgCNN(NUM_CLASSES, dim=64 * sp * sp),
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


class Native(nn.Module):
    """BaseHeadSplit with --fedsa_projector native (FedScout --scout_projector none): backbone -> own classifier"""

    def __init__(self, arch):
        super().__init__()
        self.backbone, dim = backbone(arch)
        self.head = nn.Linear(dim, NUM_CLASSES)

    def forward(self, x):
        return self.head(torch.flatten(self.backbone(x), 1))


# ---- workloads: each is one optimizer step (training) or one forward (inference) ----
def make_workloads(priv, scout, x, y):
    ce = F.cross_entropy
    opt_p = torch.optim.SGD(priv.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(scout.parameters(), lr=0.03)

    def step(loss, opts):
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()

    def sub(k):
        return max(2, int(round(k * x.shape[0])))  # BatchNorm in train mode needs >= 2 samples

    def scout_train():
        step(ce(scout(x), y), [opt_s])

    def learner_post(k):
        scout.eval()
        with torch.no_grad():
            t = scout(x)  # screening needs the frozen scout on every sample
        scout.train()
        m = sub(k)
        step(ce(priv(x[:m]) + t[:m], y[:m]), [opt_p])

    def every_round(k):
        scout.eval()
        with torch.no_grad():
            t = scout(x)  # the received scout as teacher
        scout.train()
        m = sub(k)
        step(ce(priv(x[:m]) + t[:m], y[:m]) + ce(scout(x), y), [opt_p, opt_s])

    def local():
        step(ce(priv(x), y), [opt_p])

    def fedapen():
        lp, ls = priv(x), scout(x)
        w = 0.2  # learned ensemble weight; its cost is negligible
        step(ce(lp, y) + ce(ls, y) + ce((1 - w) * ls.detach() + w * lp, y), [opt_p, opt_s])

    w = {"scout_train": scout_train, "local": local, "fedapen": fedapen}
    for k in (100, 30):
        w["learner_post_k%d" % k] = (lambda k=k: learner_post(k / 100.0))
    w["every_round_k85"] = lambda: every_round(0.85)  # npsa8 logs 74-97% of samples for the learner
    return w


def make_infer(priv, scout, xi):
    def run(m):
        m.eval()
        with torch.no_grad():
            m(xi)
    return {"infer_scout": lambda: run(scout), "infer_private": lambda: run(priv)}


def export_onnx(archs, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    xi = torch.randn(1, 3, SIZE, SIZE)
    for arch in ["c8"] + archs:
        m = Native(arch).eval()
        path = os.path.join(out_dir, "%s_%dpx_%dcls.onnx" % ("scout_c8" if arch == "c8" else arch, SIZE, NUM_CLASSES))
        torch.onnx.export(m, xi, path, opset_version=11, input_names=["input"], output_names=["logits"])
        print("wrote", path)


# ---- power: Jetson tegrastats (board input) or Raspberry Pi 5 PMIC (sum over SoC/memory rails) ----
class Tegrastats(object):
    """Jetson: tegrastats POM_5V_IN / VDD_IN (whole-board input). Raspberry Pi 5 without tegrastats: `vcgencmd
    pmic_read_adc`, power = sum over rails of V x A. The PMIC rails exclude USB peripherals and the 5 V input
    conversion loss, so the Pi number is a lower bound of wall power (use an external meter for the board total)."""

    def __init__(self):
        self.samples, self.proc, self.source, self.running = [], None, "none", True
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, universal_newlines=True)
            self.source = "tegrastats"
            threading.Thread(target=self._read, daemon=True).start()
            return
        except OSError:
            pass
        try:
            subprocess.check_output(["vcgencmd", "pmic_read_adc"], stderr=subprocess.DEVNULL)
            self.source = "pmic"
            threading.Thread(target=self._pmic, daemon=True).start()
        except (OSError, subprocess.CalledProcessError):
            pass

    def _read(self):
        for line in self.proc.stdout:
            m = re.search(r"(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?/", line)
            if m:
                self.samples.append((time.time(), int(m.group(1)) / 1000.0))

    def _pmic(self):
        while self.running:
            out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], universal_newlines=True)
            amps = dict(re.findall(r"(\S+)_A current\(\d+\)=([0-9.]+)A", out))
            volts = dict(re.findall(r"(\S+)_V volt\(\d+\)=([0-9.]+)V", out))
            self.samples.append((time.time(), sum(float(amps[k]) * float(volts[k]) for k in amps if k in volts)))
            time.sleep(0.1)

    def mean_watts(self, t0, t1):
        v = [w for t, w in self.samples if t0 <= t <= t1]
        return sum(v) / len(v) if v else float("nan")

    def stop(self):
        self.running = False
        if self.proc is not None:
            self.proc.terminate()


def main():
    global NUM_CLASSES, SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--archs", default="cnn,mnv2,gnet,r18")
    ap.add_argument("--size", type=int, default=32, help="input side: 32 CIFAR-100, 64 Flowers102 / TinyImagenet")
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--workloads", default="scout_train,learner_post_k30,learner_post_k100,every_round_k85,local,fedapen,"
                                           "infer_scout,infer_private")
    ap.add_argument("--batch", type=int, default=32, help="training batch")
    ap.add_argument("--infer_batch", type=int, default=1, help="inference batch (1 = per-sample latency)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--export_onnx", default="", help="write ONNX models for the Hailo compiler into this dir and exit")
    ap.add_argument("--out", default="device_bench.csv")
    a = ap.parse_args()
    NUM_CLASSES, SIZE = a.classes, a.size
    if a.export_onnx:
        export_onnx(a.archs.split(","), a.export_onnx)
        return
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    x = torch.randn(a.batch, 3, SIZE, SIZE, device=dev)
    y = torch.randint(0, NUM_CLASSES, (a.batch,), device=dev)
    xi = torch.randn(a.infer_batch, 3, SIZE, SIZE, device=dev)
    power = Tegrastats()
    print("power source:", power.source)
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)

    scout = Native("c8").to(dev).train()
    print("scout c8 (no projector) params %d (upload = download = %.3f MB fp32 per round, whole model)"
          % (sum(p.numel() for p in scout.parameters()),
             4 * sum(v.numel() for v in scout.state_dict().values() if v.is_floating_point()) / 1e6))
    rows = []
    for arch in a.archs.split(","):
        priv = Native(arch).to(dev).train()
        works = make_workloads(priv, scout, x, y)
        works.update(make_infer(priv, scout, xi))
        for name in a.workloads.split(","):
            if name in ("scout_train", "infer_scout") and any(r["workload"] == name for r in rows):
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
            nb = a.infer_batch if name.startswith("infer") else a.batch
            r = {"device": a.device, "arch": arch if name not in ("scout_train", "infer_scout") else "-", "size": SIZE,
                 "classes": NUM_CLASSES, "workload": name, "batch": nb, "ms_per_step": round(ms, 1), "ms_std": round(sd, 1),
                 "samples_per_s": round(1000 * nb / ms, 1), "peak_mem_MB": round(mem, 1),
                 "avg_power_W": round(watts, 2), "energy_J_per_step": round(watts * ms / 1000, 3),
                 "t_start": round(t0, 1), "t_end": round(t1, 1)}
            rows.append(r)
            print("%-5s %-17s %8.1f ms/step (+-%.1f)  %7.1f samples/s  mem %7.1f MB  %5.2f W"
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
