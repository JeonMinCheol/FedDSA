"""Real federated rounds on lab devices over the LAN (external review, item 14): measured, not composed.
One server (aggregates) and N clients (Raspberry Pi 5 / Jetson Orin), real CIFAR-100 client partitions, TCP transfer.
Per round: synchronous round time (broadcast -> last update), each client's compute time, bytes up/down.
Methods (same local work as the simulation, one local epoch per round):
  fedapen   private + shared c8 (pool projector to 512, 512 x C head) mutual learning + ensemble weight learned on a
            held-out 5% before training (FedAPEN); the shared model is uploaded
  fedscout  only the c8 scout (own 64 x C head) trains, CE, lr 0.03 (head x8); after the last round each client runs the
            local stage: head personalization (1 epoch) + E epochs of the residual learner on the samples the frozen scout
            is unsure of (p_y < 0.5), then a batch-1 cascade over its test set (learner only where the scout's top-1
            probability < 0.5): latency p50 / p95 and accuracy. Peak memory is reported for every phase.
Standalone (torch + torchvision); model code copied from device_bench.py. Energy is logged outside (J7-C USB meter on
the Pi, tegrastats on Jetson) and aligned with the phase timestamps written here.
usage:
  server: python fl_device.py server --method fedscout --devices 10 --rounds 10 --port 29500 --out fl_fedscout.json
  client: python fl_device.py client --host 192.168.0.11 --port 29500 --cids 3,16 --archs r18,cnn --device cpu   (data: client_<cid>_{train,test}.npz)"""
import argparse
import io
import json
import pickle
import resource
import socket
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_CLASSES = 100


# ---- models (device_bench.py / system/flcore/trainmodel/models.py) ----
class _WBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.down is None else self.down(x)))


class CIFARResNet(nn.Module):
    def __init__(self, depth=8, num_classes=10):
        super().__init__()
        n, c = (depth - 2) // 6, [16, 32, 64]
        self.conv1, self.bn1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
        stages, cin = [], 16
        for i, cout in enumerate(c):
            stages.append(nn.Sequential(_WBlock(cin, cout, 1 if i == 0 else 2), *[_WBlock(cout, cout, 1) for _ in range(n - 1)]))
            cin = cout
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(x))))))
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
    tv = torchvision.models
    m = {"cnn": lambda: FedAvgCNN(NUM_CLASSES), "mnv2": lambda: tv.mobilenet_v2(num_classes=NUM_CLASSES),
         "gnet": lambda: tv.googlenet(num_classes=NUM_CLASSES, aux_logits=False, init_weights=False),
         "r18": lambda: tv.resnet18(num_classes=NUM_CLASSES), "r34": lambda: tv.resnet34(num_classes=NUM_CLASSES),
         "r50": lambda: tv.resnet50(num_classes=NUM_CLASSES), "r101": lambda: tv.resnet101(num_classes=NUM_CLASSES),
         "r152": lambda: tv.resnet152(num_classes=NUM_CLASSES), "c8": lambda: CIFARResNet(8, NUM_CLASSES)}
    net = m[arch]()
    if hasattr(net, "classifier"):
        dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
    else:
        dim = net.fc.in_features
        net.fc = nn.Identity()
    return net, dim


class Native(nn.Module):
    """backbone -> own classifier (FedScout scout and learner, --scout_projector none)"""

    def __init__(self, arch):
        super().__init__()
        self.backbone, dim = backbone(arch)
        self.head = nn.Linear(dim, NUM_CLASSES)

    def forward(self, x):
        return self.head(torch.flatten(self.backbone(x), 1))


class Pool512(nn.Module):
    """backbone -> AdaptiveAvgPool1d to 512 -> head 512 x C (baselines' BaseHeadSplit with the pool projector)"""

    def __init__(self, arch):
        super().__init__()
        self.backbone, _ = backbone(arch)
        self.pool = nn.AdaptiveAvgPool1d(512)
        self.head = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        return self.head(self.pool(torch.flatten(self.backbone(x), 1)[:, None, :])[:, 0])


# ---- messaging: 8-byte length + pickle protocol 4 (Python 3.6 on Jetson Nano); state dicts with torch's legacy
# serialization so torch 1.10 (Nano) and 2.x (Pi, Orin, server) read each other's files ----
def send(sock, obj):
    b = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(b)) + b)
    return len(b) + 8


def recv(sock):
    def exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise ConnectionError("peer closed")
            buf += chunk
        return bytes(buf)
    n = struct.unpack("!Q", exact(8))[0]
    return pickle.loads(exact(n)), n + 8


def pack(state):
    bio = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, bio, _use_new_zipfile_serialization=False)
    return bio.getvalue()


def unpack(b):
    try:
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    except TypeError:  # torch 1.10 has no weights_only
        return torch.load(io.BytesIO(b), map_location="cpu")


def to_tensor(arr, dtype):
    """numpy -> torch through raw bytes: the Jetson Orin torch wheel is built against NumPy 1.x and cannot read NumPy 2
    arrays (torch.frombuffer exists from torch 1.10)"""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32 if dtype == np.float32 else torch.int64)
    return t.reshape(arr.shape)


def peak_mem_mb(dev):
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def shared_model(method):
    return Native("c8") if method == "fedscout" else Pool512("c8")


# ---- server: one connection per device; a device hosts one or more clients and trains them one after another ----
def server(a):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(a.devices)
    conns = []
    while len(conns) < a.devices:
        c, addr = srv.accept()
        hello, _ = recv(c)
        conns.append((c, hello))
        print(f"device {hello['host']} ({hello['device']}) from {addr[0]}: clients " +
              ", ".join(f"{x['cid']}:{x['arch']}(n={x['n']})" for x in hello["clients"]), flush=True)
    torch.manual_seed(0)
    state = shared_model(a.method).state_dict()
    log = {"method": a.method, "devices": [h for _, h in conns], "rounds": [], "evals": []}

    def evaluate(r):
        for c, _ in conns:
            send(c, {"type": "eval", "state": pack(state)})
        res = []
        for c, h in conns:
            msg, _ = recv(c)
            res += msg["clients"]
        acc = 100 * sum(x["correct"] for x in res) / sum(x["n"] for x in res)
        log["evals"].append({"round": r, "t": time.time(), "acc": acc, "clients": res})
        print(f"eval after round {r}: accuracy {acc:.2f}", flush=True)

    t_begin = time.time()
    for r in range(a.rounds):
        blob = pack(state)
        t0 = time.time()
        down = [send(c, {"type": "train", "round": r, "state": blob}) for c, _ in conns]
        ups, rows = [], []
        for (c, h), db in zip(conns, down):
            msg, nbytes = recv(c)
            for u in msg["updates"]:
                ups.append((unpack(u["state"]), u["n"]))
            rows.append({"host": h["host"], "device": h["device"], "clients": [{k: v for k, v in u.items() if k != "state"} for u in msg["updates"]],
                         "device_compute_s": msg["device_compute_s"], "arrived_s": time.time() - t0, "up_bytes": nbytes,
                         "down_bytes": db, "peak_mem_mb": msg["peak_mem_mb"]})
        tot = sum(n for _, n in ups)
        state = {k: (sum(s_[k].float() * n for s_, n in ups) / tot).to(v.dtype) for k, v in state.items()}
        rec = {"round": r, "t_start": t0, "round_s": time.time() - t0, "clients_total": len(ups), "devices": rows}
        log["rounds"].append(rec)
        print(f"round {r}: {rec['round_s']:.1f} s  " + "  ".join(f"{x['host']}:{x['device_compute_s']:.1f}s" for x in rows), flush=True)
        if a.eval_every and ((r + 1) % a.eval_every == 0 or r + 1 == a.rounds):
            evaluate(r + 1)
        if a.time_budget and time.time() - t_begin > a.time_budget:
            print(f"time budget {a.time_budget} s reached after round {r}", flush=True)
            if a.eval_every and (r + 1) % a.eval_every:
                evaluate(r + 1)
            break
    log["federated_wall_s"] = time.time() - t_begin
    if a.method == "fedscout" and a.local_epochs > 0:
        if a.init_scout:  # local stage from a scout federated to R490 in simulation (realistic screening and cascade)
            sim = unpack(open(a.init_scout, "rb").read())
            state = {k.replace("base.0.", "backbone.", 1) if k.startswith("base.0.") else k: v for k, v in sim.items()}
            assert set(state) == set(shared_model("fedscout").state_dict()), "scout keys do not match"
            log["init_scout"] = a.init_scout
            print(f"local stage starts from {a.init_scout}", flush=True)
        t0 = time.time()
        for c, _ in conns:
            send(c, {"type": "local", "state": pack(state), "epochs": a.local_epochs, "cascade_n": a.cascade_n})
        log["local"] = []
        for c, h in conns:
            msg, _ = recv(c)
            for x in msg["clients"]:
                log["local"].append(dict(host=h["host"], device=h["device"], **x))
                print(f"local stage {h['host']} client {x['cid']} ({x['arch']}): {x['local_s']:.1f} s, peak {x['peak_mem_mb']:.0f} MB, "
                      f"acc {100 * x['cascade_acc']:.2f}, cascade p50 {x['cascade_p50_ms']:.1f} ms p95 {x['cascade_p95_ms']:.1f} ms", flush=True)
        log["local_wall_s"] = time.time() - t0
    for c, _ in conns:
        send(c, {"type": "done"})
    json.dump(log, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


# ---- client side: one process per device, hosting the device's clients ----
class VClient:
    def __init__(self, cid, arch, method, dev, batch):
        self.cid, self.arch, self.method, self.dev, self.B = cid, arch, method, dev, batch
        tr = np.load(f"client_{cid}_train.npz", allow_pickle=True)["data"].tolist()
        te = np.load(f"client_{cid}_test.npz", allow_pickle=True)["data"].tolist()
        self.X, self.Y = to_tensor(tr["x"], np.float32), to_tensor(tr["y"], np.int64)
        self.Xt, self.Yt = to_tensor(te["x"], np.float32), to_tensor(te["y"], np.int64)
        perm = torch.randperm(len(self.Y), generator=torch.Generator().manual_seed(cid))
        n_val = max(1, int(0.05 * len(self.Y)))
        self.val, self.train = perm[:n_val], perm[n_val:]
        torch.manual_seed(cid)
        self.learner = (Pool512(arch) if method == "fedapen" else Native(arch)).to(dev)
        self.w = torch.tensor([0.5], requires_grad=True)

    def batches(self, idx, shuffle=True):
        idx = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(idx), self.B):
            j = idx[i:i + self.B]
            if len(j) >= 2:
                yield self.X[j].to(self.dev), self.Y[j].to(self.dev)

    def train_round(self, blob):
        dev = self.dev
        t0 = time.time()
        shared = shared_model(self.method)
        shared.load_state_dict(unpack(blob))
        shared = shared.to(dev).train()
        if self.method == "fedscout":
            opt = torch.optim.SGD([{"params": shared.backbone.parameters(), "lr": 0.03},
                                   {"params": shared.head.parameters(), "lr": 0.24}], lr=0.03)
            for x, y in self.batches(self.train):
                opt.zero_grad(); F.cross_entropy(shared(x), y).backward(); opt.step()
        else:  # FedAPEN: ensemble weight on held-out (10 passes), then one epoch of mutual learning
            learner = self.learner
            learner.eval(); shared.eval()
            wopt = torch.optim.SGD([self.w], lr=1e-3)
            for _ in range(10):
                for x, y in self.batches(self.val, shuffle=False):
                    with torch.no_grad():
                        p, s_ = learner(x), shared(x)
                    wopt.zero_grad(); F.cross_entropy(self.w.to(dev) * p + (1 - self.w.to(dev)) * s_, y).backward(); wopt.step()
                    self.w.data.clamp_(0.0, 1.0)
            learner.train(); shared.train()
            wp = float(self.w)
            opt = torch.optim.SGD(learner.parameters(), lr=0.01)
            opt_g = torch.optim.SGD(shared.parameters(), lr=0.01)
            kl = nn.KLDivLoss(reduction="batchmean")
            for x, y in self.batches(self.train):
                out, out_g = learner(x), shared(x)
                loss = F.cross_entropy(out, y) + kl(F.log_softmax(out, 1), F.softmax(out_g.detach(), 1)) \
                    + F.cross_entropy(wp * out + (1 - wp) * out_g.detach(), y)
                loss_g = F.cross_entropy(out_g, y) + kl(F.log_softmax(out_g, 1), F.softmax(out.detach(), 1)) \
                    + F.cross_entropy(wp * out.detach() + (1 - wp) * out_g, y)
                opt.zero_grad(); opt_g.zero_grad(); (loss + loss_g).backward()
                nn.utils.clip_grad_norm_(learner.parameters(), 10); nn.utils.clip_grad_norm_(shared.parameters(), 10)
                opt.step(); opt_g.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return {"cid": self.cid, "arch": self.arch, "n": len(self.train), "state": pack(shared.state_dict()), "compute_s": time.time() - t0}

    def personalized_scout(self, blob):
        """scout with its head fine-tuned on my data for one epoch (frozen backbone): the model FedScout deploys"""
        scout = Native("c8")
        scout.load_state_dict(unpack(blob))
        scout = scout.to(self.dev)
        hopt = torch.optim.SGD(scout.head.parameters(), lr=0.24)
        scout.backbone.eval()
        for x, y in self.batches(self.train):
            with torch.no_grad():
                f = torch.flatten(scout.backbone(x), 1)
            hopt.zero_grad(); F.cross_entropy(scout.head(f), y).backward(); hopt.step()
        return scout.eval()

    def evaluate(self, blob):
        with torch.no_grad():
            if self.method == "fedscout":
                with torch.enable_grad():
                    model = self.personalized_scout(blob)
                pred = lambda x: model(x)
            else:
                shared = shared_model("fedapen")
                shared.load_state_dict(unpack(blob))
                shared = shared.to(self.dev).eval()
                self.learner.eval()
                wp = float(self.w)
                pred = lambda x: wp * self.learner(x) + (1 - wp) * shared(x)
            correct = 0
            for i in range(0, len(self.Yt), 256):
                correct += int((pred(self.Xt[i:i + 256].to(self.dev)).argmax(1).cpu() == self.Yt[i:i + 256]).sum())
        return {"cid": self.cid, "arch": self.arch, "n": len(self.Yt), "correct": correct}

    def local_stage(self, blob, epochs, cascade_n):
        dev = self.dev
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        scout = self.personalized_scout(blob)
        X, Y = self.X, self.Y
        with torch.no_grad():  # the frozen scout's logits are computed once and reused (the cache in clientscout.py)
            T = torch.cat([scout(X[self.train[i:i + 256]].to(dev)).float() for i in range(0, len(self.train), 256)])
        keep = T.softmax(1).gather(1, Y[self.train].to(dev)[:, None]).squeeze(1) < 0.5
        kidx, Tk = self.train[keep.cpu()], T[keep]
        pos = {int(v): k for k, v in enumerate(kidx)}
        learner = self.learner
        learner.train()
        lopt = torch.optim.SGD(learner.parameters(), lr=0.01)
        t1 = time.time()
        for _ in range(epochs):
            order = kidx[torch.randperm(len(kidx))]
            for i in range(0, len(order), self.B):
                j = order[i:i + self.B]
                if len(j) < 2:
                    continue
                x, y = X[j].to(dev), Y[j].to(dev)
                t = Tk[torch.tensor([pos[int(v)] for v in j], device=dev)]
                lopt.zero_grad(); F.cross_entropy(learner(x) + t, y).backward(); lopt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        learner_s, local_s, mem = time.time() - t1, time.time() - t0, peak_mem_mb(dev)
        learner.eval()
        lat, correct, runs = [], 0, 0
        n_eval = min(len(self.Yt), cascade_n)
        with torch.no_grad():
            for i in range(n_eval):
                x = self.Xt[i:i + 1].to(dev)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                st = time.perf_counter()
                z = scout(x)
                if float(z.softmax(1).max()) < 0.5:
                    z = z + 0.25 * learner(x)
                    runs += 1
                pred = int(z.argmax(1))
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                lat.append(1000 * (time.perf_counter() - st))
                correct += int(pred == int(self.Yt[i]))
        lat = np.array(lat[5:]) if len(lat) > 10 else np.array(lat)  # drop warm-up
        return {"cid": self.cid, "arch": self.arch, "local_s": local_s, "learner_s": learner_s, "kept_frac": float(keep.float().mean()),
                "peak_mem_mb": mem, "cascade_p50_ms": float(np.percentile(lat, 50)), "cascade_p95_ms": float(np.percentile(lat, 95)),
                "cascade_acc": correct / n_eval, "cascade_run_frac": runs / n_eval, "t_start": t0, "t_end": time.time()}


def client(a):
    dev = torch.device(a.device)
    cids, archs = [int(c) for c in a.cids.split(",")], a.archs.split(",")
    assert len(cids) == len(archs)
    vcs = [VClient(c, ar, a.method_hint, dev, a.batch) for c, ar in zip(cids, archs)]
    sock = socket.create_connection((a.host, a.port))
    send(sock, {"host": socket.gethostname(), "device": a.device,
                "clients": [{"cid": v.cid, "arch": v.arch, "n": len(v.train)} for v in vcs]})
    while True:
        msg, _ = recv(sock)
        if msg["type"] == "done":
            break
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if msg["type"] == "train":
            ups = [v.train_round(msg["state"]) for v in vcs]  # the device trains its clients one after another
            send(sock, {"updates": ups, "device_compute_s": time.time() - t0, "peak_mem_mb": peak_mem_mb(dev)})
        elif msg["type"] == "eval":
            send(sock, {"clients": [v.evaluate(msg["state"]) for v in vcs]})
        elif msg["type"] == "local":
            send(sock, {"clients": [v.local_stage(msg["state"], msg["epochs"], msg["cascade_n"]) for v in vcs]})
    sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["server", "client"])
    ap.add_argument("--method", choices=["fedscout", "fedapen"], default="fedscout")
    ap.add_argument("--method_hint", choices=["fedscout", "fedapen"], default="fedscout", help="client: which method")
    ap.add_argument("--devices", type=int, default=2, help="server: number of device connections to wait for")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=0, help="server: evaluate every k rounds (0 = never)")
    ap.add_argument("--time_budget", type=float, default=0, help="server: stop the federated phase after this many seconds")
    ap.add_argument("--cascade_n", type=int, default=300)
    ap.add_argument("--init_scout", default="", help="server: scout state for the local stage (saved with --scout_save_path)")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--host", default="192.168.0.11")
    ap.add_argument("--cids", default="0", help="client: comma-separated client ids hosted on this device")
    ap.add_argument("--archs", default="cnn", help="client: their private architectures (HtFE-img-8: arch of client i = models[i % 8])")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="fl_result.json")
    a = ap.parse_args()
    server(a) if a.role == "server" else client(a)
