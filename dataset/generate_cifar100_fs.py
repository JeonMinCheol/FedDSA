"""Controlled feature shift on CIFAR-100 (analysis D): client i sees its images through domain i % 5, a fixed
label-independent transform, so p_k(x|y) differs across clients while the label split is the source's.
Domains on the stored [-1, 1] arrays: 0 original, 1 grayscale, 2 inverted, 3 rotated 90 deg, 4 pixelated (16x16).
usage: python generate_cifar100_fs.py iid     -> Cifar100_iid/ (IID labels) and Cifar100_iid_fs/
       python generate_cifar100_fs.py Cifar100 -> Cifar100_fs/ (Dir 0.1 labels + feature shift)"""
import json
import os
import sys

import numpy as np

DOMAINS = ("original", "gray", "invert", "rot90", "pixel2")


def shift(x, d):
    if d == 1:
        return np.repeat(x.mean(1, keepdims=True), 3, axis=1)
    if d == 2:
        return -x
    if d == 3:
        return np.ascontiguousarray(np.rot90(x, 1, axes=(2, 3)))
    if d == 4:
        return np.ascontiguousarray(x[:, :, ::2, ::2].repeat(2, axis=2).repeat(2, axis=3))
    return x


src = sys.argv[1]
if src == "iid":  # IID labels, same pipeline as the default generator
    np.random.seed(1)
    import generate_Cifar100 as g
    os.makedirs("Cifar100_iid", exist_ok=True)
    if not os.path.exists("Cifar100_iid/rawdata"):
        os.symlink(os.path.abspath("rawcifar100"), "Cifar100_iid/rawdata")
    g.generate_dataset("Cifar100_iid/", 20, False, True, None)
    src = "Cifar100_iid"
out = f"{src}_fs"
cfg = json.load(open(f"{src}/config.json"))
for part in ("train", "test"):
    os.makedirs(f"{out}/{part}", exist_ok=True)
    for i in range(cfg["num_clients"]):
        d = np.load(f"{src}/{part}/{i}.npz", allow_pickle=True)["data"].tolist()
        x = shift(d["x"], i % len(DOMAINS))
        assert x.shape == d["x"].shape and x.dtype == np.float32 and np.abs(x).max() <= 1.0 + 1e-6
        with open(f"{out}/{part}/{i}.npz", "wb") as f:
            np.savez_compressed(f, data={"x": x, "y": d["y"]})
cfg["feature_shift"] = {str(i): DOMAINS[i % len(DOMAINS)] for i in range(cfg["num_clients"])}
json.dump(cfg, open(f"{out}/config.json", "w"))
print("wrote", out)
