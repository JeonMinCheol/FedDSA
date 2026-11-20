"""DomainNet with the 10 classes FedBN uses, 6 domains, 20 clients: real feature shift (external review, item 12).
- Pure feature shift: every domain is subsampled to equal per-class counts, and each domain's images are split IID
  among its clients, so every client has the same (uniform) class proportions.
- Domain of client i = perm[i] % 6 with a fixed random permutation, independent of the model assignment (HtFE: i % 8).
- Images: RGB, resized to 64 x 64, scaled to [-1, 1] like the other image sets; 75/25 train/test split per client.
usage: python generate_domainnet10.py   (reads DomainNet_raw/<domain>/<class>/*, writes DomainNet10_fs/)"""
import json
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CLS = ["bird", "feather", "headphones", "ice_cream", "teapot", "tiger", "whale", "windmill", "wine_glass", "zebra"]
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
NC, RAW, OUT = 20, "DomainNet_raw", "DomainNet10_fs"
rng = np.random.RandomState(1)


def load(path):
    img = Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0)


perm = rng.permutation(NC)
dom_of = {i: DOMAINS[perm[i] % len(DOMAINS)] for i in range(NC)}
clients = {d: [i for i in range(NC) if dom_of[i] == d] for d in DOMAINS}
xs, ys = {i: [] for i in range(NC)}, {i: [] for i in range(NC)}
for d in DOMAINS:
    files = {c: sorted(os.listdir(f"{RAW}/{d}/{c}")) for c in CLS}
    m = min(len(v) for v in files.values())  # equal per-class counts inside the domain
    print(d, "per-class counts", [len(files[c]) for c in CLS], "-> keep", m, "per class; clients", clients[d])
    for k, c in enumerate(CLS):
        pick = rng.permutation(len(files[c]))[:m]
        for j, idx in enumerate(pick):
            i = clients[d][j % len(clients[d])]  # IID split of each class among the domain's clients
            xs[i].append(load(f"{RAW}/{d}/{c}/{files[c][idx]}"))
            ys[i].append(k)
os.makedirs(f"{OUT}/train", exist_ok=True)
os.makedirs(f"{OUT}/test", exist_ok=True)
stat = []
for i in range(NC):
    x, y = np.stack(xs[i]), np.array(ys[i], dtype=np.int64)
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.75, shuffle=True, stratify=y, random_state=i)
    for part, (a, b) in (("train", (xtr, ytr)), ("test", (xte, yte))):
        with open(f"{OUT}/{part}/{i}.npz", "wb") as f:
            np.savez_compressed(f, data={"x": a, "y": b})
    stat.append([[int(c), int((y == c).sum())] for c in np.unique(y)])
    assert x.shape[1:] == (3, 64, 64) and np.abs(x).max() <= 1.0 + 1e-6
json.dump({"num_clients": NC, "num_classes": len(CLS), "non_iid": False, "balance": True, "partition": "domain",
           "classes": CLS, "domain_of_client": dom_of, "Size of samples for labels in clients": stat},
          open(f"{OUT}/config.json", "w"))
print("wrote", OUT, "samples per client", [len(ys[i]) for i in range(NC)])
