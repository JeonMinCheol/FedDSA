"""CIFAR-100, 20 clients, Dirichlet label skew with a chosen alpha (the default generator fixes alpha = 0.1 in
utils/dataset_utils.py). Same pipeline as generate_Cifar100.py otherwise (train/test merged, 75/25 per client).
usage: python generate_cifar100_alpha.py 0.3   -> Cifar100_a0.3/      (numpy seed 1 for a reproducible split)"""
import os
import sys

import numpy as np

alpha = float(sys.argv[1])
np.random.seed(1)
import utils.dataset_utils as du  # noqa: E402

du.alpha = alpha  # separate_data / check / save_file read this module global
import generate_Cifar100 as g  # noqa: E402

out = f"Cifar100_a{alpha:g}/"
os.makedirs(out, exist_ok=True)
if not os.path.exists(out + "rawdata"):
    os.symlink(os.path.abspath("rawcifar100"), out + "rawdata")  # reuse the already downloaded CIFAR-100
g.generate_dataset(out, 20, True, False, "dir")
