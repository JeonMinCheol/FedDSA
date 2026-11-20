# Table 2. FedScout accuracy and training cost (last evaluation, seed mean ± std)

| method | Cifar100 | Flowers102 | TinyImagenet | training compute per client, CIFAR-100 (TFLOPs) |
|---|---|---|---|---|
| FedAPEN (paper lr; tuned version replaces it after the grid) | 54.83 ± 0.27 (n=3) | 59.74 ± 0.44 (n=3) | 41.84 ± 0.32 (n=3) | 545 |
| FedAPEN + prior, same checkpoint | 61.14 ± 0.36 (n=3) | running | 48.33 ± 0.33 (n=3) | 545 (same training) |
| scout alone (c8 FedAvg + head FT, no private model) | running | 68.71 ± 1.14 (n=3) | running | 78 |
| FedScout (490R + 50 local) | running | 69.62 ± 0.99 (n=3) | running | 78 + 13.1 (learner) |

Device line (estimate from per-step costs measured on the device): Raspberry Pi 5, CIFAR-100, per client: FedScout 3.6 h / 80 kJ vs FedAPEN 27.7 h / 888 kJ (wall power).
