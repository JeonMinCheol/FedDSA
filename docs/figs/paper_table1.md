# Table 1. Training vs inference contribution of the private model (last evaluation, seed mean ± std)

| method / data | CE-only shared + prior | coupled shared alone | coupled shared + prior | method's own prediction (FedAPEN: learned ensemble; FML: private) | training effect (coupled − CE-only, shared + prior) | inference-rule effect (own prediction − coupled shared + prior) |
|---|---|---|---|---|---|---|
| FedAPEN / Cifar100 | running | 33.48 ± 0.58 (n=3) | 61.14 ± 0.36 (n=3) | 54.83 (n=3) | running | -6.31 |
| FedAPEN / TinyImagenet | running | 23.50 ± 0.16 (n=3) | 48.33 ± 0.33 (n=3) | 41.84 (n=3) | running | -6.49 |
| FML / Cifar100 (its prediction = private model) | running | running | running | running | running | running |

FedAPEN coupled rows: ariel runs (512 x C head, paper lr), R500. Tuned FedAPEN replaces them once the HP-fairness grid picks a configuration. FML uses the paper's alpha = beta = 0.5.
