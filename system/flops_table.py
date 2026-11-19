"""Measured FLOPs per sample of every model the compared methods train, and local training-set sizes, per dataset
-> ../docs/figs/data/flops.json (read by comm_vs_acc.py --x compute). torch.utils.flop_counter counts the real ops
(FLOPs = 2 x MACs): <name> = forward (eval), <name>_train = forward + cross-entropy backward through autograd in train
mode, both per sample (batch of 4 divided by 4). HtFE-img-8: client i runs model i % 8; baselines use the pool projector, FedScout the linear one.
usage: python flops_table.py"""
import json
import types

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode, conv_flop_count

from flcore.trainmodel.models import BaseHeadSplit, CIFARResNet

MODELS = [  # main.py, model_family HtFE-img-8
    'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
    'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
    'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
    'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)']
DATA = {"Cifar100": (100, 32), "Flowers102": (102, 64), "TinyImagenet": (200, 64)}
NC = 20


B = 4


def conv_backward(grad_out_shape, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, _output_padding,
                  _groups, output_mask, out_shape=None, **kwargs):
    """torch's formula ignores groups and overcounts depthwise convs by the group count (MobileNetV2: 20x);
    each requested gradient (input, weight) costs one forward convolution, which conv_flop_count gets right."""
    return (int(output_mask[0]) + int(output_mask[1])) * conv_flop_count(x_shape, w_shape, grad_out_shape, transposed)


FIX = {torch.ops.aten.convolution_backward: conv_backward}


def flops(model, size):
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(B, 3, size, size))
    return int(fc.get_total_flops()) // B


def train_flops(model, size, C):
    model.train()
    with FlopCounterMode(display=False, custom_mapping=FIX) as fc:
        torch.nn.functional.cross_entropy(model(torch.randn(B, 3, size, size)), torch.randint(0, C, (B,))).backward()
    model.zero_grad(set_to_none=True)
    return int(fc.get_total_flops()) // B


def args_for(d, projector, models=MODELS):
    C, size = DATA[d]
    sp = ((size - 4) // 2 - 4) // 2  # main.fedavg_cnn_input_dim
    return types.SimpleNamespace(dataset=d, num_classes=C, feature_dim=512, fedavg_cnn_dim=64 * sp * sp,
                                 fedsa_projector=projector, algorithm="FedScout", models=models)


out = {}
for d, (C, size) in DATA.items():
    ns = [len(np.load(f"../dataset/{d}/train/{i}.npz", allow_pickle=True)["data"].tolist()["y"]) for i in range(NC)]
    row = {"n_train": ns}
    for proj in ("pool", "linear"):
        ms = [BaseHeadSplit(args_for(d, proj), cid) for cid in range(len(MODELS))]
        row[f"private_{proj}"] = [flops(m, size) for m in ms]
        row[f"private_{proj}_train"] = [train_flops(m, size, C) for m in ms]
    c8 = args_for(d, "linear", ['CIFARResNet(8, width=1.0, num_classes=args.num_classes)'])
    scout = BaseHeadSplit(c8, 0)
    row["scout"] = flops(scout, size)
    row["scout_train"] = train_flops(scout, size, C)
    row["scout_backbone"] = flops(scout.base[0], size)
    apen = CIFARResNet(8, num_classes=C)  # FedAPEN global model
    row["apen_shared_c8"], row["apen_shared_c8_train"] = flops(apen, size), train_flops(apen, size, C)
    moe = BaseHeadSplit(args_for(d, "pool", c8.models), 0)  # pFedMoE shared extractor (+ a head only to get a loss)
    row["pfedmoe_shared_c8"] = flops(moe.base, size)
    row["pfedmoe_shared_c8_train"] = train_flops(moe, size, C)  # shortcut: includes a 512xC head the extractor lacks (<1%)
    out[d] = row
    print(d, "n_train", sum(ns))
    for k, v in row.items():
        if k != "n_train":
            print(f"  {k:26s}", [f"{x / 1e6:.1f}" for x in v] if isinstance(v, list) else f"{v / 1e6:.1f}", "MFLOPs")
    assert all(2.0 < t / f < 3.2 for t, f in zip(row["private_pool_train"], row["private_pool"])), "train/forward ratio"
    print("  train/forward ratio, private (pool):", [f"{t / f:.2f}" for t, f in zip(row["private_pool_train"], row["private_pool"])],
          "scout:", f"{row['scout_train'] / row['scout']:.2f}")
json.dump(out, open("../docs/figs/data/flops.json", "w"), indent=1)
print("wrote ../docs/figs/data/flops.json")
