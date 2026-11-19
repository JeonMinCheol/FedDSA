#!/usr/bin/env python
import torch
# 10+ runs share this node's 40 cores; PyTorch's default of one intra-op thread per core oversubscribes the CPU
torch.set_num_threads(int(__import__("os").environ.get("TORCH_NUM_THREADS", "4")))
import argparse
import os
import time
import warnings
import numpy as np
import logging

from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.servergen import FedGen
from flcore.servers.serverfd import FD
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.servergh import FedGH
from flcore.servers.servertgp import FedTGP
from flcore.servers.serverfeddsa import FedDSA
from flcore.servers.serverktl_stylegan_xl import FedKTL as FedKTL_stylegan_xl
from flcore.servers.serverktl_stylegan_3 import FedKTL as FedKTL_stylegan_3
from flcore.servers.serverktl_stable_diffusion import FedKTL as FedKTL_stable_diffusion
from flcore.servers.servermrl import FedMRL
from flcore.servers.serverapen import FedAPEN
from flcore.servers.serverpfedmoe import PFedMoE
from flcore.servers.serverscout import FedScout
from flcore.trainmodel.input_utils import image_input_shape
from flcore.servers.serversa import FedSA
from flcore.servers.serveretfproto import ETFProtoFL
from flcore.servers.serverfedrg import FedRG

try:
    from flcore.servers.serverl2g_feat import FedL2G as FedL2G_feat
    from flcore.servers.serverl2g_logit import FedL2G as FedL2G_logit
except ModuleNotFoundError as e:
    FedL2G_feat = None
    FedL2G_logit = None
    FedL2G_import_error = e
else:
    FedL2G_import_error = None

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def fedavg_cnn_input_dim(dataset):
    """Return the flattened dimension before FedAvgCNN's first FC layer."""
    image_size = 64 if str(dataset).lower() in {"tinyimagenet", "flowers102"} else 32
    spatial = (image_size - 5 + 1) // 2
    spatial = (spatial - 5 + 1) // 2
    return 64 * spatial * spatial


def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        args.fedavg_cnn_dim = fedavg_cnn_input_dim(args.dataset)
        print(f"FedAvgCNN input dimension: {args.fedavg_cnn_dim}")

        # Generate args.models
        if args.model_family == "HtFE-img-2":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-3":
            args.models = [
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-img-5":
            args.models = [
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "Homo-img-R18":  # homogeneous control: every client runs ResNet18
            args.models = ['torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)']

        elif args.model_family == "HtFE-img-lowend":
            # IoT-like tiers: 60% low-end (CNN, MobileNetV2), 40% larger ResNets; client cid gets models[cid % 10]
            cnn = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)'
            mob = 'mobilenet_v2(pretrained=False, num_classes=args.num_classes)'
            args.models = [cnn, cnn, cnn, mob, mob, mob,
                           'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                           'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                           'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                           'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)']

        elif args.model_family == "HtFE-img-9":
            args.models = [
                'resnet4(num_classes=args.num_classes)', 
                'resnet6(num_classes=args.num_classes)', 
                'resnet8(num_classes=args.num_classes)', 
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-8-HtC-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)'
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "Res34-HtC-img-4":
            args.models = [
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)'
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtM-img-10":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-txt-2":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-txt-4":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)'
            ]

        elif args.model_family == "HtFE-txt-5-1":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-2":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-3":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]
        
        elif args.model_family == "HtFE-txt-6":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)', 
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'GRUNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]

        elif args.model_family == "MLPs":
            args.models = [
                'AmazonMLP(feature_dim=[])', 
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "MLP_1layer":
            args.models = [
                'AmazonMLP(feature_dim=[200])', 
                'AmazonMLP(feature_dim=[500])', 
            ]

        elif args.model_family == "MLP_layers":
            args.models = [
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "HtFE-sen-het8":
            # IoT sensor tiers (sensor_models.py): MCU MLP / small CNN, mid deep CNN / GRU, edge DeepConvLSTM / ResNet / TCN / Transformer
            args.sen_shape = image_input_shape(args)
            args.models = [
                'SenMLP(args.sen_shape, args.num_classes)',
                'SenCNN(args.sen_shape, args.num_classes)',
                'SenCNN(args.sen_shape, args.num_classes, deep=True)',
                'DeepConvLSTM(args.sen_shape, args.num_classes)',
                'SenGRU(args.sen_shape, args.num_classes)',
                'WidthSenResNet(args.sen_shape, args.num_classes)',
                'SenTCN(args.sen_shape, args.num_classes)',
                'SenTransformer(args.sen_shape, args.num_classes)',
            ]

        elif args.model_family == "HtFE-sen-2":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            ]

        elif args.model_family == "HtFE-sen-3":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
            ]

        elif args.model_family == "HtFE-sen-5":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
            ]

        elif args.model_family == "HtFE-sen-8":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=2)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=3)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=2)',
            ]

        else:
            raise NotImplementedError

        if args.model_offset:  # client cid gets models[(cid + offset) % n]: decouples architecture from data split
            k = args.model_offset % len(args.models)
            args.models = args.models[k:] + args.models[:k]

        if args.global_arch:  # override the shared model of FML/FedKD-style baselines (default: family's global_model or models[0])
            args.global_model = {"cnn": 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.fedavg_cnn_dim)',
                                 "mnv2": 'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                                 "r18": 'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                                 "sres": 'WidthSenResNet(args.sen_shape, args.num_classes)',
                                 }.get(args.global_arch)
            if args.global_arch[0] == "c" and args.global_arch[1:].isdigit():  # c<depth>: same CIFAR ResNet as our twin
                args.global_model = f'CIFARResNet({int(args.global_arch[1:])}, num_classes=args.num_classes)'

        for model in args.models:
            print(model)

        if hasattr(args, 'global_model'):
            print('global_model:', args.global_model)

        if hasattr(args, 'heads'):
            for head in args.heads:
                print('head:', head)

        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedSA":
            server = FedSA(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "ETFProtoFL":
            server = ETFProtoFL(args, i)

        elif args.algorithm == "FedGen":
            server = FedGen(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedRG":
            server = FedRG(args, i)

        elif args.algorithm == "LG-FedAvg":
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedAPEN":
            server = FedAPEN(args, i)

        elif args.algorithm == "pFedMoE":
            server = PFedMoE(args, i)

        elif args.algorithm == "FedScout":
            server = FedScout(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedGH":
            server = FedGH(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)

        elif args.algorithm == "FedDSA":
            server = FedDSA(args, i)
            
        elif args.algorithm == "FedKTL-stylegan-xl":
            server = FedKTL_stylegan_xl(args, i)

        elif args.algorithm == "FedKTL-stylegan-3":
            server = FedKTL_stylegan_3(args, i)

        elif args.algorithm == "FedKTL-stable-diffusion":
            server = FedKTL_stable_diffusion(args, i)

        elif args.algorithm == "FedMRL":
            server = FedMRL(args, i)
            
        elif args.algorithm == "FedL2G-feat":
            if FedL2G_feat is None:
                raise FedL2G_import_error
            server = FedL2G_feat(args, i)

        elif args.algorithm == "FedL2G-logit":
            if FedL2G_logit is None:
                raise FedL2G_import_error
            server = FedL2G_logit(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    experimental = parser.add_argument_group(
        "Experimental settings (실험용)",
        "Ablation controls, diagnostics, and alternative anchor updates; not all are part of the default method.",
    )
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="HtM10")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-mfn', "--models_folder_name", type=str, default='',
                        help="The folder of pre-trained models")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--max_parallel_clients', type=int, default=1)
    parser.add_argument('--keep_client_models_on_gpu', type=str2bool, nargs='?', const=True, default=True,
                        help="FedDSA/FedSA: retain client models in VRAM between phases; false enables CPU offloading")
    parser.add_argument('--pin_memory', type=str2bool, nargs='?', const=True, default=True)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedProto
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-bt', "--beta", type=float, default=1.0)
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    # FedRG: uniform reproduces the official aggregator's target overwrite
    parser.add_argument('--rg_graph_target', choices=['uniform', 'reliability'], default='uniform')
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--adjalpha', type=float, default=3.0)
    parser.add_argument('--gc_epoch', type=int, default=10)
    parser.add_argument('--gnn_weight_decay', type=float, default=0.0001)
    parser.add_argument('--gnn_learning_rate', type=float, default=0.001)
    # FedDSA
    parser.add_argument('--use_amp', type=str2bool, nargs='?', const=True, default=False)
    experimental.add_argument('--anchor_codebook', type=str, default='gaussian',
                        choices=['gaussian', 'simplex'],
                        help="[EXPERIMENTAL] Initial anchor geometry; simplex is a fixed-codebook diagnostic")
    experimental.add_argument('--separate_anchor_roles', type=str2bool, nargs='?', const=True, default=True,
                        help="[EXPERIMENTAL] FedDSA: use semantic anchor for MSE and adaptive anchor for MCL in role-separated modes")
    parser.add_argument('--fast_anchor_proto_trust', choices=['off', 'auto'], default='auto',
                        help='Use adopted NextGap RAAR or disable server trust')
    parser.add_argument('--fast_anchor_radius_rho', type=float, default=0.6,
                        help="Prototype trust disagreement scale")
    experimental.add_argument('--fixed_anchor', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] FedDSA control: keep the initialized random anchor fixed and skip all server updates")
    experimental.add_argument('--semantic_gate_mode', type=str, default='off',
                        choices=['memory', 'meta', 'off'],
                        help="[EXPERIMENTAL] FedDSA: optional prototype-memory collection or meta-gradient update")
    experimental.add_argument('--new_anchor_mode', type=str, default='none',
                        choices=['none', 'relation_memory', 'semantic_discriminative',
                                 'observation_validation', 'meta_separated', 'boundary',
                                 'class_boundary'],
                        help="[EXPERIMENTAL] FedDSA role-separated anchor experiment")
    experimental.add_argument('--new_anchor_ema', type=float, default=0.25,
                        help="[EXPERIMENTAL] EMA step for the recent-observation semantic anchor")
    experimental.add_argument('--relation_memory_lr', type=float, default=0.25,
                        help="[EXPERIMENTAL] Pairwise relation-memory update rate")
    experimental.add_argument('--relation_anchor_lr', type=float, default=0.02,
                        help="[EXPERIMENTAL] Geometry-anchor pairwise fitting step")
    experimental.add_argument('--relation_observation_weight', type=float, default=0.1,
                        help="[EXPERIMENTAL] Weight of observed semantic positions in relation memory")
    experimental.add_argument('--relation_step_scale', type=float, default=0.25,
                        help="[EXPERIMENTAL] Maximum relation-anchor movement as a nearest-neighbor fraction")
    experimental.add_argument('--discriminative_anchor_lr', type=float, default=0.05,
                        help="[EXPERIMENTAL] Discriminative-anchor prototype CE step")
    experimental.add_argument('--discriminative_cosine_scale', '--shared_classifier_scale',
                        dest='discriminative_cosine_scale', type=float, default=5.0,
                        help="[EXPERIMENTAL] Semantic-discriminative anchor cosine objective scale; never applied to the linear classifier")
    experimental.add_argument('--discriminative_anchor_reg', type=float, default=0.01,
                        help="[EXPERIMENTAL] Discriminative-anchor stability regularizer")
    experimental.add_argument('--discriminative_step_scale', type=float, default=0.25,
                        help="[EXPERIMENTAL] Maximum discriminative-anchor movement")
    experimental.add_argument('--discriminative_codebook_reg', type=float, default=0.0,
                        help="[EXPERIMENTAL] Optional regularizer keeping the adaptive target near its initial codebook")
    experimental.add_argument('--freeze_semantic_anchor', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] Keep the semantic/MSE target at its initial codebook in explicit role modes")
    experimental.add_argument('--reliability_anchor_source', type=str, default='mcl',
                        choices=['mcl', 'semantic'],
                        help="[EXPERIMENTAL] Anchor used for classifier/agreement reliability weighting")
    experimental.add_argument('--scalar_reliability_state', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] Maintain a class-wise evidence state for role-separated target interpolation")
    experimental.add_argument('--reliability_weighted_aggregation', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] Weight server prototype aggregation by cross-client consistency")
    experimental.add_argument('--reliability_aggregation_temperature', type=float, default=1.0,
                        help="[EXPERIMENTAL] Temperature for consistency-based prototype aggregation weights")
    experimental.add_argument('--reliability_aggregation_geometry', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="[EXPERIMENTAL] Geometry used to score cross-client prototype consistency")
    parser.add_argument('--utility_inner_lr', type=float, default=0.01,
                        help="Virtual projector/classifier step size for local CAU and RAAR evidence probes")
    parser.add_argument('--local_marginal_cau', type=str2bool, nargs='?', const=True, default=False,
                        help="Enable private current-state local CAU before SGD; false disables it; no server EMA or temperature")
    parser.add_argument('--local_marginal_gap', type=str2bool, nargs='?', const=True, default=True,
                        help="Include MCL gap in the local marginal utility probe; actual local/RAAR MCL gaps are unaffected")
    parser.add_argument('--raar_evidence_assignment', choices=['observed', 'shuffled', 'uniform', 'geometry'], default='observed',
                        help='Server-only RAAR evidence assignment control; never changes local CAU gates')
    parser.add_argument('--raar_evidence_ema', '--utility_ema', dest='raar_evidence_ema', type=float, default=0.25,
                        help="EMA rate for RAAR-only raw loss-gain evidence; never broadcast as local CAU weights")
    experimental.add_argument('--utility_validation_probe', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] Measure independent single-anchor intervention gain for utility validation")
    experimental.add_argument('--reliability_state_alpha', type=float, default=0.25,
                        help="[EXPERIMENTAL] EMA rate for the class-wise evidence state")
    experimental.add_argument('--reliability_staleness_tau', type=float, default=20.0,
                        help="[EXPERIMENTAL] Staleness time constant for the class-wise evidence state")
    experimental.add_argument('--discriminative_metric', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="[EXPERIMENTAL] Similarity used by the server discriminative-anchor update")
    experimental.add_argument('--validation_anchor_lr', type=float, default=0.05,
                        help="[EXPERIMENTAL] Train-only query validation-anchor step")
    experimental.add_argument('--validation_step_scale', type=float, default=0.25,
                        help="[EXPERIMENTAL] Maximum validation-anchor movement")
    experimental.add_argument('--meta_anchor_lr', type=float, default=1000.0,
                        help="[EXPERIMENTAL] FedDSA anchor-space meta-gradient step size")
    experimental.add_argument('--meta_inner_lr', type=float, default=0.01,
                        help="[EXPERIMENTAL] FedDSA virtual projector/classifier update step for meta-gradient")
    experimental.add_argument('--meta_step_scale', type=float, default=0.25,
                        help="[EXPERIMENTAL] Maximum meta-learned adaptive-anchor movement as a nearest-neighbor fraction")
    experimental.add_argument('--boundary_anchor_ema', type=float, default=0.001,
                        help="[EXPERIMENTAL] FedDSA boundary mode EMA for the hardest rival prototype")
    experimental.add_argument('--boundary_loss_weight', type=float, default=1.0,
                        help="[EXPERIMENTAL] FedDSA boundary mode weight for hard-rival separation")
    experimental.add_argument('--class_boundary_ema', type=float, default=0.1,
                        help="[EXPERIMENTAL] FedDSA class-boundary mode EMA for per-class margin targets")
    experimental.add_argument('--class_boundary_min', type=float, default=0.1,
                        help="[EXPERIMENTAL] FedDSA minimum class-conditioned boundary margin")
    experimental.add_argument('--class_boundary_max', type=float, default=1.5,
                        help="[EXPERIMENTAL] FedDSA maximum class-conditioned boundary margin")
    experimental.add_argument('--memory_max_per_class', type=int, default=2,
                        help="[EXPERIMENTAL] FedDSA trusted prototype memory slots per class")
    experimental.add_argument('--memory_min_support', type=float, default=1.0,
                        help="[EXPERIMENTAL] FedDSA minimum client support for a remembered prototype")
    experimental.add_argument('--memory_confidence_floor', type=float, default=0.7,
                        help="[EXPERIMENTAL] FedDSA minimum prototype-to-anchor class confidence for memory")
    experimental.add_argument('--memory_temperature', type=float, default=1.0,
                        help="[EXPERIMENTAL] FedDSA prototype classification temperature for memory consistency")
    parser.add_argument('--agreement_weighting', '--agrement_weighting',
                        dest='agreement_weighting', type=str2bool, nargs='?', const=True, default=True,
                        help="FedDSA: enable anchor/head agreement sample weighting")
    parser.add_argument('--pcw_prediction_source', choices=['classifier_logits', 'cosine'],
                        default='classifier_logits',
                        help="PCW correctness from actual linear logits or cosine with classifier weight directions; classification stays linear")
    parser.add_argument('--anchor_momentum', '--stable_ema', '--ema',
                        dest='anchor_momentum', type=float, default=0.9999,
                        help="FedDSA: stable anchor EMA momentum")
    parser.add_argument('--anchor_candidate_mix', type=float, default=None,
                        help='New prototype fraction gamma; defaults to 0.5')
    experimental.add_argument('--moving_anchor_ema', type=float, default=None,
                        help='Compatibility alias for anchor_candidate_mix, the new prototype fraction')
    parser.add_argument('--disable_prototype_trust', type=str2bool, nargs='?', const=True, default=False,
                        help='Disable RAAR while keeping the same EMA candidate')
    experimental.add_argument('--raar_diagnostics', type=str2bool, nargs='?', const=True, default=False,
                        help='Optional B4 train query; never used to select RAAR updates')
    experimental.add_argument('--fast_anchor_ema', type=float, default=1.0,
                        help="[EXPERIMENTAL] FedDSA dual gate: EMA step for the adaptive legacy anchor")
    experimental.add_argument('--dual_disagreement_gate', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] FedDSA: use stable/adaptive disagreement to select adaptive MCL supervision")
    experimental.add_argument('--dual_disagreement_temperature', type=float, default=0.05,
                        help="[EXPERIMENTAL] FedDSA dual gate: soft margin-difference temperature")
    # FedKTL
    parser.add_argument('-GPath', "--generator_path", type=str, default='stylegan/stylegan-xl-models/imagenet64.pkl')
    parser.add_argument('-prompt', "--stable_diffusion_prompt", type=str, default='a cat')
    parser.add_argument('-sbs', "--server_batch_size", type=int, default=100)
    parser.add_argument('-gbs', "--gen_batch_size", type=int, default=4,
                        help="Not related to the performance. A small value saves GPU memory.")
    parser.add_argument('-mu', "--mu", type=float, default=50.0)
    # FedMRL
    parser.add_argument('-sfd', "--sub_feature_dim", type=int, default=128)
    # FedL2G
    parser.add_argument('-mdb', "--meta_study_batches", type=int, default=1)
    parser.add_argument('-mqb', "--meta_quiz_batches", type=int, default=1)
    parser.add_argument('-wu', "--warm_up_rounds", type=int, default=50)
    # FedSA
    parser.add_argument('--fedsa_anchor_momentum', type=float, default=0.9999,
                        help="FedSA semantic-anchor EMA momentum")
    parser.add_argument('--fedsa_anchor_codebook', type=str, default='gaussian', choices=['gaussian', 'simplex'],
                        help="FedSA initial anchors: Gaussian through a random MLP (paper) or simplex ETF (same mean norm)")
    parser.add_argument('--fedsa_align_detach', type=str2bool, nargs='?', const=True, default=False,
                        help="FedSA: anchor losses update only the projector (needs --projector linear/ortholearn); backbone trained by CE only")
    # ETF-ProtoFL (ESWA 2027): projector to d=C, fixed simplex-ETF classifier, AMCL; use -lam 0.1 for CIFAR
    parser.add_argument('--etf_margin', type=float, default=0.5, help="ETF-ProtoFL AMCL additive cosine margin m")
    parser.add_argument('--etf_tau', type=float, default=0.07, help="ETF-ProtoFL AMCL scaling factor tau")
    parser.add_argument('--etf_adaptive_margin', type=str, default='off', choices=['off', 'min', 'min_centered'],
                        help="ours: off = fixed m (paper); min = min(m, client's local angular prototype gap)")
    parser.add_argument('--etf_agg_weight', type=str, default='count', choices=['count', 'gap'],
                        help="ours: prototype aggregation weight = sample count (paper) or count x client's centred local angular gap")
    parser.add_argument('--etf_exclude_clients', type=str, default='',
                        help="oracle diagnostic: comma-separated client ids left out of prototype aggregation")
    parser.add_argument('--etf_logit_scale', type=float, default=0.0,
                        help="ETF-ProtoFL CE logit scale (not in the paper); <=0 uses 1/etf_tau")
    parser.add_argument('--fedsa_eval', type=str, default='proto', choices=['proto', 'head'],
                        help="FedSA test prediction: nearest global prototype (paper) or local linear head (FedDSA)")
    parser.add_argument('--projector', '--fedsa_projector', dest='fedsa_projector', type=str, default='pool', choices=['pool', 'linear', 'orthofixed', 'orthoaffine', 'ortholearn', 'dimaware', 'pool_linear'],
                        help="BaseHeadSplit feature map for all baselines: AdaptiveAvgPool1d (default) or trainable Linear projector (FedDSA)")
    parser.add_argument('--fedsa_augment', type=str2bool, nargs='?', const=True, default=False,
                        help="standard augmentation (random crop + flip) on each training batch; FedSA, FedProto, FedTGP and the other baselines (Local, FedGH, LG-FedAvg, FD, FML, FedKD, FedMRL, FedGen, FedL2G)")
    parser.add_argument('--fedsa_bottleneck', type=str, default='off', choices=['off', 'fixed', 'fed'],
                        help="ours: stochastic bottleneck in the anchor space; KL to N(anchor_y, s^2) with s^2 = federated per-class variance (fed) or one scalar (fixed, VIB-style control)")
    parser.add_argument('--fedsa_bottleneck_beta', type=float, default=1.0, help="weight of the bottleneck KL term")
    parser.add_argument('--fedsa_peer_head', type=float, default=0.0,
                        help="ours: weight of CE under a random other client's head (heads share the fixed anchor space); 0 = off")
    parser.add_argument('--fedsa_common_model', type=str2bool, nargs='?', const=True, default=False,
                        help="ours: every client also trains a shared small CNN (FedAvg'd by all) and mutually distills with it")
    parser.add_argument('--fedsa_deep_anchor', type=float, default=0.0,
                        help="ours: weight of the fixed-anchor loss on intermediate backbone stages (~50%%/75%% depth, GAP+linear to anchor dim); 0 = off")
    parser.add_argument('--oracle_mash', type=int, default=1, help="with --oracle_pool: share means of k same-class images per client (FedMix-style baseline); 1 = raw images (oracle)")
    parser.add_argument('--oracle_pool', type=float, default=0.0, help="ceiling test only: add other clients' real samples of local classes (x local size)")
    parser.add_argument('--fedsa_kernel_bank', type=int, default=0, help="FedSA: M>0 builds every backbone conv from M shared bank kernels per kernel size (bank FedAvg'd across architectures)")
    parser.add_argument('--fedsa_bank_chunk', type=int, default=16, help="FedSA kernel bank: chunk size B (B x B x k x k)")
    parser.add_argument('--fedsa_bank_private', type=str2bool, nargs='?', const=True, default=False, help="FedSA kernel bank control: never average the bank (capacity-only effect)")
    parser.add_argument('--global_arch', type=str, default="", choices=["", "cnn", "mnv2", "r18", "sres", "c8", "c14", "c20", "c56"], help="shared/global model for FML, FedKD, FedAPEN, FedMRL (c<depth> = CIFAR ResNet, match the twin)")
    parser.add_argument('--fedsa_common_arch', type=str, default="cnn", choices=["cnn", "bn", "r18", "mnv2", "gnet", "sres", "c8", "c14", "c20", "c32", "c56"], help="FedSA common model: cnn = FedAvgCNN, bn = SmallBNCNN (inversion), r18 = ResNet18 averaged twin, mnv2 = MobileNetV2, gnet = GoogLeNet, c<depth> = CIFAR ResNet-depth")
    parser.add_argument('--fedsa_cm_width', type=float, default=1.0, help="channel multiplier of a c<depth> twin (twin-size study; ignored with --fedsa_cm_widths)")
    parser.add_argument('--fedsa_cm_private', type=str2bool, nargs='?', const=True, default=False, help="control: never average the common model (inversion then carries only local knowledge)")
    parser.add_argument('--fedsa_cm_invert', type=float, default=0.0, help="FedSA: weight of private-model loss on inputs synthesized by inverting the FedAvg'd common model")
    parser.add_argument('--fedsa_inv_n', type=int, default=256, help="synthetic inputs per client per round")
    parser.add_argument('--fedsa_inv_steps', type=int, default=30, help="inversion optimization steps")
    parser.add_argument('--fedsa_inv_bn', type=float, default=1.0, help="weight of BN-statistics matching in inversion")
    parser.add_argument('--fedsa_inv_start', type=int, default=20, help="first round that uses synthetic inputs (the averaged model is weak before)")
    parser.add_argument('--fedsa_peer_invert', type=float, default=0.0, help="FedSA (A): weight of loss on other clients' backbone-inversion images of my classes")
    parser.add_argument('--fedsa_peer_invert_local', type=str2bool, nargs='?', const=True, default=False, help="control for --fedsa_peer_invert: use only my own backbone's inversions (self-distillation)")
    parser.add_argument('--fedsa_inv_recv', type=int, default=512, help="max synthetic images a client receives per round")
    parser.add_argument('--fedsa_inv_anchor', type=str2bool, nargs='?', const=True, default=False, help="anchor protocol: invert toward the fixed anchors and distill the producer's anchor-space representation (private anchors should break it)")
    parser.add_argument('--fedsa_shrink_perturb', type=float, default=1.0, help="control: <1 shrinks the backbone by this factor every round (local reset effect, no federation)")
    parser.add_argument('--fedsa_sp_sigma', type=float, default=0.1, help="shrink-perturb noise, relative to each tensor's std")
    parser.add_argument('--fedsa_shape_avg', type=str2bool, nargs='?', const=True, default=False, help="FedAvg every backbone tensor whose name and shape match across >= 2 clients (cross-architecture partial averaging)")
    parser.add_argument('--fedsa_memref', type=float, default=0.0, help="weight of extra label smoothing on classes I learn faster than the reference (federated memorization reference)")
    parser.add_argument('--fedsa_memref_mode', type=str, default="fed", choices=["fed", "local"], help="memref reference: fed = federation's per-class loss, local = my own mean loss (control)")
    parser.add_argument('--fedsa_inv_deep', type=str2bool, nargs='?', const=True, default=False, help="with --fedsa_inv_anchor and --fedsa_deep_anchor: also exchange per-stage features in the shared stage coordinates")
    parser.add_argument('--fedsa_inv_jitter', type=int, default=0, help="inversion: random shift (px) + flip at every step (0 = off)")
    parser.add_argument('--fedsa_inv_first_bn', type=float, default=1.0, help="inversion: multiplier of the first BN layer's statistic loss")
    parser.add_argument('--fedsa_inv_conf', type=float, default=0.0, help="inversion: send only images the producer assigns to their class with prob >= this")
    parser.add_argument('--fedsa_inv_bank', type=int, default=0, help="receiver keeps the newest N received images across rounds (0 = replace every round)")
    parser.add_argument('--fedsa_inv_relay', type=int, default=0, help="relay inversion: refine received images this many steps under my backbone before passing them on (0 = off)")
    parser.add_argument('--fedsa_inv_relay_n', type=int, default=128, help="relay inversion: how many received images to refine and pass on per round")
    parser.add_argument('--fedsa_inv_every', type=int, default=1, help="synthesize and exchange images every k rounds (speed)")
    parser.add_argument('--fedsa_syn_frac', type=float, default=0.5, help="synthetic images per training step as a fraction of the batch (speed)")
    parser.add_argument('--fedsa_faa', type=float, default=0.0, help="weight of the backbone loss on the federated adversarial augmenter's views (0 = off)")
    parser.add_argument('--fedsa_faa_eps', type=float, default=0.25, help="augmenter perturbation budget (normalized input units)")
    parser.add_argument('--fedsa_faa_private', type=str2bool, nargs='?', const=True, default=False, help="control: never average the augmenter")
    parser.add_argument('--fedsa_faa_lr_mult', type=float, default=10.0, help="augmenter learning-rate multiplier (via gradient scaling)")
    parser.add_argument('--fedsa_mix', type=str, default="off", choices=["off", "mixup", "cutmix"], help="strong-augmentation baseline on top of --fedsa_augment")
    parser.add_argument('--fedsa_mix_prob', type=float, default=0.5, help="probability of mixing a batch")
    parser.add_argument('--fedsa_inv_fast', type=str2bool, nargs='?', const=True, default=False, help="relay: optimize fresh + received images together for --fedsa_inv_relay steps (one pass)")
    parser.add_argument('--fedsa_inv_train', type=float, default=0.0, help="training-time inversion: sign step size applied to the pooled synthetic images from my training loss's input gradient (0 = off)")
    parser.add_argument('--fedsa_faa_mode', type=str, default="pixel", choices=["pixel", "semantic"], help="augmenter: pixel = bounded per-pixel delta; semantic = low-frequency delta + per-channel contrast/brightness")
    parser.add_argument('--fedsa_syn_filter', type=int, default=0, help="train on a received image only if my model ranks its label in the top-k (0 = off)")
    parser.add_argument('--fedsa_inv_relay_oldest', type=str2bool, nargs='?', const=True, default=False, help="relay the oldest held images first (they accumulate refinement across hops)")
    parser.add_argument('--fedsa_inv_chunk', type=int, default=128, help="images optimized together per inversion pass (time scales with passes, not images)")
    parser.add_argument('--fedsa_sp_weights_only', type=str2bool, nargs='?', const=True, default=False, help="shrink-perturb only conv/linear weights (not BN/biases)")
    parser.add_argument('--fedsa_vgen', type=float, default=0.0, help="weight of the backbone loss on federated variation-generator views (0 = off)")
    parser.add_argument('--fedsa_vgen_private', type=str2bool, nargs='?', const=True, default=False, help="control: never average the variation generator")
    parser.add_argument('--fedsa_vgen_scale', type=float, default=1.0, help="x' = x + scale * (dec(z_sample) - dec(mu))")
    parser.add_argument('--fedsa_vgen_temp', type=float, default=1.0, help="latent sampling temperature")
    parser.add_argument('--fedsa_vgen_start', type=int, default=10, help="first round whose backbone uses the variations (generator trains from round 0)")
    parser.add_argument('--fedsa_cm_base_only', type=str2bool, nargs='?', const=True, default=False, help="average only the common model's backbone (personal projector/head), as in the homogeneous FedAvg control")
    parser.add_argument('--fedsa_vgen_mode', type=str, default="local", choices=["local", "global", "transport", "noise"], help="variation generator: local = around my encoding; global = toward a sample of the pooled class latent; transport = move my class latent distribution onto the pooled one; noise = smooth random-noise control (no generator use)")
    parser.add_argument('--fedsa_vgen_mix', type=float, default=0.5, help="global mode: max interpolation toward the class latent sample")
    parser.add_argument('--fedsa_res_standalone', type=str2bool, nargs='?', const=True, default=False, help="residual ablation: private also trained alone (CE + KL to the base), FedAPEN-style")
    parser.add_argument('--fedsa_priv_anchor', type=float, default=1.0, help="scale of the anchor loss on the private model (0 = private without anchors)")
    parser.add_argument('--fedsa_cm_anchor', type=float, default=1.0, help="scale of the anchor loss on the shared twin (0 = twin without anchors)")
    parser.add_argument('--fedsa_res_readout', type=float, default=0.0, help="residual: weight of CE(received twin head(private features), y)")
    parser.add_argument('--fedsa_res_feat', type=float, default=0.0, help="residual: weight of 1 - cos(private features, received twin features)")
    parser.add_argument('--fedsa_res_ce', type=float, default=0.0, help="residual: weight of a plain standalone CE on the private logits")
    parser.add_argument('--fedsa_res_live', type=str2bool, nargs='?', const=True, default=False, help="residual ablation: sum with the locally-updating twin instead of the frozen received copy")
    parser.add_argument('--fedsa_cm_widths', type=str, default="", help="nested twin: comma-separated width per model index (cid %% len(models)), e.g. 0.25,0.5,0.25,0.5,1,1,1,1; uses WidthResNet18")
    parser.add_argument('--fedsa_cm_wd', type=float, default=0.0, help="weight decay of the common model (twin)")
    parser.add_argument('--fedsa_cm_val', type=float, default=0.0, help="held-out fraction of each client's training data for the twin memorization gate")
    parser.add_argument('--fedsa_cm_gate', type=int, default=0, help="memorization gate: freeze the twin backbone at its best held-out round after this many rounds without improvement (0 = off; needs --fedsa_cm_val > 0)")
    parser.add_argument('--fedsa_residual', type=str2bool, nargs='?', const=True, default=False, help="private model learns the residual of the received common model (logit sum)")
    parser.add_argument('--fedsa_kd_mix', type=float, default=0.0, help="with a common model: (1-a)*CE + a*KD for the private model (a>0 also sets the KD weight to a)")
    parser.add_argument('--fedsa_vgen_replace', type=float, default=0.0, help="variation generator: replace this fraction of each batch by variations (no extra backbone pass, no extra loss term); 0 = extra loss mode")
    parser.add_argument('--fedsa_vgen_noise_mag', type=float, default=0.1, help="noise control: mean |delta| of the smooth perturbation")
    parser.add_argument('--fedsa_pub_kd', type=float, default=0.0, help="weight of edge-ensemble distillation on unlabeled public images (0 = off)")
    parser.add_argument('--fedsa_pub_path', type=str, default="../dataset/public/tiny32_10k.npz", help="unlabeled public pool (x only)")
    parser.add_argument('--fedsa_pub_n', type=int, default=512, help="public images per round")
    parser.add_argument('--fedsa_pub_frac', type=float, default=0.5, help="public images per training step, as a fraction of the batch")
    parser.add_argument('--fedsa_pub_start', type=int, default=10, help="first round with public targets")
    parser.add_argument('--fedsa_edge_num', type=int, default=4, help="number of edge gateways (clients grouped by id)")
    parser.add_argument('--fedsa_pub_target', type=str, default="edge", choices=["edge", "global", "self"], help="ensemble target: my edge / all edges / my own outputs (control)")
    parser.add_argument('--fedsa_mem_probe', type=str2bool, nargs='?', const=True, default=False, help="at eval rounds: membership AUC of private confidence vs private-twin gap (memorization probe)")
    parser.add_argument('--fedsa_pub_space', type=str, default="anchor", choices=["anchor", "logit"], help="edge ensemble space: fixed anchor-space representations (ours) or head logits (FedDF/DS-FL-style baseline)")
    parser.add_argument('--fedsa_mem_dump', type=str, default="", help="comma-separated eval rounds at which each client saves per-sample probe scores for its whole train set (shard order) to logs/memdump/<goal>/")
    parser.add_argument('--fedsa_pub_source', type=str, default="public", choices=["public", "peer", "peer_self"], help="ensemble inputs: public pool at the edge / peer models on my own images (no public data) / my own model on my images (control)")
    parser.add_argument('--fedsa_private_anchors', type=str2bool, nargs='?', const=True, default=False,
                        help="control: each client gets its own random rotation of the anchors (no shared coordinate system)")
    parser.add_argument('--model_offset', type=int, default=0,
                        help="rotate the model list so client cid gets models[(cid+offset) %% n] (deconfound architecture vs client data)")
    parser.add_argument('--fedsa_kd_frozen', type=str2bool, nargs='?', const=True, default=False,
                        help="private model distills from the received common model frozen (not adapted to local data); common copy still learns from the private")
    parser.add_argument('--fedsa_kd_weight', type=float, default=1.0, help="weight of mutual distillation with the shared model")
    parser.add_argument('--fedsa_kd_feat', type=str2bool, nargs='?', const=True, default=True,
                        help="also distill in the shared anchor space (feature MSE), not only logits")
    parser.add_argument('--fedsa_group_avg', type=str2bool, nargs='?', const=True, default=False,
                        help="ours: FedAvg of whole models among clients with the same architecture (cohort averaging)")
    parser.add_argument('--fedsa_weight_decay', type=float, default=0.0, help="FedSA local SGD weight decay")
    parser.add_argument('--fedsa_head_avg', type=str2bool, nargs='?', const=True, default=False,
                        help="ours: server FedAvg of the 512->C heads (all clients share the fixed anchor space)")
    parser.add_argument('--fedsa_proto_soft', type=float, default=0.0,
                        help="ours: weight beta of the global-prototype soft target mixed into the CE label (0 = hard labels)")
    parser.add_argument('--fedsa_proto_soft_temp', type=float, default=0.1,
                        help="temperature (relative to mean prototype distance) for the prototype soft target")
    parser.add_argument('--fedsa_backbone_view', type=float, default=0.0,
                        help="ours: view = noise on the pre-projector backbone feature (norm = scale x ||feat||), bound to anchors + KL; 0 = off")
    parser.add_argument('--fedsa_group_alpha', type=float, default=1.0,
                        help="with --fedsa_group_avg: <1 = soft pull of the backbone (base.0.*) toward the same-architecture cohort mean; 1 = full model copy")
    parser.add_argument('--fedsa_relation', type=float, default=0.0,
                        help="ours (anchor-free): relational consistency weight; codes = softmax cosine to own class prototypes, matched to the federation's per-class relational structure + view consistency")
    parser.add_argument('--fedsa_relation_view', type=float, default=0.3, help="relational view noise norm relative to ||z||")
    parser.add_argument('--fedsa_shift_fixed', type=float, default=0.0,
                        help="view norm = c * nearest-anchor distance (constant over training); 0 = off")
    parser.add_argument('--fedsa_shift_norm', type=str2bool, nargs='?', const=True, default=False,
                        help="proto_shift view: keep the cross-client deviation direction but rescale it to the energy of the matching isotropic noise")
    parser.add_argument('--fedsa_stat_view_scale', type=float, default=1.0,
                        help="variance multiplier for the feature-space second view (stat/noise)")
    parser.add_argument('--fedsa_second_view', type=str, default='off', choices=['off', 'ce', 'anchor', 'consist', 'proto', 'stat', 'noise', 'local', 'stat_ce', 'proto_shift', 'noise_kl', 'noise_fixed'],
                        help="FedSA: second augmented view; ce = extra CE only (control), anchor = ours (both views to the same anchor + view consistency)")
    parser.add_argument('--fedsa_anchor_mixup', type=float, default=0.0,
                        help="ours: mixup of features with shared anchors of other classes; Beta(a, a) parameter, 0 = off")
    parser.add_argument('--fedsa_gap_warmup', type=int, default=0,
                        help="FedSA: first K local rounds use max(local, global) gap before --fedsa_gap_mode applies")
    parser.add_argument('--fedsa_proto_pass', type=str, default='train', choices=['train', 'eval'],
                        help="FedSA prototypes: accumulated during training (paper) or recollected after training in eval mode (FedDSA)")
    parser.add_argument('--fedsa_gap_mode', type=str, default='max', choices=['max', 'min'],
                        help="FedSA MCL gap: max(local, global) (paper) or min (FedDSA)")
    parser.add_argument('--fedsa_res_sum_loss', type=str2bool, nargs='?', const=True, default=True,
                        help="FedSA residual: train the private model on CE(private + frozen twin) (True) or on its own CE (False; sum only at test)")
    parser.add_argument('--fedsa_res_kl', type=float, default=1.0,
                        help="FedSA residual standalone term: weight of KL(private || frozen twin); 0 = no distillation")
    parser.add_argument('--pfedmoe_gate_lr', type=float, default=0.01, help="pFedMoE: learning rate of the local gating network (paper grid 0.001-1)")
    # FedScout (defaults = the final configuration)
    parser.add_argument('--scout_arch', type=str, default="c20", help="FedScout: scout architecture, c<depth> = CIFAR ResNet")
    parser.add_argument('--scout_width', type=float, default=1.0, help="FedScout: channel multiplier of the scout")
    parser.add_argument('--scout_lr_scale', type=float, default=3.0, help="FedScout: scout lr = local lr x this")
    parser.add_argument('--scout_wd', type=float, default=0.0, help="FedScout: scout weight decay")
    parser.add_argument('--scout_val', type=float, default=0.05, help="FedScout: held-out fraction of local training data (gate, alpha)")
    parser.add_argument('--scout_share', type=str, default="backbone", choices=["backbone", "all", "none"],
                        help="FedScout: what of the scout is FedAvg'd (all = head too, none = no averaging; ablations)")
    parser.add_argument('--scout_realign', type=int, default=1, help="FedScout: head-realignment epochs after each average (0 = off)")
    parser.add_argument('--scout_gate', type=int, default=20, help="FedScout: memorization-gate patience in rounds (0 = never freeze)")
    parser.add_argument('--scout_freeze_at', type=int, default=0, help="FedScout: force the freeze at this round (control)")
    parser.add_argument('--scout_freeze_head', type=str2bool, nargs='?', const=True, default=True,
                        help="FedScout: at the freeze, realign the personal head once and fix it (the scout stops changing)")
    parser.add_argument('--scout_cache', type=str2bool, nargs='?', const=True, default=True,
                        help="FedScout: after the freeze, cache the frozen backbone's features of the local training data")
    parser.add_argument('--scout_residual', type=str, default="p2", choices=["p2", "i1"],
                        help="FedScout: p2 = learner only through CE(scout + learner); i1 = + its own CE + KL to the scout")
    parser.add_argument('--scout_screen_tau', type=float, default=0.5,
                        help="FedScout: the learner trains only on samples where the received scout's p(y) < tau (0 = off)")
    parser.add_argument('--scout_alphas', type=str, default="0,0.1,0.25,0.5,0.75,1", help="FedScout: fusion weights of the learner, picked on the pooled held-out splits (ascending)")
    parser.add_argument('--scout_screen_prior', type=str2bool, nargs='?', const=True, default=False,
                        help="FedScout: with --scout_share all, screen on scout logits + log local class frequencies until the freeze")
    parser.add_argument('--scout_cascade_taus', type=str, default="0.5,0.7,0.9",
                        help="FedScout: evaluated inference cascades (learner only where the scout's top-1 prob < tau)")
    parser.add_argument('--pfedmoe_shared_lr_scale', type=float, default=1.0, help="pFedMoE: lr multiplier of the shared extractor (paper uses 1; fairness control)")
    parser.add_argument('--pfedmoe_gate_hidden', type=int, default=128, help="pFedMoE: hidden width of the gating network (not reported in the paper)")
    parser.add_argument('--apen_shared_lr_scale', type=float, default=1.0,
                        help="FedAPEN: multiply the shared model's local learning rate")
    parser.add_argument('--apen_personal_head', type=str2bool, nargs='?', const=True, default=False,
                        help="FedAPEN: keep the shared model's head per client (only its body is averaged)")
    parser.add_argument('--fedsa_cm_server_lr', type=float, default=1.0,
                        help="FedSA twin: server step on the averaged backbone update, prev + lr * (avg - prev)")
    parser.add_argument('--fedsa_cm_agree', type=float, default=0.0,
                        help="FedSA twin: average only coordinates whose weighted client sign agreement is >= this (0 = off)")
    parser.add_argument('--fedsa_cm_realign', type=int, default=0,
                        help="FedSA twin: epochs of personal projector/head training on the received averaged backbone (frozen) right after averaging, before evaluation and local training")
    parser.add_argument('--fedsa_cm_bn_local', type=str2bool, nargs='?', const=True, default=False,
                        help="FedSA twin: do not average BN running statistics (FedBN-style, kept per client)")
    parser.add_argument('--fedsa_cm_freeze_head', type=str2bool, nargs='?', const=True, default=False, help="at the gate's freeze also refit (realign) and then fix the twin's personal projector/head: the twin stops changing entirely")
    parser.add_argument('--fedsa_cm_freeze_at', type=int, default=0, help="force the gate's freeze at this round (fixed-round control; needs --fedsa_cm_gate > 0)")
    parser.add_argument('--fedsa_recover', type=str2bool, nargs='?', const=True, default=False, help="after the gate freezes the twin, add the private model's standalone CE + KL to the frozen twin (makes it usable alone)")
    parser.add_argument('--fedsa_save_at', type=int, default=0, help="save every client's twin and private model at this round (offline probes)")
    parser.add_argument('--fedsa_recover_temp', type=float, default=1.0, help="KD temperature of the post-freeze recovery (loss scaled by T^2)")
    parser.add_argument('--fedsa_recover_teacher', type=str, default="twin", choices=["twin", "fused"], help="KD target of the post-freeze recovery: the frozen twin, or the ensemble (twin + alpha * private) at freeze time")
    parser.add_argument('--fedsa_priv_gate', type=float, default=0.0, help="twin-gated private training: the private residual trains only on samples where the received twin's p(y) is below this (0 = off)")
    parser.add_argument('--fedsa_cm_cache', type=str2bool, nargs='?', const=True, default=False,
                        help="after the gate freezes the twin backbone, cache its features of my training data once and skip the backbone (exact: eval BN, no augmentation)")
    parser.add_argument('--fedsa_dump_from', type=int, default=0,
                        help="FedSA twin: from this round on, save twin/private logits on held-out and test per client at each evaluation to logs/dump_<goal>/ (0 = off)")
    parser.add_argument('--fedsa_cm_realign_before', type=str2bool, nargs='?', const=True, default=False,
                        help="control for --fedsa_cm_realign: run the same head epochs on the client's own backbone BEFORE the average replaces it (extra head training, no alignment)")
    parser.add_argument('--fedsa_cm_bsm', type=str2bool, nargs='?', const=True, default=False,
                        help="FedSA twin: balanced softmax CE (logits + log local class prior) so the twin stays prior-free and the private residual takes the local prior (FedRoD-style role split)")
    parser.add_argument('--fedsa_cm_lr_scale', type=float, default=1.0,
                        help="FedSA twin: multiply the twin's local learning rate (control for 'anchors only enlarge the step')")
    parser.add_argument('--fedsa_cm_from_priv', type=float, default=0.0,
                        help="FedSA twin: weight of KL(twin || private, detached), i.e. the twin also learns from the private model (FML/FedAPEN direction)")
    parser.add_argument('--fedsa_cm_ntd', type=float, default=0.0,
                        help="FedSA twin: weight of FedNTD not-true distillation from the received (frozen) twin")
    parser.add_argument('--fedsa_frozen_head', type=str, default='none', choices=['none', 'twin', 'both'],
                        help="FedSA twin: replace the twin's head (twin) or the twin's and private model's heads (both) by one fixed random linear head, identical on all clients (FedBABU-style)")
    parser.add_argument('--fedsa_gap_scale', type=float, default=1.0,
                        help="FedSA MCL: multiply the class margin (gap) by this; 0 = distance softmax to fixed anchors without margin")
    parser.add_argument('-mcl', '--mcl', "--margin_contrastive", dest="margin_contrastive", type=float, default=1.0)
    parser.add_argument('--mcl_variant', type=str, default='fedsa',
                        choices=['fedsa', 'hard_negative'],
                        help="FedSA-compatible distance softmax or boundary-focused hard-negative MCL")
    parser.add_argument('--mcl_gap_mode', type=str, default='min',
                        choices=['min', 'max'],
                        help="Combine local/global prototype gaps with the smaller or larger gap")
    parser.add_argument('--mcl_gap_class_cap', type=str2bool, nargs='?', const=True, default=False,
                        help="Cap the MCL gap per class (old adopted profile used true)")
    parser.add_argument('--mcl_hard_margin', type=float, default=0.5,
                        help="Target anchor margin for hard-negative MCL")
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0)


    args = parser.parse_args()
    if args.algorithm == "FedScout":
        args.fedsa_projector = "linear"  # residual learner = backbone -> Linear(-fd) -> head, like the scout
    if args.algorithm == "ETFProtoFL":
        # paper: projector output dim = number of classes, trainable projector after the backbone
        args.feature_dim = args.num_classes
        args.fedsa_projector = "linear"
    if args.algorithm == "FedDSA":
        from utils.feddsa_utils import configure_utility_module
        configure_utility_module(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if cuda_available:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = True  # fixed 32x32 inputs: ~7% faster per client step on 2080 Ti (measured)

    if args.device == "cuda" and not cuda_available:
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    experimental_names = {action.dest for action in experimental._group_actions}
    for arg in vars(args):
        label = f"[EXPERIMENTAL] {arg}" if arg in experimental_names else arg
        print(label, '=', getattr(args, arg))
    print("=" * 50)


    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
