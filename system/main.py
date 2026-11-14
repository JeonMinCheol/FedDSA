#!/usr/bin/env python
import torch
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
from flcore.servers.serversa import FedSA
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
    parser.add_argument('--prototype_utility', type=str2bool, nargs='?', const=True, default=False,
                        help="Estimate delayed class-conditional utility with a one-step counterfactual probe")
    parser.add_argument('--utility_inner_lr', type=float, default=0.01,
                        help="Virtual local step size for the prototype utility probe")
    parser.add_argument('--utility_temperature', type=float, default=0.1,
                        help="Temperature for utility-conditioned local anchor supervision")
    parser.add_argument('--utility_ema', type=float, default=0.25,
                        help="EMA rate for server class utility state")
    parser.add_argument('--utility_min_weight', type=float, default=0.25,
                        help="Minimum utility-conditioned local anchor weight")
    parser.add_argument('--utility_max_weight', type=float, default=1.5,
                        help="Maximum utility-conditioned local anchor weight")
    experimental.add_argument('--utility_local_gate', type=str2bool, nargs='?', const=True, default=True,
                        help="[EXPERIMENTAL] Apply utility-conditioned weights to local anchor supervision")
    experimental.add_argument('--utility_weight_normalize', type=str2bool, nargs='?', const=True, default=True,
                        help="[EXPERIMENTAL] Normalize local utility weights to mean one; false uses 2*sigmoid for signed utility (zero maps to one)")
    experimental.add_argument('--utility_signal_mode', type=str, default='observed',
                        choices=['observed', 'shuffled', 'uniform', 'zscore'],
                        help="[EXPERIMENTAL] Utility signal for controls: observed, shuffled, uniform, or z-scored")
    experimental.add_argument('--utility_positive_only', type=str2bool, nargs='?', const=True, default=False,
                        help="[EXPERIMENTAL] Only allow positive counterfactual utility to increase anchor usage")
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
    parser.add_argument('-mcl', '--mcl', "--margin_contrastive", dest="margin_contrastive", type=float, default=1.0)
    parser.add_argument('--mcl_variant', type=str, default='fedsa',
                        choices=['fedsa', 'hard_negative'],
                        help="FedSA-compatible distance softmax or boundary-focused hard-negative MCL")
    parser.add_argument('--mcl_gap_mode', type=str, default='min',
                        choices=['min', 'max'],
                        help="Combine local/global prototype gaps with the smaller or larger gap")
    parser.add_argument('--mcl_hard_margin', type=float, default=0.5,
                        help="Target anchor margin for hard-negative MCL")
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0)


    args = parser.parse_args()
    if args.algorithm == "FedDSA":
        from utils.feddsa_utils import configure_utility_module
        configure_utility_module(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if cuda_available:
        torch.cuda.manual_seed_all(args.random_seed)

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
