"""Heterogeneous on-device models for inertial-sensor activity recognition (HAR, PAMAP2).

Input windows are (N, C, H, L) as saved by dataset/generate_HAR.py (9, 1, 128) and generate_PAMAP2.py (9, 3, 256);
every model flattens them to C*H channels over L time steps, sizes itself from `in_shape`, and ends in a Linear
`fc` so BaseHeadSplit can swap the classifier. The eight families mirror the device tiers of the IoT scenario:
MCU-class MLP / small CNN, mid-range CNN / GRU, and edge-class DeepConvLSTM / ResNet / TCN / Transformer.
`WidthSenResNet` is the width-nested shared model (its width-r tensors are leading slices of the width-1 model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _seq(x):
    return x.flatten(1, 2) if x.dim() == 4 else x  # (N, C, H, L) -> (N, C*H, L)


class SenMLP(nn.Module):
    def __init__(self, in_shape, num_classes):
        super().__init__()
        c, h, l = in_shape
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(c * h * l, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.body(_seq(x)))


class SenCNN(nn.Module):
    """HARCNN-style shallow CNN (2 conv + pool) or a deeper BN CNN (4 conv)."""

    def __init__(self, in_shape, num_classes, deep=False):
        super().__init__()
        c = in_shape[0] * in_shape[1]
        if deep:
            chans, k = [64, 64, 128, 128], 5
            layers = []
            for i, o in enumerate(chans):
                layers += [nn.Conv1d(c if i == 0 else chans[i - 1], o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
                if i % 2 == 1:
                    layers.append(nn.MaxPool1d(2))
        else:
            chans = [32, 64]
            layers = [nn.Conv1d(c, 32, 9), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 9), nn.ReLU(), nn.MaxPool1d(2)]
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x):
        return self.fc(self.body(_seq(x)))


class SenGRU(nn.Module):
    def __init__(self, in_shape, num_classes, hidden=128):
        super().__init__()
        self.rnn = nn.GRU(in_shape[0] * in_shape[1], hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(_seq(x).transpose(1, 2))
        return self.fc(out[:, -1])


class DeepConvLSTM(nn.Module):
    """Ordonez & Roggen (2016) style: 4 conv layers then a 2-layer LSTM."""

    def __init__(self, in_shape, num_classes, filters=64, hidden=128):
        super().__init__()
        c = in_shape[0] * in_shape[1]
        self.conv = nn.Sequential(*[m for i in range(4) for m in
                                    (nn.Conv1d(c if i == 0 else filters, filters, 5, padding=2), nn.ReLU())])
        self.rnn = nn.LSTM(filters, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(self.conv(_seq(x)).transpose(1, 2))
        return self.fc(out[:, -1])


class _Block1d(nn.Module):
    def __init__(self, cin, cout, stride, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride, dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.down is None else self.down(x)))


class WidthSenResNet(nn.Module):
    """1D ResNet (stem + 3 stages of 2 basic blocks, 64/128/256 channels at width 1). Used as a private model
    (width=1) and as the width-nested shared model."""

    def __init__(self, in_shape, num_classes, width=1.0):
        super().__init__()
        c = [max(4, int(round(64 * width * m))) for m in (1, 2, 4)]
        self.stem = nn.Sequential(nn.Conv1d(in_shape[0] * in_shape[1], c[0], 7, 1, 3, bias=False),
                                  nn.BatchNorm1d(c[0]), nn.ReLU())
        stages, cin = [], c[0]
        for i, cout in enumerate(c):
            stages.append(nn.Sequential(_Block1d(cin, cout, 1 if i == 0 else 2), _Block1d(cout, cout, 1)))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(c[-1], num_classes)

    def forward(self, x):
        return self.fc(self.stages(self.stem(_seq(x))).mean(-1))


class SenTCN(nn.Module):
    """Temporal convolutional network: dilated residual blocks (dilation 1, 2, 4, 8)."""

    def __init__(self, in_shape, num_classes, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(in_shape[0] * in_shape[1], ch, 1)
        self.blocks = nn.Sequential(*[_Block1d(ch, ch, 1, d) for d in (1, 2, 4, 8)])
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        return self.fc(self.blocks(self.inp(_seq(x))).mean(-1))


class SenTransformer(nn.Module):
    """Patch-embedded (stride 4) 2-layer Transformer encoder with mean pooling."""

    def __init__(self, in_shape, num_classes, d=64, patch=4):
        super().__init__()
        self.embed = nn.Conv1d(in_shape[0] * in_shape[1], d, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, in_shape[2] // patch, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 2)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        z = self.embed(_seq(x)).transpose(1, 2)
        return self.fc(self.enc(z + self.pos[:, : z.shape[1]]).mean(1))
