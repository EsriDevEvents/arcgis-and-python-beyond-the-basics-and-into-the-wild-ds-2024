# Referred from https://github.com/tcapelle/timeseries_fastai
import torch
import torch.nn as nn
from .layers import *


class Shortcut(nn.Module):
    def __init__(self, ni, nf, device=None):
        super().__init__()
        if ni == nf:
            self.conv = noop
        else:
            self.conv = convlayer(ni, nf, ks=1, act_fn=False)

    def forward(self, x):
        return self.conv(x)


class _TSResNet(nn.Module):
    def __init__(self, input, output, device=None):
        super().__init__()
        conv_sizes = [64, 128, 128]
        kss = [7, 5, 3]

        self.layers = []
        self.block1 = nn.ModuleList(
            [
                convlayer(input, conv_sizes[0], kss[0], act_fn="relu"),
                convlayer(conv_sizes[0], conv_sizes[0], kss[1], act_fn="relu"),
                convlayer(conv_sizes[0], conv_sizes[0], kss[2], act_fn=False),
                Shortcut(input, conv_sizes[0]),
                nn.ReLU(),
            ]
        )

        self.block2 = nn.ModuleList(
            [
                convlayer(conv_sizes[0], conv_sizes[1], kss[0], act_fn="relu"),
                convlayer(conv_sizes[1], conv_sizes[1], kss[1], act_fn="relu"),
                convlayer(conv_sizes[1], conv_sizes[1], kss[2], act_fn=False),
                Shortcut(conv_sizes[0], conv_sizes[1]),
                nn.ReLU(),
            ]
        )

        self.block3 = nn.ModuleList(
            [
                convlayer(conv_sizes[1], conv_sizes[2], kss[0], act_fn="relu"),
                convlayer(conv_sizes[2], conv_sizes[2], kss[1], act_fn="relu"),
                convlayer(conv_sizes[2], conv_sizes[2], kss[2], act_fn=False),
                Shortcut(conv_sizes[1], conv_sizes[2]),
                nn.ReLU(),
            ]
        )

        self.blocks = [self.block1, self.block2, self.block3]

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(conv_sizes[-1], output)

    def forward(self, x):
        for block in self.blocks:
            orig = x
            for layer in block:
                if not isinstance(layer, Shortcut):
                    x = layer(x)
                else:
                    x = x + layer(orig)

        x = self.avg_pooling(x).squeeze(-1)
        output = self.linear(x)

        return output
