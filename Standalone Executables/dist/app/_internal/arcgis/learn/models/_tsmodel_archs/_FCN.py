# Referred from https://github.com/tcapelle/timeseries_fastai
import torch
import torch.nn as nn
from .layers import *


class _TSFCN(nn.Module):
    def __init__(
        self, input, output, layers=[128, 256, 128], kss=[7, 5, 3], device=None
    ):
        super().__init__()

        if not isinstance(kss, list) or len(kss) == 0:
            kss = [7, 5, 3]

        i = 0

        self.conv_layers = nn.ModuleList([])
        sizes = zip([input] + layers, layers)
        for n1, n2 in sizes:
            if i < len(kss):
                ks = kss[i]
            else:
                ks = kss[-1]
            self.conv_layers.append(convlayer(n1, n2, ks))

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(layers[-1], output)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.avg_pooling(x).squeeze(-1)
        x = self.linear(x)
        return x
