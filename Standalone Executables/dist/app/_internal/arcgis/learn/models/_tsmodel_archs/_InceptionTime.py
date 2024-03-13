# Referred from https://github.com/tcapelle/timeseries_fastai

import torch
import torch.nn as nn
from .layers import *


def shortcut(c_in, c_out):
    return nn.Sequential(
        *[nn.Conv1d(c_in, c_out, kernel_size=1), nn.BatchNorm1d(c_out)]
    )


class _TSInceptionBlock(nn.Module):
    def __init__(self, input, nb_filters=32, ks=40, bottleneck=32):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(input, bottleneck, 1) if bottleneck and input > 1 else noop
        )

        kss = [ks // (2**i) for i in range(3)]
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck if (bottleneck > 1 and input > 1) else input,
                    nb_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                )
                for ks in kss
            ]
        )

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = nn.Conv1d(input, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        layers_output = None
        for layer in self.conv_layers:
            output = layer(x)

            if layers_output is None:
                layers_output = output
            else:
                layers_output = torch.cat((layers_output, output), 1)

        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((layers_output, mp), 1)
        return self.act(self.bn(inc_out))


class _TSInceptionTime(nn.Module):
    def __init__(self, input, output, ks=40, depth=6, bottleneck=32, nb_filters=32):
        super().__init__()

        inception_layers = []
        residual_layers = []
        res = 0
        for d in range(depth):
            inception_layers.append(
                _TSInceptionBlock(
                    input if d == 0 else nb_filters * 4,
                    bottleneck=bottleneck if d > 0 else 0,
                    ks=ks,
                    nb_filters=nb_filters,
                )
            )
            if d % 3 == 2:
                if res == 0:
                    residual_layers.append(shortcut(input, nb_filters * 4))
                else:
                    residual_layers.append(shortcut(nb_filters * 4, nb_filters * 4))

                res = res + 1
            else:
                residual_layers.append(None)

        self.depth = depth
        self.inception_layers = nn.ModuleList(inception_layers)
        self.residual_layers = nn.ModuleList(residual_layers)
        self.act_fn = nn.ReLU()
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(nb_filters * 4, output)

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception_layers[d](x)
            if d % 3 == 2:
                res = self.residual_layers[d](res)
                x += res
                res = x
                x = self.act_fn(x)

        x = self.avg_pooling(x).squeeze(-1)
        x = self.linear(x)
        return x
