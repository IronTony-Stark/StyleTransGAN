# Equalized Learning Rate
# Introduced in Progressive GAN https://arxiv.org/abs/1710.10196
# Implementation taken from https://nn.labml.ai/gan/stylegan/index.html#section-165
# Another implementation: https://personal-record.onrender.com/post/equalized-lr/

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()

        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        super().__init__()

        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c
