# Based on the paper Making Convolutional Networks Shift-Invariant Again
# https://papers.labml.ai/paper/1904.11486

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()

        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


# aka Blur
class Smooth(nn.Module):
    def __init__(self):
        super().__init__()

        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()  # normalize

        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = x.view(-1, 1, h, w)

        x = self.pad(x)

        x = F.conv2d(x, self.kernel)

        return x.view(b, c, h, w)