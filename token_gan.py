import torch
import torch.nn as nn
from equalized_lr import EqualizedLinear
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    def __init__(self, style_dim: int, style_num: int, mlp_layers_num: int = 8):
        super().__init__()

        layers = []  # [PixelNorm()]
        for _ in range(mlp_layers_num):
            layers.append(EqualizedLinear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(EqualizedLinear(style_dim, style_num * style_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.style = nn.Sequential(*layers)
        self.style_num = style_num
        self.style_dim = style_dim

    def forward(self, z: torch.Tensor):
        """
        :param z: [batch_size, style_dim]
        :return: [batch_size, style_num, style_dim]
        """
        z = F.normalize(z, dim=1)
        return self.style(z).view(-1, self.style_num, self.style_dim)


class StyleModulation(nn.Module):
    def __init__(
            self,
            size: int, content_dim: int, style_num: int, style_dim: int, patch_size: int,
            style_mod='prod',
            norm_type='layernorm'
    ):
        super().__init__()

        self.style_mod = style_mod
        self.norm_type = norm_type
        self.patch_size = patch_size
        self.keys = nn.Parameter(nn.init.orthogonal_(torch.empty(1, style_num, content_dim)))
        self.pos = nn.Parameter(torch.zeros(1, (size // patch_size) ** 2, content_dim))
        self.attention = MultiHeadAttention(content_dim, content_dim, content_dim, style_dim)

    def forward(self, x, s, is_new_style=False):
        """
        :param x: [batch_size, width * height, content_dim]
        :param s: [batch_size, style_num, style_dim]
        :param is_new_style: for debugging
        :return: input with style applied
        """
        b, t, c = x.size()

        # remove old style
        x = norm(x)
        x = x.view(b, t, -1, self.patch_size, self.patch_size)

        # calculate new style
        if not is_new_style:
            # multi-head attention
            query = torch.mean(x, dim=[3, 4])
            keys = self.keys.repeat(x.size(0), 1, 1)
            pos = self.pos.repeat(x.size(0), 1, 1)
            new_style, _ = self.attention(q=query + pos, k=keys, v=s)
        else:
            new_style = s

        # append new style
        if self.style_mod == 'prod':
            out = x * new_style.unsqueeze(-1).unsqueeze(-1)
        elif self.style_mod == 'plus':
            out = x + new_style.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('Have not implemented this type of style modulation')

        out = out.view(b, t, c)
        return out, (new_style.detach(),)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, qdim, kdim, vdim):
        super(MultiHeadAttention, self).__init__()

        self.scale = embed_dim ** -0.5
        self.to_q = EqualizedLinear(qdim, embed_dim, bias=False)
        self.to_k = EqualizedLinear(kdim, embed_dim, bias=False)
        self.to_v = EqualizedLinear(vdim, embed_dim, bias=False)

    def forward(self, q, k, v):
        b, n, dim = q.size()
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.bmm(attn, v)
        return out, (attn.detach(),)


def norm(input, norm_type='layernorm'):
    # [b, hw, c]
    if norm_type == 'layernorm' or norm_type == 'l2norm':
        normdim = -1
    elif norm_type == 'insnorm':
        normdim = 1
    else:
        raise NotImplementedError('have not implemented this type of normalization')

    if norm_type != 'l2norm':
        mean = torch.mean(input, dim=normdim, keepdim=True)
        input = input - mean

    demod = torch.rsqrt(torch.sum(input ** 2, dim=normdim, keepdim=True) + 1e-8)
    return input * demod


# Pixelwise feature vector normalization introduced in Progressive GAN https://arxiv.org/abs/1710.10196
class PixelNorm(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()

        self.feature_dim = feature_dim

    def forward(self, input: torch.Tensor):
        return input / torch.sqrt(torch.mean(input ** 2, dim=self.feature_dim, keepdim=True) + 1e-8)
