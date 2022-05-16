import torch
import torch.nn as nn
import torch.nn.functional as F

from equalized_lr import EqualizedLinear


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

    def forward(self, x: torch.Tensor, s: torch.Tensor, is_new_style: bool = False):
        """
        :param x: [batch_size, content_dim, width, height]
        :param s: [batch_size, style_num, style_dim]
        :param is_new_style: for debugging
        :return: input with style applied
        """
        # [batch_size, content_dim, width * height]
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        # [batch_size, width * height, content_dim]
        x = x.permute(0, 2, 1)

        b, t, c = x.size()

        # remove old style
        x = demodulate(x)
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

        # [batch_size, content_dim, width * height]
        out = out.permute(0, 2, 1)
        # [batch_size, content_dim, width, height]
        out = out.view(out.size(0), out.size(1), int(out.size(2) ** 0.5), int(out.size(2) ** 0.5))

        return out, (new_style.detach(),)


# todo wait.. is this really multi head?
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, qdim, kdim, vdim):
        super(MultiHeadAttention, self).__init__()

        self.scale = embed_dim ** -0.5
        self.to_q = EqualizedLinear(qdim, embed_dim, bias=False)
        self.to_k = EqualizedLinear(kdim, embed_dim, bias=False)
        self.to_v = EqualizedLinear(vdim, embed_dim, bias=False)

    def forward(self, q, k, v):
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.bmm(attn, v)
        return out, (attn.detach(),)


def demodulate(x: torch.Tensor, norm_type: str = "LN"):
    """
    :param x: [batch_size, width * height, content_dim]
    :param norm_type: LN - layer norm, L2 - L2 norm, IN - instance norm
    :return: demodulated (normalized) x
    """
    if norm_type == "LN" or norm_type == "L2":
        norm_dim = -1
    elif norm_type == "IN":
        norm_dim = 1
    else:
        raise NotImplementedError('have not implemented this type of normalization')

    if norm_type != "L2":
        mean = torch.mean(x, dim=norm_dim, keepdim=True)
        x = x - mean

    demodulation = torch.rsqrt(torch.sum(x ** 2, dim=norm_dim, keepdim=True) + 1e-8)

    return x * demodulation


# Pixelwise feature vector normalization introduced in Progressive GAN https://arxiv.org/abs/1710.10196
class PixelNorm(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()

        self.feature_dim = feature_dim

    def forward(self, input: torch.Tensor):
        return input / torch.sqrt(torch.mean(input ** 2, dim=self.feature_dim, keepdim=True) + 1e-8)
