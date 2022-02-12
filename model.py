import math
from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from equalized_lr import EqualizedLinear, EqualizedConv2d, EqualizedWeight


class MappingNetwork(nn.Module):
    """
    <a id="mapping_network"></a>

    ## Mapping Network

    ![Mapping Network](mapping_network.svg)

    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)


class Generator(nn.Module):
    """
    <a id="generator"></a>

    ## StyleGAN2 Generator

    ![Generator](style_gan2.svg)

    ---*$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [`toRGB`](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.*---

    The generator starts with a learned constant.
    Then it has a series of blocks. The feature map resolution is doubled at each block
    Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.
    """

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `d_latent` is the dimensionality of $w$
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Calculate the number of features for each block
        #
        # Something like `[512, 512, 256, 128, 64, 32]`
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        # Number of generator blocks
        self.n_blocks = len(features)

        # Trainable $4 \times 4$ constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # First style block for $4 \times 4$ resolution and layer to get RGB
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        # Generator blocks
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # $2 \times$ up sampling layer. The feature space is up sampled
        # at each block
        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        * `w` is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate
        $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]`.
        * `input_noise` is the noise for each block.
        It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs
        after each convolution layer (see the diagram).
        """

        # Get batch size
        batch_size = w.shape[1]

        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # The first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Get first rgb image
        rgb = self.to_rgb(x, w[0])

        # Evaluate rest of the blocks
        for i in range(1, self.n_blocks):
            # Up sample the feature map
            x = self.up_sample(x)
            # Run it through the [generator block](#generator_block)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # Up sample the RGB image and add to the rgb from the block
            rgb = self.up_sample(rgb) + rgb_new

        # Return the final RGB image
        return rgb


class GeneratorBlock(nn.Module):
    """
    <a id="generator_block"></a>

    ### Generator Block

    ![Generator block](generator_block.svg)

    ---*$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [`toRGB`](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.*---

    The generator block consists of two [style blocks](#style_block) ($3 \times 3$ convolutions with style modulation)
    and an RGB output.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        # Second [style block](#style_block)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        # *toRGB* layer
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])

        # Get RGB image
        rgb = self.to_rgb(x, w)

        # Return feature map and rgb image
        return x, rgb


class StyleBlock(nn.Module):
    """
    <a id="style_block"></a>

    ### Style Block

    ![Style block](style_block.svg)

    ---*$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is single channel).*---

    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tensor of shape `[batch_size, 1, height, width]`
        """
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    <a id="to_rgb"></a>

    ### To RGB

    ![To RGB](to_rgb.svg)

    ---*$A$ denotes a linear layer.*---

    Generates an RGB image from a feature map using $1 \times 1$ convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation

    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        # Calculate the number of features for each block.
        # Something like `[64, 128, 256, 512, 512, 512]`.
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1  # + 1 because of MiniBatchStdDev
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        # Try to normalize the image (this is totally optional, but speeds up the early training a little)
        x = x - 0.5

        x = self.from_rgb(x)

        x = self.blocks(x)

        x = self.std_dev(x)

        x = self.conv(x)

        x = x.reshape(x.shape[0], -1)  # flatten

        return self.final(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.residual = nn.Sequential(
            DownSample(),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample()

        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)

        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """
    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        #
        # \begin{align}
        # \mu_{i} &= \frac{1}{N} \sum_g x_{g,i} \\
        # \sigma_{i} &= \sqrt{\frac{1}{N} \sum_g (x_{g,i} - \mu_i)^2  + \epsilon}
        # \end{align}
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


# Based on the paper Making Convolutional Networks Shift-Invariant Again
# https://papers.labml.ai/paper/1904.11486
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
