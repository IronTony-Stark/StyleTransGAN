import math

import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


# https://nn.labml.ai/gan/wasserstein/index.html
def discriminator_loss(f_real: torch.Tensor, f_fake: torch.Tensor):
    """
    This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
    which are later added.
    They are kept separate for logging.
    We use ReLUs to clip the loss to keep $f in [-1, +1]$ range.
    """
    return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


def generator_loss(f_fake: torch.Tensor):
    return -f_fake.mean()


# https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html
def gradient_penalty(x: torch.Tensor, d: torch.Tensor):
    batch_size = x.shape[0]

    # Calculate gradients of D(x) with respect to x.
    # `grad_outputs` is set to 1 since we want the gradients of D(x),
    # and we need to create and retain graph since we have to compute gradients
    # with respect to weight on this loss.
    gradients, *_ = torch.autograd.grad(outputs=d,
                                        inputs=x,
                                        grad_outputs=d.new_ones(d.shape),
                                        create_graph=True)

    gradients = gradients.reshape(batch_size, -1)

    norm = gradients.norm(2, dim=-1)

    return torch.mean(norm ** 2)


# https://nn.labml.ai/gan/stylegan/index.html#section-180
class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True
        )

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss
