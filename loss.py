import torch.utils.data
from torch.nn import functional as F


# todo refactor
class DiscriminatorLoss(torch.nn.Module):
    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        We use ReLUs to clip the loss to keep $f in [-1, +1]$ range.
        """
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(torch.nn.Module):
    def forward(self, f_fake: torch.Tensor):
        return -f_fake.mean()
