import argparse
import math
from typing import Iterator, Tuple

import torch
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from loss import discriminator_loss, generator_loss, gradient_penalty, PathLengthPenalty
from model import Discriminator, Generator, MappingNetwork
from utils import cycle_dataloader, log_weights, pretty_json, ImageDataset, Checkpoint


manual_seed = True  # for reproducibility


class Trainer:
    # Logger
    writer: SummaryWriter

    # Networks
    discriminator: Discriminator
    generator: Generator
    mapping_network: MappingNetwork

    # Penalties
    gradient_penalty_coefficient: float = 10.

    path_length_penalty: PathLengthPenalty
    path_length_beta = 0.99

    # Optimizers
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam

    # Checkpoint
    checkpoint: Checkpoint

    # DataLoader
    loader: Iterator

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device

        self.writer = SummaryWriter()
        self.writer.add_text('Config', pretty_json(vars(self.args)))

        log_resolution = int(math.log2(self.args.image_size))

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.args.d_latent).to(self.device)
        self.mapping_network = MappingNetwork(self.args.d_latent, self.args.mapping_network_layers).to(self.device)

        self.path_length_penalty = PathLengthPenalty(self.path_length_beta).to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.learning_rate, betas=self.args.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.args.learning_rate, betas=self.args.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.args.mapping_network_learning_rate, betas=self.args.adam_betas
        )

        self.checkpoint = Checkpoint(
            self.discriminator, self.generator, self.mapping_network,
            self.discriminator_optimizer, self.generator_optimizer, self.mapping_network_optimizer,
        )

        dataset = ImageDataset(self.args.dataset_path, self.args.image_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, num_workers=2,
            shuffle=True and not manual_seed, drop_last=True, pin_memory=True
        )
        self.loader = cycle_dataloader(dataloader)

    # generate noise and use mapping network to create W (uses style mixing)
    def generate_w(self, batch_size: int):
        if torch.rand(()).item() < self.args.style_mixing_prob:
            # Style mixing
            z1 = torch.randn(batch_size, self.args.d_latent).to(self.device)
            z2 = torch.randn(batch_size, self.args.d_latent).to(self.device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            cross_over_point = int(torch.rand(()).item() * self.generator.n_blocks)
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.generator.n_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            # Without style mixing
            z = torch.randn(batch_size, self.args.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.generator.n_blocks, -1, -1)

    # Generate noise for each generator's block
    def generate_noise(self, batch_size: int):
        noise = []

        resolution = 4
        for i in range(self.generator.n_blocks):
            # The first block has only one 3x3 convolution
            if i == 0:
                n1 = None
            else:
                # Generate noise to add after the first convolution layer
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            noise.append((n1, n2))

            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, batch_size: int):
        w = self.generate_w(batch_size)
        noise = self.generate_noise(batch_size)

        images = self.generator(w, noise)

        return images, w

    def step(self, idx: int):
        # TODO generator different images for G and D update
        # Train Discriminator
        self.discriminator_optimizer.zero_grad()

        generated_images, _ = self.generate_images(self.args.batch_size)
        fake_output = self.discriminator(generated_images.detach())

        real_images = next(self.loader).to(self.device)

        # We need to calculate gradients w.r.t. real images for gradient penalty
        if (idx + 1) % self.args.lazy_gradient_penalty_interval == 0:
            real_images.requires_grad_()

        real_output = self.discriminator(real_images)

        real_loss, fake_loss = discriminator_loss(real_output, fake_output)

        dis_loss = real_loss + fake_loss

        if (idx + 1) % self.args.lazy_gradient_penalty_interval == 0:
            gp = gradient_penalty(real_images, real_output)

            # todo do you really need to multiply by interval?
            dis_loss = dis_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.args. \
                lazy_gradient_penalty_interval

            self.writer.add_scalar("Discriminator/Gradient Penalty", gp.item(), idx)

        dis_loss.backward()

        # For stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        self.discriminator_optimizer.step()

        self.writer.add_scalar("Discriminator/Loss", dis_loss.item(), idx)
        self.writer.add_scalar("Discriminator/Real Score", real_output.mean().item(), idx)
        self.writer.add_scalar("Discriminator/Fake Score", fake_output.mean().item(), idx)

        # Train the generator
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        generated_images, w = self.generate_images(self.args.batch_size)

        fake_output = self.discriminator(generated_images)

        gen_loss = generator_loss(fake_output)

        if idx > self.args.lazy_path_penalty_after and (idx + 1) % self.args.lazy_path_penalty_interval == 0:
            plp = self.path_length_penalty(w, generated_images)
            if not torch.isnan(plp):
                self.writer.add_scalar("Generator/Path Length Regularization", plp.item(), idx)
                gen_loss = gen_loss + plp

        gen_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        self.writer.add_scalar("Generator/Loss", gen_loss.item(), idx)
        self.writer.add_scalar("Generator/Score", fake_output.mean().item(), idx)

        # Logging
        if idx % self.args.log_losses_interval == 0:
            print(
                f"[Step {idx}/{self.args.training_steps}] "
                f"[D loss: {round(dis_loss.item(), 4) if dis_loss else '?'}] "
                f"[G loss: {round(gen_loss.item(), 4) if gen_loss else '?'}] "
            )

        if (idx + 1) % self.args.log_models_weights_interval == 0:
            log_weights(self.writer, self.discriminator, idx)
            log_weights(self.writer, self.generator, idx)
            log_weights(self.writer, self.mapping_network, idx)

        if (idx + 1) % self.args.log_generated_images_interval == 0:
            self.writer.add_image(
                "Generated Images",
                vutils.make_grid(generated_images, padding=2, normalize=True, scale_each=True),
                idx
            )

        # Checkpoint
        if (idx + 1) % self.args.save_checkpoint_interval == 0:
            self.checkpoint.save(f"{idx}.pth", gen_loss.item(), idx)

    def train(self, num_steps: int):
        for i in range(num_steps):
            self.step(i)


def main():
    if manual_seed:
        import random
        import numpy as np

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../input/celeba-dataset/img_align_celeba/img_align_celeba/",
        help="Path to the folder with images"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image size"
    )

    # Intervals
    # Instead of calculating the regularization losses, the paper proposes lazy regularization
    # where the regularization terms are calculated once in a while.
    # This improves the training efficiency a lot.
    parser.add_argument(
        "--lazy_gradient_penalty_interval",
        type=int,
        default=4,
        help="Gradient penalty calculation interval"
    )
    parser.add_argument(
        "--lazy_path_penalty_interval",
        type=int,
        default=32,
        help="Path length regularization calculation interval"
    )
    parser.add_argument(
        "--lazy_path_penalty_after",
        type=int,
        default=5_000,
        help="Calculate path length regularization after step n"
    )
    parser.add_argument(
        "--log_losses_interval",
        type=int,
        default=100,
        help="Log losses every n steps"
    )
    parser.add_argument(
        "--log_generated_images_interval",
        type=int,
        default=500,
        help="Log generated images every n steps"
    )
    parser.add_argument(
        "--log_models_weights_interval",
        type=int,
        default=500,
        help="Log model weights every n steps"
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=2_000,
        help="Save checkpoint every n steps"
    )

    # HyperParameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for generator and discriminator networks"
    )
    parser.add_argument(
        "--mapping_network_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for mapping network"
    )
    parser.add_argument(
        "--adam_betas",
        type=Tuple[float, float],
        default=(0.0, 0.99),
        help="Betas for Adam optimizer"
    )
    parser.add_argument(
        "--d_latent",
        type=int,
        default=512,
        help="Dimensionality of z and w"
    )
    parser.add_argument(
        "--mapping_network_layers",
        type=int,
        default=8,
        help="Number of layers in mapping network"
    )
    parser.add_argument(
        "--style_mixing_prob",
        type=float,
        default=0.9,
        help="A probability of applied style mixing"
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=150_000,
        help="Number of training steps"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    trainer.train(args.training_steps)


if __name__ == '__main__':
    main()
