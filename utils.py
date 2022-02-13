import json
import os
import typing
from pathlib import Path

import torch.utils.data
import torch.utils.tensorboard
import torchvision
from PIL import Image


def cycle_dataloader(data_loader: torch.utils.data.DataLoader):
    while True:
        for batch in data_loader:
            yield batch


def log_weights(writer: torch.utils.tensorboard.SummaryWriter, model: torch.nn.Module, iteration: int):
    for name, weight in model.named_parameters():
        if weight is not None:
            writer.add_histogram(f"Generator/{name}", weight, iteration)
        if weight.grad is not None:
            writer.add_histogram(f"Generator/{name}.grad", weight.grad, iteration)


def pretty_json(json_dict: typing.Dict):
    json_hp = json.dumps(json_dict, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, image_size: int):
        super().__init__()

        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # todo augmentation
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Checkpoint:
    def __init__(
            self,
            discriminator: torch.nn.Module, generator: torch.nn.Module, mapping_network: torch.nn.Module,
            optimizer_dis: torch.optim.Adam, optimizer_gen: torch.optim.Adam, optimizer_map: torch.optim.Adam,
            checkpoint_folder: str = "checkpoints",
    ):
        self.checkpoint_dir = checkpoint_folder
        self.discriminator = discriminator
        self.generator = generator
        self.mapping_network = mapping_network
        self.optimizer_dis = optimizer_dis
        self.optimizer_gen = optimizer_gen
        self.optimizer_map = optimizer_map

    def save(self, filename: str, score: float, step: int, save_optimizers_state: bool = False):
        state_dict = {
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "mapping_network_state_dict": self.mapping_network.state_dict(),
            "score": score,
            "step": step,
        }
        if save_optimizers_state:
            state_dict["optimizer_discriminator_state_dict"] = self.optimizer_dis.state_dict()
            state_dict["optimizer_generator_state_dict"] = self.optimizer_gen.state_dict()
            state_dict["optimizer_mapping_network_state_dict"] = self.optimizer_map.state_dict()
        torch.save(state_dict, os.path.join(self.checkpoint_dir, filename))

    def load(self, filename: str, has_optimizers_state: bool = False, device: torch.device = torch.device("cpu")):
        state_dict = torch.load(os.path.join(self.checkpoint_dir, filename), map_location=device)
        self.discriminator.load_state_dict(state_dict["discriminator_state_dict"])
        self.generator.load_state_dict(state_dict["generator_state_dict"])
        self.mapping_network.load_state_dict(state_dict["mapping_network_state_dict"])
        if has_optimizers_state:
            self.optimizer_dis.load_state_dict(state_dict["optimizer_discriminator_state_dict"])
            self.optimizer_gen.load_state_dict(state_dict["optimizer_generator_state_dict"])
            self.optimizer_map.load_state_dict(state_dict["optimizer_mapping_network_state_dict"])
        return state_dict["score"], state_dict["step"]
