import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch


class MNISTDataModule(L.LightningDataModule):
    """Data loading, preprocessing and splitting"""

    def __init__(self, data_dir="./data", batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # MNIST standard normalization values
        self.mean = (0.1307,)
        self.std = (0.3081,)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )
        # TODO: add data augmentation (rotation, random crop) for better generalization

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load full train dataset (60k images)
        mnist_full_train = datasets.MNIST(
            root=self.data_dir, train=True, transform=self.transform, download=True
        )

        # Split train into train (54k) and val (6k) - 90/10 split
        train_size = int(0.9 * len(mnist_full_train))
        val_size = len(mnist_full_train) - train_size

        generator = torch.Generator().manual_seed(42)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full_train, [train_size, val_size], generator=generator
        )

        # Load test dataset (10k images)
        self.mnist_test = datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
