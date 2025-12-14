import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch


class CIFAR10DataModule(L.LightningDataModule):
    """Data loading, preprocessing and splitting"""

    def __init__(self, data_dir="./data", batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 standard normalization values (RGB)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )
        # TODO: add data augmentation (RandomHorizontalFlip, RandomCrop) to improve accuracy

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load full train dataset (50k images)
        cifar_full_train = datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.transform
        )

        # Split train into train (45k) and val (5k) - 90/10 split
        train_size = int(0.9 * len(cifar_full_train))
        val_size = len(cifar_full_train) - train_size

        generator = torch.Generator().manual_seed(42)
        self.cifar_train, self.cifar_val = random_split(
            cifar_full_train, [train_size, val_size], generator=generator
        )

        # Load test dataset (10k images)
        self.cifar_test = datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
