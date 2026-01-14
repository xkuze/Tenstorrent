"""
Dataset for image segmentation.
Supports Oxford-IIIT Pet Dataset or custom image/mask pairs.
"""

from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision.datasets import OxfordIIITPet


class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset.
    Expects images and masks in separate directories with matching names.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform=None,
        img_size: int = 256,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.img_size = img_size

        # Get list of images
        self.images = sorted(
            list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (same name, different extension possibly)
        mask_name = img_path.stem + ".png"
        mask_path = self.masks_dir / mask_name
        if not mask_path.exists():
            mask_name = img_path.stem + ".jpg"
            mask_path = self.masks_dir / mask_name

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Normalize mask to 0-1
        mask = (mask > 127).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


class OxfordPetSegmentation(Dataset):
    """
    Oxford-IIIT Pet Dataset for segmentation.
    Downloads automatically if not present.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "trainval",
        transform=None,
        img_size: int = 256,
    ):
        self.img_size = img_size
        self.transform = transform

        # Download dataset
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # Convert PIL to numpy
        image = np.array(image)
        mask = np.array(mask)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )

        # Oxford Pet mask: 1=foreground, 2=background, 3=boundary
        # Convert to binary: 1 (pet) vs 0 (background/boundary)
        mask = (mask == 1).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_train_transforms(img_size: int = 256):
    """Training augmentations"""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256):
    """Validation/test transforms (no augmentation)"""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class SegmentationDataModule(L.LightningDataModule):
    """
    Lightning DataModule for segmentation.
    Uses Oxford-IIIT Pet Dataset by default.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        img_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Download dataset
        OxfordIIITPet(root=self.data_dir, split="trainval", download=True)
        OxfordIIITPet(root=self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = OxfordPetSegmentation(
                root=self.data_dir,
                split="trainval",
                transform=get_train_transforms(self.img_size),
                img_size=self.img_size,
            )
            # Split into train/val (80/20)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            # Use val transforms for val set
            self.val_dataset.dataset.transform = get_val_transforms(self.img_size)

        if stage == "test" or stage is None:
            self.test_dataset = OxfordPetSegmentation(
                root=self.data_dir,
                split="test",
                transform=get_val_transforms(self.img_size),
                img_size=self.img_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
