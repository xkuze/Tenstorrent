"""
Training script for UNet with VGG19 encoder.

Usage:
    cd ~/tenstorrent
    source .venv/bin/activate
    python -m unet.train
"""

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pathlib import Path

from unet.model import UNetVGG19
from unet.dataset import SegmentationDataModule


# Config
IMG_SIZE = 256
BATCH_SIZE = 8
MAX_EPOCHS = 5  # Reduced for CPU training; increase with GPU
LEARNING_RATE = 1e-4
DATA_DIR = "./data"
MODULE_DIR = Path(__file__).parent
SAVE_DIR = MODULE_DIR / "weights"
LOG_DIR = MODULE_DIR / "logs"


def train():
    """Train UNet model on Oxford-IIIT Pet Dataset"""
    print(f"\n{'=' * 60}")
    print("UNet-VGG19 Training - Image Segmentation")
    print(f"{'=' * 60}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'=' * 60}\n")

    # Create directories
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Data module
    dm = SegmentationDataModule(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    # Model
    model = UNetVGG19(
        num_classes=1,  # Binary segmentation
        pretrained=True,
        freeze_encoder=True,  # Freeze VGG19 encoder initially
        learning_rate=LEARNING_RATE,
    )

    print(f"Model parameters: {model.get_num_parameters():,} (trainable)")

    # Loggers
    csv_logger = CSVLogger(LOG_DIR, name="unet")
    tb_logger = TensorBoardLogger(LOG_DIR, name="tensorboard")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="unet_{epoch}_{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=5,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_dice",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        precision="16-mixed",  # Mixed precision for speed
    )

    # Train
    print("Starting training...")
    trainer.fit(model, dm)

    # Test
    print("\nRunning test evaluation...")
    test_results = trainer.test(model, dm)

    # Results
    best_dice = checkpoint_callback.best_model_score.item()
    test_dice = test_results[0]["test_dice"]
    test_iou = test_results[0]["test_iou"]

    # Save best model
    final_model_path = Path(SAVE_DIR) / "best_model.ckpt"
    trainer.save_checkpoint(final_model_path)

    print(f"\n{'=' * 60}")
    print("Training Results")
    print(f"{'=' * 60}")
    print(f"Best Val Dice:  {best_dice:.4f}")
    print(f"Test Dice:      {test_dice:.4f}")
    print(f"Test IoU:       {test_iou:.4f}")
    print(f"Model saved:    {final_model_path}")
    print(f"Logs:           {LOG_DIR}/")
    print(f"{'=' * 60}\n")

    return model, best_dice


def train_unfrozen():
    """
    Fine-tune with unfrozen encoder.
    Run this after initial training for better results.
    """
    print(f"\n{'=' * 60}")
    print("UNet-VGG19 Fine-tuning (Unfrozen Encoder)")
    print(f"{'=' * 60}\n")

    # Load best model from initial training
    checkpoint_path = Path(SAVE_DIR) / "best_model.ckpt"
    if not checkpoint_path.exists():
        print("No checkpoint found. Run initial training first.")
        return

    # Load model and unfreeze encoder
    model = UNetVGG19.load_from_checkpoint(checkpoint_path)

    # Unfreeze encoder
    for param in model.enc1.parameters():
        param.requires_grad = True
    for param in model.enc2.parameters():
        param.requires_grad = True
    for param in model.enc3.parameters():
        param.requires_grad = True
    for param in model.enc4.parameters():
        param.requires_grad = True
    for param in model.enc5.parameters():
        param.requires_grad = True

    # Lower learning rate for fine-tuning
    model.learning_rate = 1e-5

    dm = SegmentationDataModule(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="unet_finetuned_{epoch}_{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.fit(model, dm)

    # Save final model
    final_path = Path(SAVE_DIR) / "best_model_finetuned.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Fine-tuned model saved: {final_path}")


if __name__ == "__main__":
    train()
