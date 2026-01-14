"""
UNet with VGG19 encoder for image segmentation.
Input: 256x256 RGB images
Output: Binary or multi-class segmentation mask
"""

import torch
import torch.nn as nn
import lightning as L
from torchvision import models
from torchmetrics import F1Score, JaccardIndex


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block: Upsample -> Concat skip -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetVGG19(L.LightningModule):
    """
    UNet with VGG19 encoder.

    Architecture:
    - Encoder: VGG19 pretrained on ImageNet (feature extraction at 5 scales)
    - Decoder: 5-stage upsampling path with skip connections
    - Output: 1x1 conv to num_classes
    """

    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        freeze_encoder=True,
        learning_rate=1e-4,
        bilinear=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load VGG19 encoder
        vgg19 = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        )
        features = list(vgg19.features.children())

        # Encoder blocks (extract features at different scales)
        # VGG19 structure: Conv blocks separated by MaxPool
        # Block 1: 0-3 (64 channels) -> 256x256 -> 128x128
        # Block 2: 5-8 (128 channels) -> 128x128 -> 64x64
        # Block 3: 10-17 (256 channels) -> 64x64 -> 32x32
        # Block 4: 19-26 (512 channels) -> 32x32 -> 16x16
        # Block 5: 28-35 (512 channels) -> 16x16 -> 8x8

        self.enc1 = nn.Sequential(*features[0:4])  # 64 ch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(*features[5:9])  # 128 ch
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(*features[10:18])  # 256 ch
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(*features[19:27])  # 512 ch
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc5 = nn.Sequential(*features[28:36])  # 512 ch

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.enc1.parameters():
                param.requires_grad = False
            for param in self.enc2.parameters():
                param.requires_grad = False
            for param in self.enc3.parameters():
                param.requires_grad = False
            for param in self.enc4.parameters():
                param.requires_grad = False
            for param in self.enc5.parameters():
                param.requires_grad = False

        # Bridge
        self.bridge = DoubleConv(512, 1024)

        # Decoder blocks
        self.up5 = UpBlock(1024 + 512, 512, bilinear)
        self.up4 = UpBlock(512 + 512, 256, bilinear)
        self.up3 = UpBlock(256 + 256, 128, bilinear)
        self.up2 = UpBlock(128 + 128, 64, bilinear)
        self.up1 = UpBlock(64 + 64, 64, bilinear)

        # Final output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # Loss and metrics (F1Score = Dice for binary classification)
        self.dice_metric = F1Score(
            task="binary" if num_classes == 1 else "multiclass",
            num_classes=num_classes if num_classes > 1 else None,
        )
        self.iou_metric = JaccardIndex(
            task="binary" if num_classes == 1 else "multiclass",
            num_classes=num_classes if num_classes > 1 else None,
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 64 ch, 256x256
        e2 = self.enc2(self.pool1(e1))  # 128 ch, 128x128
        e3 = self.enc3(self.pool2(e2))  # 256 ch, 64x64
        e4 = self.enc4(self.pool3(e3))  # 512 ch, 32x32
        e5 = self.enc5(self.pool4(e4))  # 512 ch, 16x16

        # Bridge
        b = self.bridge(self.pool4(e5))  # 1024 ch, 8x8

        # Decoder
        d5 = self.up5(b, e5)  # 512 ch, 16x16
        d4 = self.up4(d5, e4)  # 256 ch, 32x32
        d3 = self.up3(d4, e3)  # 128 ch, 64x64
        d2 = self.up2(d3, e2)  # 64 ch, 128x128
        d1 = self.up1(d2, e1)  # 64 ch, 256x256

        # Output
        out = self.final(d1)
        return out

    def compute_loss(self, logits, masks):
        """Compute BCE + Dice loss"""
        if self.num_classes == 1:
            # Binary segmentation - ensure masks have channel dimension
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)
            dice_loss = self._dice_loss(torch.sigmoid(logits), masks)
        else:
            # Multi-class segmentation
            bce_loss = nn.functional.cross_entropy(logits, masks.long().squeeze(1))
            dice_loss = self._dice_loss(
                torch.softmax(logits, dim=1), masks.long().squeeze(1)
            )
        return bce_loss + dice_loss

    def _dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for binary segmentation"""
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        # Metrics
        if self.num_classes == 1:
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # [B, H, W]
            masks_int = (
                masks.long() if masks.dim() == 3 else masks.long().squeeze(1)
            )  # [B, H, W]
        else:
            preds = logits.argmax(1)
            masks_int = masks.long().squeeze(1)

        dice = self.dice_metric(preds, masks_int)
        iou = self.iou_metric(preds, masks_int)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        if self.num_classes == 1:
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # [B, H, W]
            masks_int = (
                masks.long() if masks.dim() == 3 else masks.long().squeeze(1)
            )  # [B, H, W]
        else:
            preds = logits.argmax(1)
            masks_int = masks.long().squeeze(1)

        dice = self.dice_metric(preds, masks_int)
        iou = self.iou_metric(preds, masks_int)

        self.log("test_loss", loss)
        self.log("test_dice", dice)
        self.log("test_iou", iou)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
