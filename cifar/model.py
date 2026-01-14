import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy


class CNN(L.LightningModule):
    def __init__(
        self,
        num_filters=[32, 64, 128],
        kernel_size=3,
        dropout_rate=0.2,
        learning_rate=1e-3,
        optimizer_name="Adam",
    ):
        super().__init__()

        self.save_hyperparameters()

        # CIFAR-10: 32x32x3 RGB images
        self.input_channels = 3
        self.output_size = 10

        # Build conv layers
        layers = []
        in_channels = self.input_channels

        for out_channels in num_filters:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            # TODO: try BatchNorm2d here for better convergence
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate size after conv layers
        # 32 -> 16 -> 8 -> 4
        feature_size = 32 // (2 ** len(num_filters))
        flatten_size = num_filters[-1] * feature_size * feature_size

        # Build connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.output_size),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy)

        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=0.9
            )
        else:  # RMSprop
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
