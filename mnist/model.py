import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy


class MLP(L.LightningModule):
    def __init__(
        self,
        hidden_sizes=[256, 128],
        dropout_rate=0.2,
        learning_rate=1e-3,
        optimizer_name="Adam",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_size = 784  # 28x28 flattened
        self.output_size = 10  # digits 0-9

        # Build network layers
        layers = []
        prev_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # TODO: try different activation functions (LeakyReLU, ELU, GELU)
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.output_size))

        self.network = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        return self.network(x)

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
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=0.9
            )
        else:  # RMSprop
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.hparams.learning_rate
            )

        return optimizer

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
