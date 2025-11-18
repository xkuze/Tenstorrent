import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

from mnist.model import MLP
from mnist.utils import MNISTDataModule


# Config
N_TRIALS = 5
EPOCHS_PER_TRIAL = 5
FINAL_EPOCHS = 20
DATA_DIR = "./data"
SAVE_DIR = "weights_mnist"


def objective(trial: optuna.Trial):
    """Optuna - tries different hyperparameters and returns validation accuracy"""
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    hidden_sizes = []
    for i in range(n_hidden_layers):
        hidden_size = trial.suggest_categorical(f"hidden_size_{i}", [64, 128, 256, 512])
        hidden_sizes.append(hidden_size)

    dropout = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=batch_size)

    model = MLP(
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout,
        learning_rate=lr,
        optimizer_name=optimizer,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename=f"trial_{trial.number}_{{epoch}}_{{val_acc:.2f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")

    trainer = L.Trainer(
        max_epochs=EPOCHS_PER_TRIAL,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, pruning_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, dm)
    # print(f"Trial {trial.number} finished with val_acc: {trainer.callback_metrics['val_acc']}")

    return trainer.callback_metrics["val_acc"].item()


def train_best_model(best_params):
    """Train final model with best params from Optuna search"""
    n_hidden_layers = best_params["n_hidden_layers"]
    hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(n_hidden_layers)]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    optimizer_name = best_params["optimizer"]
    # print(f"Debug - best_params: {best_params}")

    print(
        f"\nTraining final model: {hidden_sizes}, dropout={dropout_rate:.2f}, lr={learning_rate:.6f}, batch={batch_size}, opt={optimizer_name}"
    )

    dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=batch_size)

    model = MLP(
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="best_model_{epoch}_{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max",
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=FINAL_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=True,
    )

    trainer.fit(model, dm)
    test_results = trainer.test(model, dm)

    best_val_acc = checkpoint_callback.best_model_score.item()
    test_acc = test_results[0]["test_acc"]

    final_model_path = Path(SAVE_DIR) / "best_model.ckpt"
    trainer.save_checkpoint(final_model_path)

    print(
        f"\nResults: Val Acc={best_val_acc * 100:.2f}%, Test Acc={test_acc * 100:.2f}%, Model saved to {final_model_path}"
    )

    return model, best_val_acc


def run_hyperparameter_search():
    """Run Optuna search to find best hyperparameters"""
    print(f"\nOptuna search: {N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs each")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(
        lambda trial: objective(trial),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(
        f"\nBest trial #{study.best_trial.number}: Val Acc={study.best_value * 100:.2f}%"
    )

    return study.best_params


if __name__ == "__main__":
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    best_params = run_hyperparameter_search()
    final_model, final_accuracy = train_best_model(best_params)
