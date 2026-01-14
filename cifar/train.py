import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pathlib import Path
import json
from datetime import datetime

from cifar.model import CNN
from cifar.utils import CIFAR10DataModule


# Config
N_TRIALS = 20  # Increased for better search
EPOCHS_PER_TRIAL = 5
FINAL_EPOCHS = 30
DATA_DIR = "./data"
MODULE_DIR = Path(__file__).parent
SAVE_DIR = MODULE_DIR / "weights"
LOG_DIR = MODULE_DIR / "logs"


def objective(trial: optuna.Trial):
    """Optuna - tries different hyperparameters and returns validation accuracy"""
    n_conv_layers = trial.suggest_int("n_conv_layers", 2, 4)
    num_filters = []
    for i in range(n_conv_layers):
        filters = trial.suggest_categorical(f"num_filters_{i}", [32, 64, 128, 256])
        num_filters.append(filters)

    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dropout = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    dm = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=batch_size)

    model = CNN(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout,
        learning_rate=lr,
        optimizer_name=optimizer,
    )

    # Logger for each trial
    csv_logger = CSVLogger(LOG_DIR, name=f"trial_{trial.number}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename=f"trial_{trial.number}_{{epoch}}_{{val_acc:.4f}}",
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
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=csv_logger,
    )

    trainer.fit(model, dm)

    val_acc = trainer.callback_metrics["val_acc"].item()
    print(
        f"  Trial {trial.number}: filters={num_filters}, kernel={kernel_size}, "
        f"dropout={dropout:.2f}, lr={lr:.6f}, opt={optimizer} -> val_acc={val_acc * 100:.2f}%"
    )

    return val_acc


def train_best_model(best_params, trial_number=0):
    """Train final model with best params from Optuna search"""
    n_conv_layers = best_params["n_conv_layers"]
    num_filters = [best_params[f"num_filters_{i}"] for i in range(n_conv_layers)]
    kernel_size = best_params["kernel_size"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    optimizer_name = best_params["optimizer"]

    print(
        f"\nTraining final model: filters={num_filters}, kernel={kernel_size}, "
        f"dropout={dropout_rate:.2f}, lr={learning_rate:.6f}, batch={batch_size}, "
        f"opt={optimizer_name}"
    )

    dm = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=batch_size)

    model = CNN(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )

    # Loggers
    csv_logger = CSVLogger(LOG_DIR, name="final_model")
    tb_logger = TensorBoardLogger(LOG_DIR, name="tensorboard")

    # Save top-20 best checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="top20_{epoch}_{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=20,  # Save top-20 models
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=FINAL_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    test_results = trainer.test(model, dm)

    best_val_acc = checkpoint_callback.best_model_score.item()
    test_acc = test_results[0]["test_acc"]

    # Save the best model separately
    final_model_path = Path(SAVE_DIR) / "best_model.ckpt"
    trainer.save_checkpoint(final_model_path)

    # Save parameters to JSON
    params_path = Path(SAVE_DIR) / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "num_filters": num_filters,
                "kernel_size": kernel_size,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "val_acc": best_val_acc,
                "test_acc": test_acc,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Val Acc:  {best_val_acc * 100:.2f}%")
    print(f"Test Acc: {test_acc * 100:.2f}%")
    print(f"Model:    {final_model_path}")
    print(f"Params:   {params_path}")
    print(f"Logs:     {LOG_DIR}/")
    print(f"Top-20:   {SAVE_DIR}/top20_*.ckpt")
    print(f"{'=' * 60}")

    return model, best_val_acc


def run_hyperparameter_search():
    """Run Optuna search to find best hyperparameters"""
    print(f"\n{'=' * 60}")
    print("Optuna Hyperparameter Search - CIFAR-10 CNN")
    print(f"{'=' * 60}")
    print(f"Trials: {N_TRIALS}, Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"{'=' * 60}\n")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(
        lambda trial: objective(trial),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Output top-20 best trials
    print(f"\n{'=' * 60}")
    print("Top 20 Trials")
    print(f"{'=' * 60}")

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, trial in enumerate(sorted_trials[:20]):
        if trial.value:
            print(f"{i + 1:2d}. Trial #{trial.number}: val_acc={trial.value * 100:.2f}%")

    print(f"\nBest trial #{study.best_trial.number}: Val Acc={study.best_value * 100:.2f}%")

    # Save results of all trials
    trials_path = Path(SAVE_DIR) / "all_trials.json"
    trials_data = []
    for trial in study.trials:
        if trial.value:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                }
            )

    with open(trials_path, "w") as f:
        json.dump(sorted(trials_data, key=lambda x: x["value"], reverse=True), f, indent=2)

    print(f"All trials saved to: {trials_path}")

    return study.best_params


if __name__ == "__main__":
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    best_params = run_hyperparameter_search()
    final_model, final_accuracy = train_best_model(best_params)
