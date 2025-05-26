import os
import torch
import wandb
import logging
import numpy as np
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

from config import DEVICE, loss_functions, metric_functions
from models.build_model import build_model
from utils.set_logger import setup_logger
from utils.EarlyStopping import EarlyStopping
from utils.plots import plot_learning_curve
from utils.match_dimensions import match_dimensions
from utils import analyze_model_memory


def train_val(
    task_flag: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_params: dict,
    batch_size: int,
    target_scaler: object,
    lr: float,
    num_epochs: int,
    exp_save_dir: str,
    fig_save_dir: str,
    log_save_dir: str,
):

    # get dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    print("Number of samples in train_dataset:", len(train_dataset))
    print("Number of samples in val_dataset:", len(val_dataset))

    # build model
    model = build_model(model_params).to(DEVICE)
    wandb.log(model_params)

    # set up logging
    logger = setup_logger(
        "train_val_logger", os.path.join(log_save_dir, "train_val_logfile.log")
    )

    # set up criterion and optimizer
    criterion = loss_functions[task_flag]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    early_stopping = EarlyStopping(
        patience=10,
        verbose=True,
        delta=0,
        path=exp_save_dir,
        trace_func=print,
    )

    # train and validate
    all_train_epoch_losses = []
    all_val_epoch_losses = []
    all_val_epoch_rmse_denorms = []
    all_val_epoch_acc_scores = []  # only for classification task

    for epoch in range(num_epochs):

        # training phase
        model.train()
        sum_train_batch_losses = 0
        train_preds, train_targets = [], []

        for train_features, train_target in train_dataloader:

            train_features = train_features.to(DEVICE)
            train_target = train_target.to(DEVICE)
            train_pred = model(inputs=train_features)
            train_pred = match_dimensions(task_flag, train_pred)

            train_batch_loss = criterion(train_pred, train_target)
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
            sum_train_batch_losses += train_batch_loss.item()

            train_pred = (
                train_pred.argmax(dim=1)
                if task_flag == "classification"
                else train_pred
            )
            train_preds.append(train_pred.detach().cpu().numpy())
            train_targets.append(train_target.detach().cpu().numpy())
        train_epoch_loss = sum_train_batch_losses / len(train_dataloader)
        all_train_epoch_losses.append(train_epoch_loss)

        # validation phase
        model.eval()
        sum_val_batch_losses = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for val_features, val_target in val_dataloader:
                val_features = val_features.to(DEVICE)
                val_target = val_target.to(DEVICE)
                val_pred = model(inputs=val_features)

                val_pred = match_dimensions(task_flag, val_pred)
                val_batch_loss = criterion(val_pred, val_target)
                sum_val_batch_losses += val_batch_loss.item()

                val_pred = (
                    val_pred.argmax(dim=1)
                    if task_flag == "classification"
                    else val_pred
                )
                val_preds.append(val_pred.detach().cpu().numpy())
                val_targets.append(val_target.detach().cpu().numpy())

        val_epoch_loss = sum_val_batch_losses / len(val_dataloader)
        all_val_epoch_losses.append(val_epoch_loss)

        # Early stopping check
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            logging.info("executed_valid_epochs: {}".format(epoch - 10))
            break

        # calculate the denomarlized val rmse, only for forecasting task
        if task_flag == "forecasting" and target_scaler is not None:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            val_preds = val_preds.reshape(-1, val_preds.shape[-1])
            val_targets = val_targets.reshape(-1, val_targets.shape[-1])

            val_preds_denorm = target_scaler.inverse_transform(val_preds)
            val_targets_denorm = target_scaler.inverse_transform(val_targets)

            val_epoch_loss_denorm = criterion(
                torch.tensor(val_preds_denorm, dtype=torch.float32).to(DEVICE),
                torch.tensor(val_targets_denorm, dtype=torch.float32).to(DEVICE),
            ).item()
            val_epoch_rmse_denorm = np.sqrt(val_epoch_loss_denorm)
            all_val_epoch_rmse_denorms.append(val_epoch_rmse_denorm)

        # Compute evaluation metrics
        metrics = {"train_loss": train_epoch_loss, "val_loss": val_epoch_loss}
        for phase in ["train", "val"]:
            metrics.update(
                metric_functions[task_flag](
                    phase,
                    preds=train_preds if phase == "train" else val_preds,
                    targets=train_targets if phase == "train" else val_targets,
                )
            )
        if task_flag == "classification":
            all_val_epoch_acc_scores.append(metrics["val_acc"])

        # Log metrics
        headers = ["epoch", "train_loss", "val_loss"] + list(metrics.keys())[2:]
        row = [
            [epoch + 1, f"{train_epoch_loss:.4f}", f"{val_epoch_loss:.4f}"]
            + [f"{metrics[k]:.4f}" for k in list(metrics.keys())[2:]]
        ]
        logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
        wandb.log(metrics)

    wandb.log({"timestamp": str(exp_save_dir).split("/")[-1]})
    print("Training finished. Please find the checkpoint at:", exp_save_dir)

    # nessessary for optuna search
    best_val_loss = min(all_val_epoch_losses)
    if task_flag == "forecasting":
        best_val_rmse_denorm = min(all_val_epoch_rmse_denorms)
        pareto_acc = best_val_rmse_denorm
    elif task_flag == "classification":
        pareto_acc = max(all_val_epoch_acc_scores)
    elif task_flag == "anomaly_detection":
        pareto_acc = None
    weights_mem, activations_mem, total_macs = analyze_model_memory(
        model_params=model_params, model=model
    )

    plot_learning_curve(
        epochs=range(1, len(all_train_epoch_losses) + 1),
        train_losses=all_train_epoch_losses,
        val_losses=all_val_epoch_losses,
        save_path=fig_save_dir,
        prefix=model_params["enable_qat"],
    )

    return best_val_loss, pareto_acc, weights_mem, activations_mem, total_macs
