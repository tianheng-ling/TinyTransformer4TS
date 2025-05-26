import os
import torch
import wandb
import numpy as np
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset

from config import DEVICE, loss_functions, metric_functions
from models.build_model import build_model
from utils.set_logger import setup_logger
from utils.plots import (
    plot_preds_truths,
    plot_confusion_matrix,
    plot_fault,
    plot_custom_confusion_matrix,
)
from utils.eval_metrics import get_model_complexity, get_residuals, uva_evaluate
from utils.match_dimensions import match_dimensions


def test(
    task_flag: str,
    data_flag: str,  # not used, placeholder
    test_dataset: Dataset,
    model_params: dict,
    exp_mode: str,
    batch_size: int,
    target_scaler: object,
    exp_save_dir: str,
    fig_save_dir: str,
    log_save_dir: str,
):

    # get dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    # build model and load weights
    model = build_model(model_params).to(DEVICE)
    checkpoint = torch.load(exp_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    # set up logging
    logger = setup_logger("test_logger", os.path.join(log_save_dir, "test_logfile.log"))

    # set criterion
    criterion = loss_functions[task_flag]

    # set prefix
    prefix = (
        "int_"
        if model_params["enable_qat"] == True and model_params["do_int_forward"] == True
        else ""
    )

    # calculate model complexity
    model_complexity = get_model_complexity(model_params, model, prefix)

    # test
    model.eval()
    sum_batch_losses = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for _, (features, target) in enumerate(test_dataloader):
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            pred = model(inputs=features)

            pred = match_dimensions(task_flag, pred)
            pred = pred.to(target.device)
            test_batch_loss = criterion(pred, target)
            sum_batch_losses += test_batch_loss.item()

            pred = pred.argmax(dim=1) if task_flag == "classification" else pred

            test_preds.append(pred.view(-1, 1).detach().cpu().numpy())
            test_targets.append(target.view(-1, 1).detach().cpu().numpy())

    test_loss = sum_batch_losses / len(test_dataloader)

    metrics = {f"{prefix}test_loss": test_loss}
    metrics.update(
        metric_functions[task_flag](f"{prefix}test", test_preds, test_targets)
    )

    # calculate the denomarlized test rmse, only for forecasting task
    if task_flag == "forecasting" and target_scaler is not None:
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)

        test_preds_denorm = target_scaler.inverse_transform(test_preds)
        test_targets_denorm = target_scaler.inverse_transform(test_targets)

        test_loss_denorm = criterion(
            torch.tensor(test_preds_denorm, dtype=torch.float32).to(DEVICE),
            torch.tensor(test_targets_denorm, dtype=torch.float32).to(DEVICE),
        ).item()

        metrics.update({f"{prefix}test_denorm_loss": test_loss_denorm})
        metrics.update(
            metric_functions[task_flag](
                f"{prefix}test_denorm", test_preds_denorm, test_targets_denorm
            )
        )

    # log test results
    print(f"---------------- {prefix}Test Results ----------------")
    headers = list(metrics.keys())[1:] + list(model_complexity.keys())
    row = [
        [f"{metrics[k]:.4f}" for k in list(metrics.keys())[1:]]
        + [
            (
                model_complexity[k]
                if isinstance(model_complexity[k], (int, float))
                else str(model_complexity[k])
            )
            for k in model_complexity.keys()
        ]
    ]
    logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
    if exp_mode == "train":

        wandb.log({**metrics, **model_complexity})

    if task_flag == "forecasting":
        plot_preds_truths(
            preds=test_preds,
            truths=test_targets,
            save_path=fig_save_dir,
            plot_len=batch_size,
            prefix=prefix,
        )
    elif task_flag == "classification":
        plot_confusion_matrix(
            labels=[item for sublist in test_targets for item in sublist],
            preds=[item for sublist in test_targets for item in sublist],
            save_path=fig_save_dir,
            prefix=prefix,
        )
    else:
        raise ValueError("Invalid task_flag")

    return metrics


def get_beta_threshold(
    data_flag: str,
    val_dataset: Dataset,
    data_config: dict,
    exp_mode: str,
    model_params: dict,
    exp_save_dir: str,
):
    # build model and load weights
    model = build_model(model_params).to(DEVICE)
    checkpoint = torch.load(exp_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Step 1: Find the best beta and threshold based on the validation data
    # get validation data
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    # get predictions of the validation data
    shape = (len(val_dataset), model_params["num_out_features"])
    val_preds, val_targets = np.empty(shape), np.empty(shape)

    if data_flag == "ALFAData":
        for idx, (features, target) in enumerate(val_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            pred = model(features)
            val_preds[idx] = pred.cpu().detach().numpy()
            val_targets[idx] = target.cpu().detach().numpy()

        # initialize the search range of beta
        beta_start, beta_end, beta_stepsize = data_config["beta_range"]
        num_betas = int((beta_end - beta_start) / beta_stepsize) + 2

        # find the beta and threshold from the validation data
        betas = np.zeros(num_betas, dtype="float32")
        thresholds = np.zeros(num_betas, dtype="float32")  # threshold for each beta
        threshold_errors = np.zeros(num_betas, dtype="float32")

        for idx in range(num_betas):
            # get beta and smooth beta
            betas[idx] = beta_start + beta_stepsize * idx
            smoothed_betas = get_residuals(
                val_preds,
                val_targets,
                val_dataset.files,
                val_dataset.file_lengths,
                betas[idx],
            )
            # calculate the threshold for each beta
            thresholds[idx] = np.mean(smoothed_betas) + 3.291 * np.std(
                smoothed_betas
            )  # ALFAData
            # calculate the threshold error for each beta
            threshold_errors[idx] = abs(
                thresholds[idx] - max(smoothed_betas)
            )  # smaller is better

        # get the beta and threshold with the minimum threshold error
        argmin = np.argmin(threshold_errors)
        beta = betas[argmin]
        threshold = thresholds[argmin]

    elif data_flag == "SKABData":
        valid_residuals = []
        for idx, (features, target) in enumerate(val_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            pred = model(features)
            val_preds[idx] = pred.cpu().detach().numpy()
            val_targets[idx] = target.cpu().detach().numpy()

            residuals = torch.abs(target - pred).sum(dim=1)

            # absolute residual value, sum the residuals of all features
            valid_residuals.extend(residuals.cpu().detach().numpy())

        # set threshold
        quantile = 99
        threshold = np.percentile(valid_residuals, quantile)
        beta = np.median(valid_residuals)

    else:
        raise ValueError("Invalid data_flag")

    print(f"---------------- Best Beta and Threshold ----------------")
    print(f"Beta: {beta:.4f}" if beta is not None else "Beta: N/A", end="; ")
    print(f"Threshold: {threshold:.4f}")

    if exp_mode == "train":
        wandb.log({"beta": beta, "threshold": threshold})
    return beta, threshold


def test_ad(
    task_flag: str,
    data_flag: str,
    test_dataset: Dataset,
    batch_size: int,
    target_scaler: object,  # not used, placeholder
    model_params: dict,
    beta: float,
    threshold: float,
    exp_mode: str,
    exp_save_dir: str,
    fig_save_dir: str,
    log_save_dir: str,
):
    assert task_flag == "anomaly_detection", "task_flag must be 'anomaly_detection'"

    # build model and load weights
    model = build_model(model_params).to(DEVICE)
    checkpoint = torch.load(exp_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # set up logging
    logger = setup_logger("test_logger", os.path.join(log_save_dir, "test_logfile.log"))

    # set prefix
    prefix = (
        "int_"
        if model_params["enable_qat"] == True and model_params["do_int_forward"] == True
        else ""
    )

    # calculate model complexity
    model_complexity = get_model_complexity(model_params, model, prefix)

    # Step 2: Test the model with the best beta and threshold with three types of faults
    total_fp, total_tp, total_fn, total_tn = 0, 0, 0, 0
    fault_names = [f"fault{i+1}" for i in range(len(test_dataset))]

    for dataset, name in zip(test_dataset, fault_names):
        current_threshold = threshold
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        test_preds, test_targets = [], []
        for features, target in dataloader:
            features, target = features.to(DEVICE), target.to(DEVICE)
            y_pred = model(features)
            test_preds.append(y_pred.cpu().detach().numpy())
            test_targets.append(target.cpu().detach().numpy())

        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)

        if data_flag == "ALFAData":
            residuals = get_residuals(
                test_preds, test_targets, dataset.files, dataset.file_lengths, beta
            )

        elif data_flag == "SKABData":
            test_test_targets_tensor = torch.tensor(
                test_targets, dtype=torch.float32, device=DEVICE
            )
            test_test_preds_tensor = torch.tensor(
                test_preds, dtype=torch.float32, device=DEVICE
            )

            residuals = torch.abs(
                test_test_targets_tensor - test_test_preds_tensor
            ).sum(dim=1)
            residuals = residuals.cpu().detach().numpy()
            mean_test_residuals = np.median(residuals)
            mean_val_residuals = beta
            scale = mean_test_residuals / mean_val_residuals
            alpha = 0.279
            current_threshold = current_threshold * (1 - alpha + alpha * scale)
            print("calibrated threshold: ", current_threshold)
        else:
            raise ValueError("Invalid data_flag")

        plot_fault(
            name,
            residuals,
            current_threshold,
            dataset.files,
            dataset.file_lengths,
            dataset.fault_indices,
            save_fig_path=fig_save_dir,
        )

        fp, tp, fn, tn = uva_evaluate(
            residuals,
            current_threshold,
            dataset.files,
            dataset.file_lengths,
            dataset.fault_indices,
        )
        total_fp += fp
        total_tp += tp
        total_fn += fn
        total_tn += tn

    acc = (total_tn + total_tp) / (total_tn + total_fp + total_tp + total_fn)
    pre = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * ((pre * rec) / (pre + rec)) if (pre + rec) > 0 else 0.0
    metrics = {
        f"{prefix}test_FP": int(total_fp),
        f"{prefix}test_TP": int(total_tp),
        f"{prefix}test_FN": int(total_fn),
        f"{prefix}test_FP": int(total_tn),
        f"{prefix}test_acc": f"{acc:.4f}",
        f"{prefix}test_precision": f"{pre:.4f}",
        f"{prefix}test_recall": f"{rec:.4f}",
        f"{prefix}test_f1": f"{f1:.4f}",
    }
    plot_custom_confusion_matrix(
        total_tp, total_fp, total_fn, total_tn, fig_save_dir, prefix, normalize=False
    )

    # log test results
    print(f"---------------- {prefix}Test Results ----------------")

    headers = list(metrics.keys()) + list(model_complexity.keys())
    row = [
        [metrics[k] for k in metrics.keys()]
        + [
            (
                model_complexity[k]
                if isinstance(model_complexity[k], (int, float))
                else str(model_complexity[k])
            )
            for k in model_complexity.keys()
        ]
    ]
    logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
    if exp_mode == "train":
        wandb.log({**metrics, **model_complexity})
    return metrics
