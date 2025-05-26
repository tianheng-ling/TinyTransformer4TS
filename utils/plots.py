import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_learning_curve(
    epochs: list, train_losses: list, val_losses: list, save_path: str, prefix: bool
):
    prefix = "QAT_" if prefix else ""
    save_path = os.path.join(save_path, f"{prefix}learning_curve.png")

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_losses, label=f"Train Loss")
    ax.plot(epochs, val_losses, label=f"Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_preds_truths(
    preds: list, truths: list, plot_len: int, save_path: str, prefix=None
):
    plt.plot(range(plot_len), truths[:plot_len], color="green", label="Ground Truths")
    plt.plot(range(plot_len), preds[:plot_len], color="red", label="Predictions")
    plt.ylabel("Target Value")
    plt.xlabel("Timesteps")
    plt.title("Partial preds on test set")
    plt.legend()
    plt.savefig(str(save_path) + f"/{prefix}preds_truths.png")
    plt.clf()
    plt.close()


def plot_confusion_matrix(labels, preds, save_path, prefix):
    all_labels = range(0, 5)
    cm = confusion_matrix(labels, preds, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, f"{prefix}confusion_matrix.png"))
    plt.close()


def plot_fault(
    figure_name,
    residuals,
    threshold,
    files,
    file_lengths,
    fault_indices,
    save_fig_path,
):

    fig, axs = plt.subplots(files, figsize=(5, files * 5))

    i = 0
    for f in range(files):
        fl = file_lengths[f]
        re = residuals[i : i + fl]
        th = threshold * np.ones(fl)

        axs[f].plot(re, color="blue", label="Residuals")
        axs[f].plot(th, color="orange", label="Threshold")

        fault = np.zeros(fl)
        fault[fault_indices[f] :] = threshold
        axs[f].plot(fault, color="red", label="Fault")

        axs[f].legend()
        axs[f].set_xlabel("Time")
        axs[f].set_ylabel("Residual")

        i += fl
    plt.tight_layout()
    plt.savefig(os.path.join(save_fig_path, f"{figure_name}.png"))
    plt.close()


def plot_custom_confusion_matrix(
    total_tp, total_fp, total_fn, total_tn, save_path, prefix, normalize=False
):
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # 绘制热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["Actual Normal", "Actual Anomaly"],
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.show()
    plt.savefig(os.path.join(save_path, f"{prefix}confusion_matrix.png"))
    plt.close()


def plot_heatmap(
    model_params, name, attn_weights, files, file_lengths, fault_indices, save_fig_path
):
    attn_weights = np.rot90(attn_weights, 1, (2, 3))

    ls, _, y, x = attn_weights.shape
    for l in range(ls):
        for h in range(model_params["nhead"]):
            plt.imshow(
                attn_weights[l][h],
                extent=[0, x, 0, y],
                cmap="hot",
                interpolation="nearest",
            )
            plt.colorbar()
            plt.savefig(os.path.join(save_fig_path, f"layer_{l}_head_{h}.png"))
            plt.close()


def plot_pareto(
    losses, hw_metrics, non_pareto_losses, non_pareto_hw_metrics, save_path: str
):
    plt.figure(figsize=(4, 3))

    plt.scatter(
        non_pareto_losses,
        non_pareto_hw_metrics,
        c="blue",
        label="Non-Pareto Front",
        alpha=0.6,
    )

    plt.scatter(losses, hw_metrics, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel("Val MSE")
    plt.ylabel("Energy (mW)")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")


def plot_pareto_rmse(
    losses, hw_metrics, non_pareto_losses, non_pareto_hw_metrics, save_path: str
):
    plt.figure(figsize=(4, 3))

    plt.scatter(
        non_pareto_losses,
        non_pareto_hw_metrics,
        c="blue",
        label="Non-Pareto Front",
        alpha=0.6,
    )

    plt.scatter(losses, hw_metrics, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel("Val RMSE")
    plt.ylabel("Energy (mW)")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")


def plot_pareto_acc(
    losses, hw_metrics, non_pareto_losses, non_pareto_hw_metrics, save_path: str
):
    plt.figure(figsize=(4, 3))

    plt.scatter(
        non_pareto_losses,
        non_pareto_hw_metrics,
        c="blue",
        label="Non-Pareto Front",
        alpha=0.6,
    )

    plt.scatter(losses, hw_metrics, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel("Val Acc")
    plt.ylabel("Energy (mW)")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
