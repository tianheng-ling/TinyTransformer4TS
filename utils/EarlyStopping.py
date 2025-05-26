import numpy as np
import torch
import os


class EarlyStopping:

    def __init__(self, patience, verbose, delta, path, trace_func):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = os.path.join(path, "best_model.pth")
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # score = -val_loss
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
