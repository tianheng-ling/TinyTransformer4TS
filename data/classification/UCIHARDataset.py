import os
import numpy as np
import torch
from torch.utils.data import Dataset

# source https://github.com/arijitiiest/UCI-Human-Activity-Recognition

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
]
# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_UCIHARdata_features(paths):

    X_signals = []
    for signal_type_path in paths:
        if not os.path.exists(signal_type_path):
            raise FileNotFoundError(f"File {signal_type_path} not found.")
        file = open(signal_type_path, "r")
        X_signals.append(
            [
                np.array(serie, dtype=np.float32)
                for serie in [row.replace("  ", " ").strip().split(" ") for row in file]
            ]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_UCIHARdata_target(y_path):
    file = open(y_path, "r")
    y_ = np.array(
        [elem for elem in [row.replace("  ", " ").strip().split(" ") for row in file]],
        dtype=np.int32,
    )
    file.close()
    return y_ - 1


class UCIHAR_Dataset(Dataset):
    def __init__(self, data_config, data_type):

        X_signals_path = [
            data_config["data_file_path"]
            + f"{data_type}/"
            + "inertial_signals/"
            + signal
            + f"{data_type}.txt"
            for signal in INPUT_SIGNAL_TYPES
        ]
        y_train_path = (
            data_config["data_file_path"] + f"{data_type}/" + "y_" + f"{data_type}.txt"
        )

        self.X = load_UCIHARdata_features(X_signals_path)
        self.y = load_UCIHARdata_target(y_train_path)
        self.downsample_factor = data_config["downsample_rate"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = x[:: self.downsample_factor]
        x = torch.tensor(x, dtype=torch.float32)
        y = int(self.y[idx])
        return x, y


def get_UCIHARDatasets(data_config: dict):
    train_dataset = UCIHAR_Dataset(data_config, "train")
    val_dataset = UCIHAR_Dataset(data_config, "test")
    test_dataset = UCIHAR_Dataset(data_config, "test")

    num_in_features = data_config["num_in_features"]
    num_out_features = data_config["num_out_features"]

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        None,
        num_in_features,
        num_out_features,
    )
