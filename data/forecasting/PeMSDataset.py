import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_PeMSData(data_config):

    data = np.loadtxt(data_config["data_file_path"], delimiter=",", dtype=np.float32)
    selected_data = data[data_config["sensor_idx"]]

    return selected_data


def normalize_PeMSData(data, data_config):

    n_three_week_datapoints = data_config["n_three_week_datapoints"]
    train_data = data[:n_three_week_datapoints]

    scaler = StandardScaler()  # univariate
    scaler.fit(train_data.reshape(-1, 1))
    data_normed = scaler.transform(data.reshape(-1, 1)).reshape(-1)
    return data_normed, scaler


def split_PeMSData(data, data_config):

    n_three_week_datapoints = data_config["n_three_week_datapoints"]
    train_data = data[:n_three_week_datapoints]
    test_data = data[n_three_week_datapoints:]
    return train_data, test_data


class PeMSDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
    ):
        self.data = data
        self.window_size = window_size

    def __getitem__(self, idx: int):

        features = self.data[idx : idx + self.window_size]
        features = features.reshape((self.window_size, 1))
        target = self.data[idx + self.window_size]
        target = target.reshape((1, 1))

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return features, target

    def __len__(self) -> int:
        return len(self.data) - self.window_size


def get_PeMSDatasets(data_config: dict):

    num_in_features = data_config["num_in_features"]
    num_out_features = data_config["num_out_features"]

    data = load_PeMSData(data_config)
    data_normed, target_scaler = normalize_PeMSData(data, data_config)
    train_data, test_data = split_PeMSData(data_normed, data_config)

    train_dataset = PeMSDataset(train_data, data_config["window_size"])
    val_dataset = PeMSDataset(test_data, data_config["window_size"])
    test_dataset = PeMSDataset(test_data, data_config["window_size"])

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        target_scaler,
        num_in_features,
        num_out_features,
    )
