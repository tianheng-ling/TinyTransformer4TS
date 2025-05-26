import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def load_AirUData(data_config: dict):

    df = pd.read_csv(data_config["data_file_path"])
    df = df[["Timestamp"] + data_config["feature_cols"] + [data_config["target_col"]]]

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    train_data_end_idx = df.index[
        df.iloc[:, 0] == data_config["selected_split_timestamp"]
    ][0]
    return df, train_data_end_idx


def normalize_AirUData(df: pd.DataFrame, train_data_end_idx: int):

    df_train_features = df.iloc[:train_data_end_idx, 1:-1]
    df_train_target = df.iloc[:train_data_end_idx, -1:]

    features_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    features_scaler.fit(df_train_features.values)
    target_scaler.fit(df_train_target.values)

    df.iloc[:, 1:] = np.concatenate(
        (
            features_scaler.transform(df.iloc[:, 1:-1].values),
            target_scaler.transform(df.iloc[:, -1:].values),
        ),
        axis=1,
    )
    return df, target_scaler


def filter_AirUData(df: pd.DataFrame, window_size: int):

    data_filtered = []
    filtered_timestamps = []
    expected_time_interval = (window_size - 1) * 60 * 60  # seconds
    num_skipped_samples = 0
    for i in range(df.shape[0] - window_size):
        sample_data = df.iloc[i : i + window_size, :].copy()
        actual_time_interval = (
            sample_data.iloc[-1, 0] - sample_data.iloc[0, 0]
        ).total_seconds()
        if actual_time_interval != expected_time_interval:
            num_skipped_samples += 1
            continue
        data_filtered.append(sample_data.values[:, 1:])
        filtered_timestamps.append(sample_data.values[:, 0])

    return data_filtered, filtered_timestamps


def split_AirUData(data_filtered: list, filtered_timestamps: list):

    new_train_data_end_idx = 0
    for i in range(len(filtered_timestamps)):
        if str(filtered_timestamps[i][0]) == "2022-01-01 00:00:00+00:00":
            new_train_data_end_idx = i
            break

    train_data = data_filtered[0:new_train_data_end_idx]
    test_data = data_filtered[new_train_data_end_idx:]
    return train_data, test_data


class AirUDataset(Dataset):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self.data = data

    def __getitem__(self, idx: int):

        sample_data = self.data[idx]

        features = sample_data[:, :-1]
        target = np.empty((1, 1))
        target[0, 0] = np.array(sample_data[-1, -1])

        features = torch.from_numpy(features.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))

        return torch.FloatTensor(features), torch.FloatTensor(target)

    def __len__(self):
        return len(self.data)


def get_AirUDatasets(data_config: dict):
    num_in_features = data_config["num_in_features"]
    num_out_features = data_config["num_out_features"]

    df, train_data_end_idx = load_AirUData(data_config)

    df_normed, target_scaler = normalize_AirUData(df, train_data_end_idx)
    data_filtered, filtered_timestamps = filter_AirUData(
        df_normed, data_config["window_size"]
    )
    train_data, test_data = split_AirUData(data_filtered, filtered_timestamps)

    train_dataset = AirUDataset(train_data)
    val_dataset = AirUDataset(test_data)
    test_dataset = AirUDataset(test_data)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        target_scaler,
        num_in_features,
        num_out_features,
    )
