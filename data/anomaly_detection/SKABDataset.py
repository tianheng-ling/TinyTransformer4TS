import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def process_df(data_config, df: pd.DataFrame, mode: str):

    # data = np.zeros((len(df), data_config["num_in_features"]), dtype="float32")

    # extract the 8 status features
    selected_df = df[
        [
            "Accelerometer1RMS",
            "Accelerometer2RMS",
            "Current",
            "Pressure",
            "Temperature",
            "Thermocouple",
            "Voltage",
            "Volume Flow RateRMS",
        ]
    ]
    data = selected_df.astype("float32").to_numpy()

    # find out the index, where df["changepoint"]'s value are 1

    if mode == "normal" or mode == "valid":
        fault_start_idx = len(df)
        fault_end_idx = len(df)
    else:
        changepoint_idx_list = df[df["changepoint"] == 1.0].index.tolist()
        fault_start_idx = changepoint_idx_list[0]
        fault_end_idx = changepoint_idx_list[1]  # only cath the first fault

    data = data[:fault_end_idx, :]
    return data, fault_start_idx


# Dataset
class SKABDataset(Dataset):
    def __init__(
        self, data_config: dict, wdata: np.array, files, file_lengths, fault_indices
    ):
        self.data = torch.tensor(wdata)
        self.window_size = data_config["window_size"]
        self.num_in_features = data_config["num_in_features"]
        self.num_out_features = data_config["num_out_features"]
        self.files = files
        self.file_lengths = file_lengths
        self.fault_indices = fault_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        # 8 status features
        features = data[: self.window_size, : self.num_in_features]

        # 1 target feature
        targets = data[self.window_size, -self.num_out_features :]

        return features, targets


def make_dataset(data_config, files, data_mean, data_std, mode):
    full_wdata = None
    file_lengths = np.zeros(len(files), dtype=int)
    fault_indices = np.zeros(len(files), dtype=int)
    for f, file in enumerate(files):

        # parse the data
        df = pd.read_csv(file, sep=";")
        data, fault_idx = process_df(data_config, df, mode)

        # normalize the data
        data = (data - data_mean) / data_std

        # calculate the window size
        length = len(data)
        size = data_config["window_size"] + 1  # history + one step ahead

        # calculate the valid data length and the fault index
        file_lengths[f] = length - size + 1
        fault_indices[f] = fault_idx - size + 1 if fault_idx != 0 else 0

        # create windows on each flight themself to not create any overlap
        shape = (file_lengths[f], size, data_config["num_in_features"])

        wdata = np.empty(shape, dtype="float32")
        for i in range(length - size + 1):
            wdata[i] = data[i : i + size]

        if full_wdata is None:
            full_wdata = wdata
        else:
            full_wdata = np.append(full_wdata, wdata, axis=0)

    return SKABDataset(data_config, full_wdata, len(files), file_lengths, fault_indices)


def load_data_files(data_config: dict, category: str) -> list[str]:
    data_dir = os.path.join(data_config["data_file_path"], category)
    return glob.glob(os.path.join(data_dir, "*"))


def compute_mean_std(
    data_config: dict, files: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    data_list = [
        process_df(data_config, pd.read_csv(file, sep=";"), mode="normal")[0]
        for file in files
    ]
    normal_data = np.vstack(data_list)
    return np.mean(normal_data, axis=0), np.std(normal_data, axis=0)


def get_SKABDatasets(data_config: dict):
    # load all data files
    data_categories = ["normal", "valid", "valve1", "valve2"]  # , "other"]
    data_files = {cat: load_data_files(data_config, cat) for cat in data_categories}

    # compute mean and std for normal data
    data_mean, data_std = compute_mean_std(data_config, data_files["normal"])

    # create datasets
    datasets = {
        cat: make_dataset(data_config, data_files[cat], data_mean, data_std, cat)
        for cat in data_categories
    }
    test_dataset = [datasets["valve1"], datasets["valve2"]]  # , datasets["other"]]

    return (
        datasets["normal"],  # training data
        datasets["valid"],  # validation data
        test_dataset,
        None,  # target_scaler (not used)
        data_config["num_in_features"],
        data_config["num_out_features"],
    )
