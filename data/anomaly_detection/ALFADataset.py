# Implementation inspired by https://github.com/superhumangod/Model-free-unsupervised-anomaly-detection/tree/master

import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


def process_df(data_config: dict, df: pd.DataFrame):

    data = np.zeros((len(df), data_config["num_in_features"]), dtype="float32")

    # extract the 10 status features
    data[:, 0:3] = (
        df[["vel_meas_x", "vel_meas_y", "vel_meas_z"]].astype("float32").to_numpy()
    )
    data[:, 3:7] = R.from_euler(
        "xyz",  # roll, pitch, yaw to euler angles
        df[["roll_meas", "pitch_meas", "yaw_meas"]].astype("float32").to_numpy(),
        degrees=True,
    ).as_quat()
    data[:, 7:10] = (
        df[
            [
                "ang_vel_meas_x",
                "ang_vel_meas_y",
                "ang_vel_meas_z",
            ]
        ]
        .astype("float32")
        .to_numpy()
    )

    # extract 7 command features
    data[:, 10:13] = (
        df[
            [
                "vel_com_x",
                "vel_com_y",
                "vel_com_z",
            ]
        ]
        .astype("float32")
        .to_numpy()
    )
    com = df[["roll_com", "pitch_com", "yaw_com"]].astype("float32").to_numpy()
    data[:, 13:17] = R.from_euler("xyz", com, degrees=True).as_quat()

    # find the fault index and skip some initial data
    fault_time = df["fault_time"][0]
    fault_idx = (
        len(df[df["time"] < fault_time]) - data_config["skip"] if fault_time != 0 else 0
    )
    data = data[data_config["skip"] :, :]

    return data, fault_idx


class FlightDataset(Dataset):
    def __init__(
        self, data_config: dict, wdata: np.array, files, file_lengths, fault_indices
    ):
        self.data = torch.tensor(wdata)
        self.window_size = data_config["window_size"]
        self.num_out_features = data_config["num_out_features"]
        self.files = files
        self.file_lengths = file_lengths
        self.fault_indices = fault_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        # 10 status features + 7 command features
        features = data[: self.window_size, :]
        # 10 status features
        targets = data[self.window_size, : self.num_out_features]

        return features, targets


def make_dataset(data_config, files, data_mean, data_std):
    full_wdata = None
    file_lengths = np.zeros(len(files), dtype=int)
    fault_indices = np.zeros(len(files), dtype=int)
    for f, file in enumerate(files):

        # parse the data
        df = pd.read_csv(file)
        data, fault_idx = process_df(data_config, df)

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

    return FlightDataset(
        data_config, full_wdata, len(files), file_lengths, fault_indices
    )


def load_data_files(data_config: dict, category: str) -> list[str]:
    data_dir = os.path.join(data_config["data_file_path"], category)
    return glob.glob(os.path.join(data_dir, "*"))


def compute_mean_std(
    data_config: dict, files: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    data_list = [process_df(data_config, pd.read_csv(file))[0] for file in files]
    normal_data = np.vstack(data_list)
    return np.mean(normal_data, axis=0), np.std(normal_data, axis=0)


def get_ALFADatasets(data_config: dict):
    # load all data files
    data_categories = ["normal", "valid", "engine", "elevator", "rudder"]
    data_files = {cat: load_data_files(data_config, cat) for cat in data_categories}

    # compute mean and std for normal data
    data_mean, data_std = compute_mean_std(data_config, data_files["normal"])

    # create datasets
    datasets = {
        cat: make_dataset(data_config, data_files[cat], data_mean, data_std)
        for cat in data_categories
    }
    test_dataset = [datasets["engine"], datasets["elevator"], datasets["rudder"]]

    return (
        datasets["normal"],  # training data
        datasets["valid"],  # validation data
        test_dataset,
        None,  # target_scaler (not used)
        data_config["num_in_features"],
        data_config["num_out_features"],
    )
