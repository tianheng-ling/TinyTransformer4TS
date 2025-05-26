import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# source https://github.com/bartkowiaktomasz/har-wisdm-lstm-rnns

INPUT_FEATURES = ["x-axis", "y-axis", "z-axis"]
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}


def load_WISDM_raw(file_path):
    column_names = ["user", "activity", "timestamp"] + INPUT_FEATURES
    df = pd.read_csv(file_path, header=None, names=column_names)

    df["z-axis"] = df["z-axis"].str.replace(";", "", regex=False)
    df.dropna(inplace=True)
    for col in INPUT_FEATURES:
        df[col] = df[col].astype(np.float32)
    return df


def preprocess_WISDM(df, window_size, slide_step):
    segments = []
    labels = []

    for i in range(0, len(df) - window_size, slide_step):
        xs = df["x-axis"].values[i : i + window_size]
        ys = df["y-axis"].values[i : i + window_size]
        zs = df["z-axis"].values[i : i + window_size]

        if len(xs) == window_size:
            segments.append([xs, ys, zs])
            label = df["activity"][i : i + window_size].mode()[0]
            labels.append(LABEL_MAP[label])

    X = np.asarray(segments, dtype=np.float32).transpose(0, 2, 1)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


class WISDM_Dataset(Dataset):
    def __init__(self, data_array, label_array, downsample_factor):
        self.X = torch.tensor(data_array, dtype=torch.float32)
        self.y = torch.tensor(label_array, dtype=torch.long)
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = x[:: self.downsample_factor]
        return x, self.y[idx]


def balance_samples(X, y, max_per_class):
    X_balanced = []
    y_balanced = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        if len(indices) > max_per_class:
            indices = np.random.choice(indices, max_per_class, replace=False)
        X_balanced.append(X[indices])
        y_balanced.append(y[indices])

    X_new = np.concatenate(X_balanced, axis=0)
    y_new = np.concatenate(y_balanced, axis=0)

    indices = np.random.permutation(len(X_new))
    return X_new[indices], y_new[indices]


def get_WISDMDatasets(data_config: dict):
    """
    data_config["data_file_path"]: Path to WISDM_ar_v1.1_raw.txt
    """
    df = load_WISDM_raw(data_config["data_file_path"])
    data_x, data_y = preprocess_WISDM(
        df, data_config["window_size"], data_config["slide_step"]
    )

    data_x, data_y = balance_samples(data_x, data_y, data_config["max_per_class"])

    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=data_config["test_size"]
    )

    train_dataset = WISDM_Dataset(X_train, y_train, data_config["downsample_rate"])
    val_dataset = WISDM_Dataset(X_test, y_test, data_config["downsample_rate"])
    test_dataset = WISDM_Dataset(X_test, y_test, data_config["downsample_rate"])

    num_in_features = data_x.shape[2]
    num_out_features = len(LABELS)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        None,
        num_in_features,
        num_out_features,
    )
