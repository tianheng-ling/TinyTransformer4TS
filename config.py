import torch
import torch.nn as nn
from utils.eval_metrics import (
    get_forecasting_metrics,
    get_classification_metrics,
    compute_anomaly_metrics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AirUData_config = {
    "data_name": "AirUData",
    "task_flag": "forecasting",
    "data_file_path": "data/forecasting/AirUData/airU.csv",
    "feature_cols": ["PM1", "PM2_5", "PM10", "Temperature", "Humidity", "RED", "NOX"],
    "target_col": "OZONE",
    "selected_split_timestamp": "2022-01-01 00:00:00+00:00",
    "num_in_features": 7,
    "num_out_features": 1,
    "window_size": 24,
}

PeMSData_config = {
    "data_name": "PeMSData",
    "task_flag": "forecasting",
    "data_file_path": "data/forecasting/PeMSData/pems-4w.csv",
    "sensor_idx": 4291,
    "n_three_week_datapoints": 24 * 7 * 3 * 12,
    "num_in_features": 1,
    "num_out_features": 1,
    "window_size": 24,
}

UCIHARData_config = {
    "data_name": "UCIHARData",
    "task_flag": "classification",
    "data_file_path": "data/classification/UCIHARData/",
    "num_in_features": 9,
    "num_out_features": 6,
    "window_size": 128,
    "downsample_rate": 4,
}

WISDMData_config = {
    "data_name": "WISDMData",
    "task_flag": "classification",
    "data_file_path": "data/classification/WISDMData/WISDM_ar_v1.1_raw_cleaned.txt",
    "window_save_path": "data/classification/WISDMData_sliding_window",
    "num_in_features": 3,
    "num_out_features": 6,
    "test_size": 0.3,  # sampling_frequency 20Hz
    "window_size": 200,
    "slide_step": 100,
    "max_per_class": 1600,  # to save training time
    "downsample_rate": 4,
}

ALFAData_config = {
    "data_name": "ALFAData",
    "task_flag": "anomaly_detection",
    "data_file_path": "data/anomaly_detection/ALFAData",  # Data directory path
    "processed_d": "processed",  # Processed data folder in data directory
    "num_in_features": 17,  # 10 status features + 7 command features
    "num_out_features": 10,  #  10 status features
    "skip": 32,  # The amount of data points skipped
    "beta_range": (0.749, 0.971, 0.001),  # The range of beta values.
    "window_size": 24,
}

SKABData_config = {
    "data_name": "SKABData",
    "task_flag": "anomaly_detection",
    "data_file_path": "data/anomaly_detection/SKABData",  # Data directory path
    "num_in_features": 8,  # 8 features
    "num_out_features": 1,  # 1 feature
    "beta_range": (0.90, 0.99, 0.001),  # The range of beta values,
    "window_size": 24,
}

model_config = {
    "d_model": 16,
    "num_enc_layers": 1,
    "nhead": 1,
}

loss_functions = {
    "forecasting": nn.MSELoss(),
    "classification": nn.CrossEntropyLoss(),
    "anomaly_detection": nn.MSELoss(),
}

metric_functions = {
    "forecasting": get_forecasting_metrics,
    "classification": get_classification_metrics,
    "anomaly_detection": compute_anomaly_metrics,
}

search_space = {
    "quant_bits": {"low": 4, "high": 8, "step": 2},
    "batch_size": {"low": 16, "high": 256, "step": 16},
    "lr": {"low": 1e-5, "high": 1e-2, "log": True},
    "d_model": {"low": 8, "high": 64, "step": 8},
}
