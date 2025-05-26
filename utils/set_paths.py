import os
from pathlib import Path
from datetime import datetime


def set_base_paths(exp_mode: str, exp_base_save_dir: str, given_timestamp: str = None):

    if exp_mode not in {"train", "test"}:
        raise ValueError(f"Unsupported exp_mode: {exp_mode}")

    timestamp = (
        given_timestamp
        if exp_mode == "test"
        else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    exp_save_dir = Path(exp_base_save_dir) / timestamp

    fig_save_dir, log_save_dir = exp_save_dir / "figs", exp_save_dir / "logs"

    for path in [exp_save_dir, fig_save_dir, log_save_dir]:
        os.makedirs(path, exist_ok=True)

    return exp_save_dir, fig_save_dir, log_save_dir


def get_paths_list(source_path: str):
    with open(source_path, "r") as f:
        paths_list = [line.strip() for line in f.read().splitlines()]
    return paths_list


def parse_path(path):
    """
    example: exp_records/quant/forecasting/AirUData/2-ws/4-bit/2025-03-02_15-51-14
    """
    path_obj = Path(path)
    parts = path_obj.parts

    expected_labels = [
        "task_flag",
        "data_flag",
        "window_size",
        "quant_bits",
        "timestamp",
    ]
    parsed_info = {label: parts[i + 2] for i, label in enumerate(expected_labels)}
    task_flag = str(parsed_info["task_flag"])
    data_flag = str(parsed_info["data_flag"])
    window_size = int(str(parsed_info["window_size"]).split("-ws")[0])
    quant_bits = int(str(parsed_info["quant_bits"]).split("-bit")[0])
    timestamp = str(parsed_info["timestamp"])

    return {
        "timestamp": timestamp,
        "task_flag": task_flag,
        "data_flag": data_flag,
        "window_size": window_size,
        "quant_bits": quant_bits,
    }
