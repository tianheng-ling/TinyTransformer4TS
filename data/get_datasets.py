from config import (
    AirUData_config,
    PeMSData_config,
    WISDMData_config,
    UCIHARData_config,
    ALFAData_config,
    SKABData_config,
)
from data.forecasting.AirUDataset import get_AirUDatasets
from data.forecasting.PeMSDataset import get_PeMSDatasets
from data.classification.UCIHARDataset import get_UCIHARDatasets
from data.classification.WISDMDataset import get_WISDMDatasets
from data.anomaly_detection.ALFADataset import get_ALFADatasets
from data.anomaly_detection.SKABDataset import get_SKABDatasets


def get_data_config(task_flag: str, data_flag: str):
    data_configs = {
        "forecasting": {
            "AirUData": AirUData_config,
            "PeMSData": PeMSData_config,
        },
        "classification": {
            "UCIHARData": UCIHARData_config,
            "WISDMData": WISDMData_config,
        },
        "anomaly_detection": {
            "ALFAData": ALFAData_config,
            "SKABData": SKABData_config,
        },
    }

    if task_flag not in data_configs:
        raise ValueError(f"Task flag '{task_flag}' not supported")

    if data_flag not in data_configs[task_flag]:
        raise ValueError(
            f"Data flag '{data_flag}' not supported for task '{task_flag}'"
        )

    return data_configs[task_flag][data_flag]


def get_datasets(data_config: dict):

    task_flag = data_config["task_flag"]
    data_flag = data_config["data_name"]

    dataset_getters = {
        "forecasting": {
            "AirUData": lambda: get_AirUDatasets(data_config),
            "PeMSData": lambda: get_PeMSDatasets(data_config),
        },
        "classification": {
            "UCIHARData": lambda: get_UCIHARDatasets(data_config),
            "WISDMData": lambda: get_WISDMDatasets(data_config),
        },
        "anomaly_detection": {
            "ALFAData": lambda: get_ALFADatasets(data_config),
            "SKABData": lambda: get_SKABDatasets(data_config),
        },
    }

    return dataset_getters[task_flag][data_flag]()
