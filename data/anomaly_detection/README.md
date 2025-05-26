# Data for Time-Series Anomaly Detection Problem

## ALFA Dataset
Download the Air Lab Failure and Anomaly Dataset ([ALFA dataset](https://github.com/castacks/alfa-dataset)) and extract it into the data directory ``ALFAData`` under ``data/classification``. 

The ALFA dataset contains multi-modal flight data collected from autonomous UAV flights for the purpose of failure and anomaly detection. The dataset includes 47 fully autonomous flight sequences, each associated with one of eight distinct types of injected faults. These faults occur at known timestamps during the flights, and their ground-truth labels are provided. Each flight sequence is available in .csv, .mat, and .bag formats, and the signals include various telemetry measurements recorded at high frequency. The ALFA dataset has been used in real-time anomaly detection research for UAVs and is suitable for on-device inference and lightweight deep learning model evaluation.

## SKAB Human Activity Dataset

Download the SKAB v0.9 Dataset ([SKAB dataset](https://www.cis.fordham.edu/wisdm/dataset.php)) and place it into the data directory ``SKAB`` under ``data/classification``. The dataset is provided as 35 individual .csv files, each representing a single experiment.

The SKAB dataset contains multivariate time-series data collected from sensors installed on a physical testbed under industrial-like conditions. Each dataset represents one complete run with a single injected anomaly, allowing clear analysis of anomaly patterns. The sensor signals vary across files and are pre-aligned for time-series modeling.
