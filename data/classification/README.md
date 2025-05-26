# Data for Time-Series Classification Task

## UCIHARData Human Activity Dataset
Download the UCIHAR Data Set ([UCIHAR dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip)) and extract it into the data directory ``UCIHARData`` under ``data/classification``. 

The UCIHAR dataset contains recordings of 30 subjects performing 6 different human activities (walking, walking upstairs, walking downstairs, sitting, standing, and lying) while wearing a waist-mounted smartphone with embedded accelerometer and gyroscope.

The signals were recorded at a sampling frequency of **50 Hz**, and the raw data is segmented into **fixed-length windows of 2.56 seconds (128 time steps)** with **50% overlap**.

We use the pre-segmented inertial signal data from 9 sensor channels (body and total acceleration, and gyroscope in x, y, z axes) for time-series classification.

## WISDMData Human Activity Dataset

Download the WISDM v1.1 Dataset ([WISDM dataset](https://www.cis.fordham.edu/wisdm/dataset.php)) and place it under ``data/WISDMData``. The original file is named ``WISDM_ar_v1.1_raw.txt``.

The WISDM dataset contains accelerometer readings collected from smartphones placed in the user’s pocket while performing 6 daily activities: walking, jogging, sitting, standing, climbing upstairs, and downstairs. The data was collected at a sampling rate of **20 Hz**.

⚠️ **Note**: The original `WISDM_ar_v1.1_raw.txt` contains formatting issues such as two records on the same line and inconsistent delimiters like `,;`, causing parsing errors.  
In this work, we use a cleaned version of the file named **`WISDM_ar_v1.1_raw_cleaned.txt`**, where all such formatting issues have been resolved.  

We segment the continuous data stream into overlapping fixed-length windows of **9 seconds (180 time steps)** with a **step size of 100**, and use the 3-axis accelerometer values (`x`, `y`, `z`) as input features for classification.