# TinyTransformer4TS
![Time-Series](https://img.shields.io/badge/Time--Series-Forecasting%2C%20Classification%2C%20Anomaly%20Detection-orange)
![FPGA](https://img.shields.io/badge/FPGA-Optimized-blue) ![Quantization](https://img.shields.io/badge/Quantized-Transformer-green)

### Overview

**TinyTransformer4TS** is a unified, end-to-end framework for deploying **integer-only quantized Transformer models** on **low-power embedded FPGAs**. It supports **three core time-series tasks** (one-step ahead forecasting, classification, and threshold-based anomaly detection) and automates the entire pipeline from training to RTL synthesis.

This repository provides:
- 🔧 Full support for model **training and quantization-aware training (QAT)**
- 🧱 Automated **RTL code generation and FPGA synthesis** for deployment
- ⚙️ Integrated **hardware-aware hyperparameter optimization** using Optuna

> ⚠️ This repository works in tandem with our [ElasticAI.Creator](https://github.com/es-ude/elastic-ai.creator/tree/add-linear-quantization) library, which provides the core VHDL templates and quantization modules for hardware generation. Please make sure to install it as part of the setup process.

---

### Corresponding Paper

**Automating Versatile Time-Series Analysis with Tiny Transformers on Embedded FPGAs**  
📌 Accepted at **IEEE Computer Society Annual Symposium on VLSI (ISVLSI) 2025**  
📄 [Preprint (arXiv)](https://arxiv.org/abs/2505.17662) — _Camera-ready version coming soon_

If you use this repository, please consider citing our paper:
```bibtex
@misc{ling2025automating,
  title     = {Automating Versatile Time-Series Analysis with Tiny Transformers on Embedded FPGAs},
  author    = {Tianheng Ling and Chao Qian and Lukas Johannes Haßler and Gregor Schiele},
  year      = {2025},
  eprint    = {2505.17662},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2505.17662}
}
```
---
### Supported FPGA Platforms
This framework targets ultra-small, low-power FPGAs, making it ideal for on-device edge AI deployment:

| FPGA Platform	Model  | Used in Papers | Frequency | Resource Budges               |
| -------------------- | -------------- | --------- | ----------------------------- |
| AMD Spartan-7 Series | XC7S15         | 100 MHz   | 8,000 LUTs, 10 BRAMs, 20 DSPs |
| Lattice iCE40 Series | UP5K           | 16 MHz    | 5,280 LUTs, 30 EBRs, 8 DSPs   |

Deployment scripts and bitstreams for both platforms are included. 
> **Radiant Path Configuration:** If you plan to deploy models to the Lattice iCE40 platform, please **make sure to update the toolchain paths** used in the automation scripts: In `optuna_utils/radiant_runner.py` and `scripts/hw_analysis_pipeline.sh`, replace the hardcoded path `/home/tianhengling/lscc/radiant/2023.2/bin/lin64` with the path to your own Radiant installation directory. You can typically find your Radiant install path by checking the environment variable `RADIANT_PATH` or by locating the executable manually.

---

## Getting Started
#### 1. Clone and Set Up
```
git clone https://github.com/tianheng-ling/TinyTransformer4TS
cd TinyTransformer4TS
```
#### 2. Create and Activate Virtual Environment
```
python -m venv venv --python=python3.11
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

#### 3. Install Dependencies
```
pip install -r requirements.txt
```

---

## Usage
All runnable scripts are organized in the **`scripts/`** folder for convenience: You can **run scripts directly** from their folders.  
For example:
```bash
# Train FP32 model
bash scripts/normal/fp32/forecasting/train.sh

# Model QAT
bash scripts/normal/quant/forecasting/train.sh

# Run full FPGA synthesis and analysis after model QAT
bash scripts/hardware/amd/hw_analysis_pipeline.sh

# Train FP32 model with Optuna search
bash scripts/optuna/fp32_forecasting.sh

# Model QAT with bi-objective Optuna search
bash scripts/optuna/quant_forecasting.sh
```

---

### Related Repositories
This project is part of a broader family of FPGA-optimized time-series models. You may also be interested in:

- **OnDeviceSoftSensorMLPs** → [GitHub Repository](https://github.com/tianheng-ling/OnDeviceSoftSensorMLP)  
- **OnDeviceLSTM** → [GitHub Repository](https://github.com/tianheng-ling/EdgeOverflowForecast)
- **OnDevice1D-(Sep)CNN** → comming soon

---

###  Acknowledgement
This work is supported by the German Federal Ministry for Economic Affairs and Climate Action under the RIWWER project (01MD22007C). 

---

###  Contact
For questions or feedback, please feel free to open an issue or contact us at tianheng.ling@uni-due.de.