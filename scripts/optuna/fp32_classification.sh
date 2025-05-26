#!/bin/bash

# data_config
task_flag="classification"
declare -a data_flag_options=("WISDMData") # ("UCIHARData" "WISDMData")

# wandb settings
wandb_project_name="FP32Transformer_optuna_${task_flag}"
wandb_mode="disabled" #  "online" "offline" "disabled"

# experiment settings
num_epochs=1
n_trials=1

for data_flag in "${data_flag_options[@]}"; do
    exp_base_save_dir=$(printf "exp_records_optuna/fp32/")
    set -x  
    python optuna_search.py \
        --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
        --exp_base_save_dir="$exp_base_save_dir" \
        --task_flag="$task_flag" --data_flag="$data_flag" \
        --num_epochs="$num_epochs" --n_trials="$n_trials" 
    set +x  
done