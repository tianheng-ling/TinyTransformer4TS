#!/bin/bash

# data_config
task_flag="forecasting"
declare -a data_flag_options=("AirUData") # "PeMSData" "AirUData"
declare -a window_size_options=(24)

# wandb settings
wandb_project_name="FP32Transformer_${task_flag}"
wandb_mode="online" #  "online" "offline" "disabled"

# experiment settings
batch_size=208
num_epochs=100
lr=0.006357
exp_mode="train"
given_timestamp=""
num_exps=2

for ((i=1; i<=num_exps; i++)); do
    for data_flag in "${data_flag_options[@]}"; do
        for window_size in "${window_size_options[@]}"; do
            exp_base_save_dir=$(printf "exp_records/fp32/%s/%s/%d-ws" "$task_flag" "$data_flag" "$window_size")
            
            echo "Running Experiment: Task=$task_flag, Data=$data_flag, Window Size=$window_size"
            set -x  
            python main.py \
                --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
                --task_flag="$task_flag" --data_flag="$data_flag" --window_size="$window_size" \
                --exp_mode="$exp_mode" --exp_base_save_dir="$exp_base_save_dir" --lr="$lr" \
                --given_timestamp="$given_timestamp" --batch_size="$batch_size" --num_epochs="$num_epochs"
            set +x  
        done
    done
done