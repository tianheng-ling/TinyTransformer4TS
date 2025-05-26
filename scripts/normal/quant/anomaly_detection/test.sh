#!/bin/bash

# data_config
task_flag="anomaly_detection"
declare -a data_flag_options=("SKABData") 
declare -a window_size_options=(24)

# quantization settings
declare -a quant_bit_options=(8) #(8 6 4)

# wandb settings
wandb_project_name="QuantTransformer_${task_flag}"
wandb_mode="disabled" #  "online" "offline" "disabled"


# experiment settings
batch_size=256
num_epochs=10
lr=0.001
exp_mode="test"
given_timestamp="2025-04-19_17-01-36" #

# HW simulation settings
subset_size=1

for data_flag in "${data_flag_options[@]}"; do
    for window_size in "${window_size_options[@]}"; do
        for quant_bit in "${quant_bit_options[@]}"; do
            
            exp_base_save_dir=$(printf "exp_records/quant/%s/%s/%d-ws/%d-bit" "$task_flag" "$data_flag" "$window_size" "$quant_bit")

            echo "Running Experiment: Task=$task_flag, Data=$data_flag, Window Size=$window_size"
            set -x  
            python main.py \
                --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
                --task_flag="$task_flag" --data_flag="$data_flag" --window_size="$window_size" \
                --exp_mode="$exp_mode" --exp_base_save_dir="$exp_base_save_dir" --lr="$lr" \
                --given_timestamp="$given_timestamp" --batch_size="$batch_size" --num_epochs="$num_epochs" \
                --quant_bit="$quant_bit" --enable_qat --subset_size="$subset_size"
            set +x  
        done
    done
done
