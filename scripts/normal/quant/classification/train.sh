#!/bin/bash

# data_config
task_flag="classification"
declare -a data_flag_options=("UCIHARData") # ("UCIHARData" "WISDMData")

# quantization settings
declare -a quant_bit_options=(8) # 6 4)

# wandb settings
wandb_project_name="QuantTransformer_${task_flag}"
wandb_mode="online" #  "online" "offline" "disabled"

# experiment settings
batch_size=32
num_epochs=100
lr=0.001
exp_mode="train"
given_timestamp=""
num_exps=1

# HW simulation settings
subset_size=1

for ((i=1; i<=num_exps; i++)); do
    for data_flag in "${data_flag_options[@]}"; do
            for quant_bit in "${quant_bit_options[@]}"; do
                exp_base_save_dir=$(printf "exp_records/quant/%s/%s/%d-bit" "$task_flag" "$data_flag" "$quant_bit")
                echo "Running Experiment: Task=$task_flag, Data=$data_flag, Quant Bit=$quant_bit"
                set -x  
                python main.py \
                    --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
                    --task_flag="$task_flag" --data_flag="$data_flag" \
                    --exp_mode="$exp_mode" --exp_base_save_dir="$exp_base_save_dir" --lr="$lr" \
                    --given_timestamp="$given_timestamp" --batch_size="$batch_size" --num_epochs="$num_epochs" \
                    --quant_bit="$quant_bit" --enable_qat --subset_size="$subset_size" 
                set +x  
        done
    done
done