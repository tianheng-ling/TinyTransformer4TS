#!/bin/bash

# data_config
task_flag="classification"
declare -a data_flag_options=("UCIHARData") # "WISDMData" "UCIHARData"

# experiment settings
num_epochs=100

# optuna settings
n_trials=100
optuna_hw_target="energy"

# hw translation settings
subset_size=1
target_hw_options=("amd") # "lattice" "amd"

# wandb settings
wandb_mode="online" #  "online" "offline" "disabled"

for target_hw in "${target_hw_options[@]}"; do
    for data_flag in "${data_flag_options[@]}"; do
        exp_base_save_dir=$(printf "exp_records_optuna_${n_trials}/quant/${target_hw}/${task_flag}/${data_flag}/")
        wandb_project_name="optuna_${task_flag}_${target_hw}_${n_trials}trials"
        set -x  
        python optuna_search.py \
            --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
            --exp_base_save_dir="$exp_base_save_dir" \
            --task_flag="$task_flag" --data_flag="$data_flag" \
            --num_epochs="$num_epochs" --n_trials="$n_trials" \
            --enable_qat --enable_hw_simulation --subset_size="$subset_size" \
            --optuna_hw_target="$optuna_hw_target" --target_hw="$target_hw" 
        set +x  
    done
done