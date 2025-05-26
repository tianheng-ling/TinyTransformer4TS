#!/bin/bash
txt_file_paths="results_analysis/selected_records/merged_records.txt"
resource_estimation_tcl_path="scripts/hardware/amd/resource_estimation.tcl"
power_estimation_tcl_path="scripts/hardware/amd/power_estimation.tcl"
top_module_name="network"

while IFS= read -r line
do  
    (
    # set directories
    base_dir="$(pwd)/$line/hw/amd" 
    simulate_data_dir="$base_dir/data"
    vhd_source_dir="$base_dir/source"
    constrains_dir="$base_dir/constraints"
    firmware_dir="$base_dir/firmware"
    makefile_path="$base_dir/makefile"

    # [STEP 1] run GHDL simulation
    tmp_ghdl_dir="$base_dir/tmp_ghdl_proj"
    rm -rf $tmp_ghdl_dir
    mkdir $tmp_ghdl_dir
    ghdl_report_dir="$base_dir/ghdl_report"
    rm -rf $ghdl_report_dir
    mkdir $ghdl_report_dir
    cp -r $vhd_source_dir "$tmp_ghdl_dir"
    cp -r $simulate_data_dir "$tmp_ghdl_dir"
    cp $makefile_path $tmp_ghdl_dir
    tmp_ghdl_source_dir="$tmp_ghdl_dir/source"
    tmp_ghdl_data_dir="$tmp_ghdl_dir/data"
    cd $tmp_ghdl_dir
    for dir in $(find "$tmp_ghdl_source_dir" -mindepth 1 -type d)
    do  
        module=$(basename $dir)
        make TESTBENCH=$module
        cp $tmp_ghdl_dir/.simulation/make_output.txt $ghdl_report_dir/"ghdl_${module}_output.txt"
    done
    simu_time=$(grep "Time taken for processing" $ghdl_report_dir/"ghdl_network_output.txt" | awk -F'= ' '{print $2}' | awk '{print $1}')
    cd -
    
    # [STEP 2] run Vivado synthesis to get resource and power estimation
    tmp_vivado_dir="$base_dir/tmp_vivado_proj"
    rm -rf $tmp_vivado_dir
    mkdir $tmp_vivado_dir
    vivado_report_dir="$base_dir/vivado_report" 
    rm -rf $vivado_report_dir
    mkdir $vivado_report_dir
    cp -r $resource_estimation_tcl_path "$tmp_vivado_dir"
    cp -r $power_estimation_tcl_path "$tmp_vivado_dir"
    cp -r $vhd_source_dir "$tmp_vivado_dir" 
    cp -r $simulate_data_dir "$tmp_vivado_dir"
    cp -r $constrains_dir "$tmp_vivado_dir"
    cp -r "$firmware_dir"/* "$tmp_vivado_dir/source"
    absolute_data_dir="${tmp_vivado_dir}/data"
    sed -i "s|./data|$absolute_data_dir|g" $tmp_vivado_dir/source/network/network_tb.vhd
    saif_file_name="${tmp_vivado_dir}/sim_wave.saif"

    cd $tmp_vivado_dir
    source /tools/Xilinx/Vivado/2019.2/settings64.sh
    vivado -mode tcl -source resource_estimation.tcl -tclargs $vivado_report_dir  -nojournal -nolog < /dev/null
    vivado -mode tcl -source power_estimation.tcl -tclargs $saif_file_name $vivado_report_dir "${simu_time}fs" $top_module_name -nojournal -nolog < /dev/null
    cd -

    # [Optional] remove tmp_vivado_dir
    # remove tmp_vivado_dir
    # rm -rf $tmp_vivado_dir
    )
done < "$txt_file_paths"