
export QT_DEBUG_PLUGINS=1
export QT_QPA_PLATFORM=offscreen
# to avtivate lattice radiant env (if change server, need to change the path)
export bindir=/home/tianhengling/lscc/radiant/2023.2/bin/lin64
source $bindir/radiant_env

txt_file_paths="results_analysis/selected_records/merged_records.txt"
resource_estimation_tcl_path="scripts/hardware/lattice/resource_estimation.tcl"
power_simulation_tcl_path="scripts/hardware/lattice/power_simulation.tcl"
power_estimation_tcl_path="scripts/hardware/lattice/power_estimation.tcl"
while IFS= read -r line
do  
    (
    # set directories
    base_dir="$(pwd)/$line/hw/lattice" 
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

    # [STEP 2] run Radiant synthesis to get resource and power estimation
    tmp_radiant_dir="$base_dir/tmp_radiant_proj"
    rm -rf $tmp_radiant_dir
    mkdir $tmp_radiant_dir
    power_dir="$tmp_radiant_dir/power_simulation"
    rm -rf $power_dir
    mkdir -p $power_dir
    radiant_report_dir="$base_dir/radiant_report"
    rm -rf $radiant_report_dir
    mkdir $radiant_report_dir
    cp -r $vhd_source_dir "$tmp_radiant_dir"
    cp -r $constrains_dir "$tmp_radiant_dir"
    cp -r $simulate_data_dir "$tmp_radiant_dir"
    cp -r "$firmware_dir"/* "$tmp_radiant_dir/source"
    cp -r $resource_estimation_tcl_path "$tmp_radiant_dir"
    cp -r $power_simulation_tcl_path "$tmp_radiant_dir"
    cp -r $power_estimation_tcl_path "$tmp_radiant_dir"

    cd $tmp_radiant_dir
    pnmainc resource_estimation.tcl $tmp_radiant_dir "env5se_top_reconfig"
    if [ -f "$tmp_radiant_dir/impl_1/radiant_project_impl_1.par" ]; then
        cp -r "$tmp_radiant_dir/impl_1/radiant_project_impl_1.par" "$radiant_report_dir/resource_report.par"
    else
        echo "radiant_project_impl_1.par does not exist, please check the synthesis process"
        exit 1
    fi

    # run resouce estimation
    if [ -f "impl_1/radiant_project_impl_1_vo.vo" ]; then
        # run power simulation
        sed -i "s/run 2989125 ns/run $simu_time fs /g" power_simulation.tcl
        RADIANT="/home/tianhengling/lscc/radiant/2023.2" 
        LM_LICENSE_FILE="$RADIANT/license/license.dat" "$RADIANT/modeltech/linuxloem/vsim"  -c -do power_simulation.tcl
        # run power estimation
        pnmainc power_estimation.tcl  
        cp -r "power_estimation_report.txt" $radiant_report_dir
    else
        echo "radiant_project_impl_1_vo.vo does not exist, please check the synthesis process"
        exit 1
    fi
    cd -
    # [Optional] remove tmp_radiant_dir
    # remove tmp_radiant_dir
    # rm -rf $tmp_radiant_dir
    )
done < "$txt_file_paths"
