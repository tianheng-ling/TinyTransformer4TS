
# parsing input parameters
set saif_file_name [lindex $argv 0]
set report_dir [lindex $argv 1]
set simu_time [lindex $argv 2]
set top_module_name [lindex $argv 3] 
puts "SAIF File: $saif_file_name"
puts "Report Directory: $report_dir"
puts "Simulation Time: $simu_time"
puts "Top Module: $top_module_name"

# create project
set proj_name "proj_power"
create_project -force $proj_name ./$proj_name

# set the target device
set_property part xc7s15ftgb196-2 [current_project] 

# adding VHDL source files
add_files -scan_for_includes {./source}
update_compile_order -fileset sources_1
set_property default_lib work [current_project]
set_property library work [get_files -of [get_filesets {sources_1}]]
update_compile_order -fileset sources_1

# wait 5 s to load the design
after 5000

# set the top module
set_property top $top_module_name [current_fileset]
update_compile_order -fileset sources_1
set_property top "${top_module_name}_tb" [get_filesets sim_1]
update_compile_order -fileset sim_1
# add power constraints
add_files -fileset constrs_1 -norecurse {./constraints/env5_power.xdc}

# configure SAIF power analysis
set_property -name {xsim.simulate.runtime} -value {$simu_time} -objects [get_filesets sim_1] 
set_property -name {xsim.simulate.log_all_signals} -value {true} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif_all_signals} -value {true} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif_scope} -value {uut} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif} -value "$saif_file_name" -objects [get_filesets sim_1]

# run FPGA synthesis
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
open_run synth_1 -name synth_1

# run simulation and read SAIF
launch_simulation -mode post-synthesis -type functional
close_sim
read_saif "$saif_file_name"

# generate power consumption report
report_power -file "$report_dir/power_report.txt"

# exit