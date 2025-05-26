# parse the input parameters
package require fileutil
if { $argc != 2 } {
    puts "Error: The resource_sim.tcl script requires two parameters: proj_dir and top_module_name"
    puts "Usage: pnmainc resource_sim.tcl <proj_dir> <top_module_name>"
    exit 1
} else {
    set proj_dir [lindex $argv 0]
    set top_module_name [lindex $argv 1]
    puts "This is a customized build script for Lattice Radiant"
    puts "Project Directory: $proj_dir"
    puts "Top Module: $top_module_name"
}

# create a new project
prj_create -name "radiant_project" -impl "impl_1" -dev iCE40UP5K-SG48I -performance "High-Performance_1.2V" -synthesis "lse"
prj_set_strategy_value -strategy Strategy1 lse_opt_goal=Timing lse_vhdl2008=True

# add vhdl files
foreach file [fileutil::findByPattern "$proj_dir/source" "*.vhd"] { 
    prj_add_source $file
    }

# add constraints
prj_add_source "$proj_dir/constraints/env5se_clock.sdc"
prj_add_source "$proj_dir/constraints/env5se_pins.pdc"

# set the top module
prj_set_impl_opt -impl "impl_1" "top" "$top_module_name"

# run the synthesis
prj_run Synthesis -impl impl_1 -task SynTrace

# run the export
prj_run Export -impl impl_1 

# save and close the project
prj_save
prj_close