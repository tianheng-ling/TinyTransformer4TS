# [STEP 1] generate the power_simulation.vcd file
if {![file exists "power_simulation/power_simulation.mpf"]} {
    project new "power_simulation" power_simulation
    project addfile "../impl_1/radiant_project_impl_1_vo.vo" verilog
    project addfile "../source/env5se_UserLogicInterface.vhd" vhdl
    project addfile "../source/env5se_top_reconfig_tb.vhd" vhdl
} else {
    project open "power_simulation/power_simulation.mpf"
}

vlog  "+incdir+./impl_1" -work work "../impl_1/radiant_project_impl_1_vo.vo" -suppress 2388
vcom -2008 -work work "../source/env5se_UserLogicInterface.vhd" 
vcom -2008 -work work "../source/env5se_top_reconfig_tb.vhd"

vsim -L work -L pmi_work -L ovi_ice40up  -suppress vsim-7033,vsim-8630,3009,3389 +transport_path_delays +transport_int_delays env5se_top_reconfig_tb

# record the waveforms
add wave /*
vcd file power_simulation.vcd
vcd add i_top/*
run 2989125 ns
exit
