set_property -dict {PACKAGE_PIN G11 IOSTANDARD LVCMOS33} [get_ports clock]
create_clock -period 10.000 -name clock -waveform {0.000 5.000} -add [get_ports clock]
#create_clock -period 5.000 -name clock -waveform {0.000 2.500} -add [get_ports clock]

set_operating_conditions -process maximum
reset_switching_activity -all 
set_operating_conditions -ambient_temp 28.0
set_operating_conditions -board_layers 4to7
set_operating_conditions -board small
set_operating_conditions -design_power_budget 1.0
set_operating_conditions -heatsink none
reset_switching_activity -all 
reset_switching_activity -no_deassert_resets 
reset_switching_activity -no_deassert_resets 
