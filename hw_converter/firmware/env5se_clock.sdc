# clock 16 MHz
create_clock -name clk_16m -period 62.5 [get_ports clk_16m]

# SPI 1 MHz
create_clock -name spi_clk -period 1000 [get_ports spi_clk]

