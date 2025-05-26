library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
library work;
use work.all;
entity env5se_top_reconfig_tb is
    generic (
        X_COUNT : integer := ${x_count}
    );
end entity;
architecture behavior of env5se_top_reconfig_tb is
    constant SYS_CLK_PERIOD : time := 62.5 ns;
    constant SPI_CLK_PERIOD : time := 1000 ns;
    signal sys_clk : std_logic := '0';
    signal spi_clk_base : std_logic := '0';
    signal spi_clk_enable : std_logic := '0';
    signal spi_clk : std_logic := '0';
    signal spi_ss_n : std_logic := '1';
    signal spi_mosi : std_logic := '0';
    signal spi_miso : std_logic := '0';
    signal fpga_busy : std_logic := '0';
    signal rgb0 : std_logic := '0';
    signal rgb1 : std_logic := '0';
    signal rgb2 : std_logic := '0';
begin
    sys_clk_gen : process
    begin
        sys_clk <= '1';
        wait for SYS_CLK_PERIOD / 2;
        sys_clk <= '0';
        wait for SYS_CLK_PERIOD / 2;
    end process;

    spi_clk_gen : process
    begin
        spi_clk_base <= '1';
        wait for SPI_CLK_PERIOD / 2;
        spi_clk_base <= '0';
        wait for SPI_CLK_PERIOD / 2;
    end process;

    spi_clk <= spi_clk_base and spi_clk_enable;

    test_main : process
        procedure spi_send_byte(
            data_send : in std_logic_vector(7 downto 0);
            data_recv : out std_logic_vector(7 downto 0)
        ) is
            variable bit_count : integer := 8;
        begin
            wait until falling_edge(spi_clk_base);
            spi_clk_enable <= '1';
            while bit_count /= 0 loop
                -- shift out data
                wait until rising_edge(spi_clk);
                spi_mosi <= data_send(bit_count - 1);
                -- sample in data
                wait until falling_edge(spi_clk);
                data_recv(bit_count - 1) := spi_miso;
                bit_count := bit_count - 1;
            end loop;
            spi_clk_enable <= '0';
            spi_mosi <= '0';
        end procedure;
        type data_array is array(natural range <>) of std_logic_vector(7 downto 0);
        procedure spi_write(
            data : in data_array
        ) is
            variable dummy : std_logic_vector(7 downto 0);
        begin
            for i in data'range loop
                spi_send_byte(data(i), dummy);
            end loop;
        end procedure;
        procedure spi_read(
            data : out data_array
        ) is
            variable byte_buf : std_logic_vector(7 downto 0) := (others => '0');
        begin
            for i in data'range loop
                spi_send_byte("00000000", byte_buf);
                data(i) := byte_buf;
            end loop;
        end procedure;
        constant QXI_WR : std_logic_vector(7 downto 0) := x"80";
        constant QXI_RD : std_logic_vector(7 downto 0) := x"40";
        procedure qxi_write(
            addr : in integer range 0 to 2**16-1;
            data : in data_array
        ) is
            variable addr_bits : std_logic_vector(15 downto 0) := std_logic_vector(to_unsigned(addr, 16));
        begin
            spi_ss_n <= '0';
            wait for SYS_CLK_PERIOD * 10;
            spi_write((QXI_WR, addr_bits(15 downto 8), addr_bits(7 downto 0)));
            spi_write(data);
            wait for SYS_CLK_PERIOD * 10;
            spi_ss_n <= '1';
        end procedure;
        procedure qxi_read(
            addr : in integer range 0 to 2**16-1;
            data : out data_array
        ) is
            variable addr_bits : std_logic_vector(15 downto 0) := std_logic_vector(to_unsigned(addr, 16));
        begin
            spi_ss_n <= '0';
            wait for SYS_CLK_PERIOD * 10;
            spi_write((QXI_RD, addr_bits(15 downto 8), addr_bits(7 downto 0)));
            spi_read(data);
            wait for SYS_CLK_PERIOD * 10;
            spi_ss_n <= '1';
        end procedure;
        constant MW_ADDR_LEDS : integer := 16#0003#;
        constant MW_ADDR_USER_LOGIC_RESET : integer := 16#0004#;
        constant MW_ADDR_USER_LOGIC : integer := 16#0100#;
        procedure middleware_set_leds(
            leds : in std_logic_vector(7 downto 0)
        ) is
        begin
            qxi_write(MW_ADDR_LEDS, (0 => leds));
        end procedure;
        procedure middleware_user_logic_reset(
            reset : in boolean
        ) is
            variable reset_bits : std_logic_vector(7 downto 0) := (others => '0');
        begin
            --reset_bits(0) := '1' when reset else '0';
            if reset then
            reset_bits(0) :='1';
            else
                reset_bits(0) :='0';
            end if;
            qxi_write(MW_ADDR_USER_LOGIC_RESET, (0 => reset_bits));
        end procedure;
        procedure middleware_write(
            addr : in integer range 0 to 2**16-1;
            data : in data_array
        ) is
        begin
            qxi_write(addr, data); -- qxi_write(MW_ADDR_USER_LOGIC + addr, data);
        end procedure;
        procedure middleware_read(
            addr : in integer range 0 to 2**16-1;
            data : out data_array
        ) is
        begin
            qxi_read(addr, data); -- qxi_write(MW_ADDR_USER_LOGIC + addr, data);
        end procedure;
        constant UL_ADDR_DATA : integer := 0;
        constant UL_ADDR_ENABLE : integer := 1000;
        procedure model_predict(
            inputs : in data_array;
            outputs : out data_array
        ) is
        begin
            -- enable model
            middleware_user_logic_reset(false);
            wait for 1 us;
            -- send inputs
            middleware_write(UL_ADDR_DATA, inputs);
            wait for 1 us;
            -- run model
            middleware_write(UL_ADDR_ENABLE, (0 => x"01"));
            wait for 1 us;
            wait until fpga_busy = '0';
            middleware_write(UL_ADDR_ENABLE, (0 => x"00"));
            wait for 1 us;
            -- read outputs
            middleware_read(UL_ADDR_DATA, outputs);
            wait for 1 us;
            -- disable model
            middleware_user_logic_reset(true);
            wait for 1 us;
        end procedure;
        constant PATH_INPUTS : string := "../data/network_q_x.txt";
        constant PATH_OUTPUTS : string := "../data/network_q_y.txt";
        file fp_inputs : text;
        file fp_outputs : text;
        variable file_status : file_open_status;
        variable line_num : line;
        variable line_value : integer;
        variable inputs : data_array(0 to X_COUNT-1) := (others => x"00");
        variable outputs : data_array(0 to 0) := (0 => x"00");
        variable output_int : integer := 0;
    begin
        file_open(file_status, fp_inputs, PATH_INPUTS, READ_MODE);
        assert file_status = OPEN_OK
            report "failed to open inputs file" severity failure;

        file_open(file_status, fp_outputs, PATH_OUTPUTS, READ_MODE);
        assert file_status = OPEN_OK
            report "failed to open outputs file" severity failure;
        wait for 1 us;
        -- reset model
        middleware_user_logic_reset(true);
        wait for 1 us;
        while not endfile(fp_inputs) loop
            for i in inputs'range loop
                readline(fp_inputs, line_num);
                read(line_num, line_value);
                inputs(i) := std_logic_vector(to_signed(line_value, 8));
            end loop;
            model_predict(inputs, outputs);
            output_int := to_integer(signed(outputs(0)));
            readline(fp_outputs, line_num);
            read(line_num, line_value);
            if output_int /= line_value then
                report "wrong output " & integer'image(output_int) & ", expected " & integer'image(line_value)
                    severity warning;
            end if;
        end loop;
        wait for 1 us;
        report "finished simulation";
        file_close(fp_inputs);
        file_close(fp_outputs);
        std.env.stop;
    end process;
    i_top : entity work.env5se_top_reconfig(rtl)
    port map (
        clk_16m => sys_clk,
        spi_clk => spi_clk,
        spi_ss_n => spi_ss_n,
        spi_mosi => spi_mosi,
        spi_miso => spi_miso,
        fpga_busy => fpga_busy,
        rgb0 => rgb0,
        rgb1 => rgb1,
        rgb2 => rgb2
    );
end architecture;
