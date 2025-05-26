LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
entity skeleton is
    generic (
        X_COUNT : integer := ${x_count};
        DATA_WIDTH : integer := ${data_width};
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width}
    );
    port (
        clock               : in std_logic;
        reset               : in std_logic; 
        busy                : out std_logic; 
        wake_up             : out std_logic;        
        rd                  : in std_logic;
        wr                  : in std_logic;   
        data_in             : in std_logic_vector(7 downto 0);
        address_in          : in std_logic_vector(15 downto 0);
        data_out            : out std_logic_vector(7 downto 0);
        led_ctrl            : out std_logic_vector(3 DOWNTO 0)
    );
end skeleton;
architecture rtl of skeleton is
    signal network_enable :  std_logic;
    signal c_config_en :  std_logic;
    signal done :  std_logic;
    signal x_config_en :  std_logic;
    signal x_config_data :  std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_config_addr :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal network_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal input_buffer_we : std_logic:='0';
    signal x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_address : std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    signal y_address : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    type t_x_array is array (0 to X_COUNT) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_ram : t_x_array;
    attribute ram_style : string;
    attribute ram_style of x_ram : signal is "auto";
begin
    led_ctrl(2) <= network_enable;
    led_ctrl(3) <= done;
    inst_${network_name}: entity ${work_library_name}.${network_name}(rtl)
    port map (
        enable => network_enable,
        clock  => clock,
        x_address => x_address,
        y_address => y_address,
        x  => x,
        y  => network_out_data,
        done   => done
    );
    busy <= not done;
    wake_up <= done;
    process (clock, rd, wr, reset)
        variable int_addr : integer range 0 to 20000;
    begin
            if rising_edge(clock) then
                if reset = '1' then
                    network_enable <= '0';
                    led_ctrl(1) <='0';
                else
                    if wr = '1' or rd = '1' then
                        int_addr := to_integer(unsigned(address_in));
                        if wr = '1' then
                            if int_addr<X_COUNT then
                                x_config_data <= data_in(DATA_WIDTH-1 downto 0);
                                x_config_addr <= address_in(x_config_addr'length-1 downto 0);
                                input_buffer_we <= '1';
                                led_ctrl(1) <= '1';
                            elsif int_addr=1000 then
                                network_enable <= data_in(0);
                            end if;
                        elsif rd = '1' then
                            if int_addr<2000 then
                                y_address<= std_logic_vector(to_unsigned(int_addr, y_address'length));
                                data_out <= std_logic_vector(resize(signed(network_out_data), data_out'length));
                            elsif int_addr=2000 then
                                data_out(7 downto 0) <= x"14";
                            else
                                data_out(7 downto 0) <= address_in(7 downto 0);
                            end if;
                        end if;
                    else
                        input_buffer_we <= '0';
                    end if;
            end if;
        end if;
    end process;
    process(clock, input_buffer_we)
        variable var_data_to_write:std_logic_vector(DATA_WIDTH - 1 downto 0);
        variable var_addr : integer range 0 to X_COUNT;
    begin
        if rising_edge(clock) then
            if input_buffer_we='1' then
                var_data_to_write := x_config_data;
                var_addr := to_integer(unsigned(x_config_addr));
            end if;
            x_ram(var_addr) <= var_data_to_write;
        end if;
    end process;
    process(clock)
        variable var_data_to_write:std_logic_vector(DATA_WIDTH - 1 downto 0);
        variable var_addr : integer range 0 to X_COUNT;
    begin
        if rising_edge(clock) then
            x <= x_ram(to_integer(unsigned(x_address)));
        end if;
    end process;
end rtl;