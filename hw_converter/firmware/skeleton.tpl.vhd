library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity skeleton is
    port (
        clock              : in std_logic;
        reset              : in std_logic; 
        busy               : out std_logic; 
        wake_up            : out std_logic;
        rd                 : in std_logic;    
        wr                 : in std_logic;    
        data_in            : in std_logic_vector(7 downto 0);
        address_in         : in std_logic_vector(15 downto 0);
        data_out           : out std_logic_vector(7 downto 0);
        debug              : out std_logic_vector(7 downto 0);
        led_ctrl           : out std_logic_vector(3 DOWNTO 0)
    );
end;
architecture rtl of skeleton is
    constant X_DATA_WIDTH : integer := ${x_data_width};
    constant Y_DATA_WIDTH : integer := ${y_data_width};
    constant X_ADDR_WIDTH : integer := ${x_addr_width};
    constant Y_ADDR_WIDTH : integer:= ${y_addr_width};
    constant X_COUNT : integer := ${x_count};
    constant Y_COUNT : integer := ${y_count};
    signal network_enable :  std_logic;
    signal c_config_en :  std_logic;
    signal done :  std_logic;
    signal x :  std_logic_vector(X_DATA_WIDTH-1 downto 0);
    signal y : std_logic_vector(Y_DATA_WIDTH-1 downto 0);
    signal x_address :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal y_address :  std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    type buf_data_in_t is array (0 to X_COUNT) of std_logic_vector(X_DATA_WIDTH-1 downto 0);
    signal data_buf_in : buf_data_in_t;
    type skeleton_id_data_t is array (0 to 0) of std_logic_vector(7 downto 0);
    signal skeleton_id_str : skeleton_id_data_t := (0 => "01000010");
    function pad_output_to_middleware(network_out : std_logic_vector(Y_DATA_WIDTH-1 downto 0)) return std_logic_vector is
    variable k : std_logic_vector(7 downto 0);
    begin
        if Y_DATA_WIDTH /= 8 then
            k(7 downto Y_DATA_WIDTH) := (others => '0');
        end if;
        k(Y_DATA_WIDTH-1 downto 0) := network_out;
        return k;
    end function;

begin
    i_${network_name}: entity work.${network_name}(rtl)
    port map (
        clock => clock,
        enable => network_enable,
        x => x,
        y => y,
        x_address => x_address,
        y_address => y_address,
        done => done
    );
    busy <= not done;
    wake_up <= done;
    receive_data_from_middleware: process (clock, wr, address_in)
    variable int_addr : integer range 0 to 20000;
    begin
        if rising_edge(clock) then
            if reset = '1' then
                network_enable <= '0';
            else
                int_addr := to_integer(unsigned(address_in));
                if int_addr = 0 then
                    network_enable <= data_in(0);
                elsif int_addr < X_COUNT then
                    data_buf_in(int_addr) <= data_in(X_DATA_WIDTH-1 downto 0);
                end if;
            end if;
        end if;
    end process;
    sendback_data_to_middleware: process  (clock, rd, address_in)
    variable int_addr : integer range 0 to 2000;
    begin
        if rising_edge(clock) then
            int_addr := to_integer(unsigned(address_in));
            if int_addr <= Y_COUNT-1 then
                y_address <= address_in(y_address'length-1 downto 0);
                data_out(7 downto 0) <= pad_output_to_middleware(y);
            elsif int_addr = 2000  then
                data_out(7 downto 0) <= skeleton_id_str(int_addr-2000);
            end if;
        end if;
    end process;
    send_buf_to_network: process (clock, x_address)
    begin
        if rising_edge(clock) then
            x <= data_buf_in(to_integer(unsigned(x_address)));
        end if;
    end process;
end rtl;
