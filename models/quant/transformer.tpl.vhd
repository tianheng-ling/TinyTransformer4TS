library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width}
    ) ;
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done : out std_logic
    ) ;
end ${name};
architecture rtl of ${name} is
    function log2(val : INTEGER) return natural is
        variable result : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                result := i;
                exit;
            end if;
        end loop;
        return result;
    end function log2;
    signal input_linear_enable : std_logic;
    signal input_linear_clock : std_logic;
    signal input_linear_x_address : std_logic_vector(${input_linear_x_addr_width} - 1 downto 0);
    signal input_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal input_linear_y_address : std_logic_vector(${input_linear_y_addr_width} - 1 downto 0);
    signal input_linear_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal input_linear_done : std_logic;
    signal add_pos_info_enable : std_logic;
    signal add_pos_info_clock : std_logic;
    signal pos_info_address : std_logic_vector(${input_linear_y_addr_width} - 1 downto 0);
    signal pos_info : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_pos_info_x_1_address : std_logic_vector(${add_pos_info_x_addr_width} - 1 downto 0);
    signal add_pos_info_x_1: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_pos_info_x_2_address : std_logic_vector(${add_pos_info_x_addr_width} - 1 downto 0);
    signal add_pos_info_x_2: std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_pos_info_y_address : std_logic_vector(${add_pos_info_y_addr_width} - 1 downto 0);
    signal add_pos_info_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_pos_info_done : std_logic;
    signal encoder_enable : std_logic;
    signal encoder_clock : std_logic;
    signal encoder_x_address : std_logic_vector(${add_pos_info_y_addr_width} - 1 downto 0);
    signal encoder_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal encoder_y_address : std_logic_vector(${add_pos_info_y_addr_width} - 1 downto 0);
    signal encoder_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal encoder_done : std_logic;
    signal gap_enable : std_logic;
    signal gap_clock : std_logic;
    signal gap_x_address : std_logic_vector(${gap_x_addr_width} - 1 downto 0);
    signal gap_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal gap_y_address : std_logic_vector(${gap_y_addr_width} - 1 downto 0);
    signal gap_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal gap_done : std_logic;
    signal output_linear_enable : std_logic;
    signal output_linear_clock : std_logic;
    signal output_linear_x_address : std_logic_vector(${output_linear_x_addr_width} - 1 downto 0);
    signal output_linear_x : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal output_linear_y_address : std_logic_vector(${output_linear_y_addr_width} - 1 downto 0);
    signal output_linear_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal output_linear_done : std_logic;
    begin
        input_linear_enable <= enable;
        add_pos_info_enable <= input_linear_done;
        encoder_enable <= add_pos_info_done;
        gap_enable <= encoder_done;
        output_linear_enable <= gap_done;
        done <= output_linear_done;

        input_linear_clock <= clock;
        add_pos_info_clock <= clock;
        encoder_clock <= clock;
        gap_clock <= clock;
        output_linear_clock <= clock;

        x_address <= input_linear_x_address;
        input_linear_y_address <= add_pos_info_x_1_address;
        pos_info_address <= add_pos_info_x_2_address;
        add_pos_info_y_address <= encoder_x_address;
        encoder_y_address <= gap_x_address;
        gap_y_address <= output_linear_x_address;
        output_linear_y_address <= y_address;

        input_linear_x <= x;
        add_pos_info_x_1 <= input_linear_y;
        add_pos_info_x_2 <= pos_info;
        encoder_x <= add_pos_info_y;
        gap_x <= encoder_y;
        output_linear_x <= gap_y;
        y <= output_linear_y;
        
        inst_${input_linear_name}: entity ${work_library_name}.${input_linear_name}(rtl)
        port map (
            enable => input_linear_enable,
            clock => input_linear_clock,
            x_address => input_linear_x_address,
            y_address => input_linear_y_address,
            x => input_linear_x,
            y => input_linear_y,
            done => input_linear_done
        );
        inst_${add_pos_info_name}: entity ${work_library_name}.${add_pos_info_name}(rtl)
        port map (
            enable => add_pos_info_enable,
            clock => add_pos_info_clock,
            x_1_address => add_pos_info_x_1_address,
            x_2_address => add_pos_info_x_2_address,
            y_address => add_pos_info_y_address,
            x_1=> add_pos_info_x_1,
            x_2=> add_pos_info_x_2,
            y => add_pos_info_y,
            done => add_pos_info_done
        );
        inst_${encoder_name}: entity ${work_library_name}.${encoder_name}(rtl)
        port map (
            enable => encoder_enable,
            clock => encoder_clock,
            x_address => encoder_x_address,
            y_address => encoder_y_address,
            x => encoder_x,
            y => encoder_y,
            done => encoder_done
        );
        inst_${avgpooling_name}: entity ${work_library_name}.${avgpooling_name}(rtl)
        port map (
            enable => gap_enable,
            clock => gap_clock,
            x_address => gap_x_address,
            y_address => gap_y_address,
            x => gap_x,
            y => gap_y,
            done => gap_done
        );
        inst_${output_linear_name}: entity ${work_library_name}.${output_linear_name}(rtl)
        port map (
            enable => output_linear_enable,
            clock => output_linear_clock,
            x_address => output_linear_x_address,
            y_address => output_linear_y_address,
            x => output_linear_x,
            y => output_linear_y,
            done => output_linear_done
        );
        inst_${pos_info_rom_name}: entity ${work_library_name}.${pos_info_rom_name}(rtl)
        port map  (
            clk  => clock,
            en   => '1',
            addr => pos_info_address,
            data => pos_info
        );
end architecture; 

