library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;             
library ice40up;
use ice40up.components.all;
library work;
use work.userlogicinterface.all;
entity env5se_top_reconfig is
port (
clk_16m      : in std_logic;
spi_clk      : in std_logic;
spi_ss_n     : in std_logic;
spi_mosi     : in std_logic;
spi_miso     : out std_logic;
fpga_busy    : out std_logic;
rgb0, rgb1, rgb2 : out std_logic
);
end env5se_top_reconfig;
architecture rtl of env5se_top_reconfig is
    signal reset : std_logic := '1';
    signal spi_reset_n : std_logic := '0';
    signal sram_address : std_logic_vector(15 downto 0);
    signal sram_data_in, sram_data_out : std_logic_vector(7 downto 0);
    signal sram_wr, sram_rd : std_logic;
    signal mw_leds,ul_leds : std_logic_vector(3 downto 0);
    -- signals for user logic
    signal userlogic_reset, userlogic_rd, userlogic_wr : std_logic;
    signal userlogic_data_in : std_logic_vector(7 downto 0);
    signal userlogic_data_out : std_logic_vector(7 downto 0);
    signal userlogic_address : std_logic_vector(15 downto 0);
    signal userlogic_clock, userlogic_busy: std_logic;
    signal sys_clk : std_logic;
    signal leds : std_logic_vector(2 downto 0);
begin
    sys_clk <= clk_16m;
    leds <= ul_leds(2 downto 0) or mw_leds(2 downto 0);
    fpga_busy <= userlogic_busy;
    i_rgb : RGB
        generic map (
            CURRENT_MODE => "1",
            RGB0_CURRENT => "0b000011",
            RGB1_CURRENT => "0b000011",
            RGB2_CURRENT => "0b000011"
        )
        port map (
            CURREN => '1',
            RGBLEDEN => '1',
            RGB0PWM => leds(0),
            RGB1PWM => leds(1),
            RGB2PWM => leds(2),
            RGB0 => rgb0,
            RGB1 => rgb1,
            RGB2 => rgb2
        );
    i_spi_slaver: entity work.spi_slave(rtl)
    port map(
        reset_n => spi_reset_n,
        sclk => spi_clk,
        ss_n => spi_ss_n,
        mosi => spi_mosi,
        miso => spi_miso,
        clk => sys_clk,
        addr => sram_address,
        data_wr => sram_data_in,  -- tx_data,
        data_rd => sram_data_out, -- rx_data,
        we => sram_wr,            -- tx_en,
        re => sram_rd             -- rx_en
    );
    spi_reset_n <= not reset;
    -- Adapting the removed middleware 
    userlogic_data_in <= sram_data_in;
    sram_data_out     <= userlogic_data_out;
    process (sys_clk)
    begin
        if rising_edge(sys_clk) then
            if sram_wr='1' and (unsigned(sram_address)=3) then
                mw_leds <= sram_data_in(3 downto 0);
            elsif  sram_wr='1' and (unsigned(sram_address)=4) then
                userlogic_reset <= sram_data_in(0);
            end if;
            if sram_rd='1' or sram_wr='1' then
                userlogic_address <= std_logic_vector(unsigned(sram_address)-to_unsigned(256,sram_address'length));
            end if;
            userlogic_rd <= sram_rd;
            userlogic_wr <= sram_wr;
        end if;
    end process;
    process (sys_clk)
        constant reset_count : integer := 30000; -- 1ms @ 100MHz
        variable count : integer range 0 to reset_count := 0;
    begin
        if rising_edge(sys_clk) then
            if count < reset_count then
                count := count + 1;
                reset <= '1';
            else
                reset <= '0';
            end if;
        end if;
    end process;
    userlogic_clock <= sys_clk and (not userlogic_reset);
    ul : entity work.skeleton PORT MAP
        (
        clock => userlogic_clock,
        reset => userlogic_reset, -- H -> reset
        busy => userlogic_busy,
        rd => userlogic_rd,
        wr => userlogic_wr,
        data_in => userlogic_data_in,
        address_in => userlogic_address,
        data_out => userlogic_data_out,
        led_ctrl => ul_leds
        );
end rtl;
