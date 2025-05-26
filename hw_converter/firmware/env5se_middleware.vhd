library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
library work;
use work.UserLogicInterface.all;
entity middleware is
	generic
	(
		control_region		: unsigned(15 downto 0) := x"00ff"
	);
	port (
		clk 				: in std_ulogic;	--! Clock 32 MHz
		reset  				: in std_logic;
		-- userlogic
		userlogic_reset		: out std_logic;
		userlogic_data_in	: out std_logic_vector(7 downto 0);
		userlogic_data_out	: in std_logic_vector(7 downto 0);
		userlogic_address	: out std_logic_vector(15 downto 0);
		userlogic_rd		: out std_logic;
		userlogic_wr		: out std_logic;
		-- debug
		interface_leds	: out std_logic_vector(3 downto 0);
		-- sram
		sram_address 	: in std_logic_vector(15 downto 0);
		sram_data_out	: out std_logic_vector(7 downto 0); -- for reading from ext ram
		sram_data_in 	: in std_logic_vector(7 downto 0); 	-- for writing to ext ram
		sram_rd			: in std_logic;
		sram_wr			: in std_logic
	);
end middleware;
architecture rtl of middleware is
	signal led_signal : std_logic_vector(3 downto 0) := (others => '0');
	signal userlogic_reset_signal : std_logic := '0';
	constant LED : uint16_t := x"0003";
	constant USERLOGIC_CONTROL : uint16_t := x"0004";
begin
	-- assign sram interface to correct ul or mw interface
	interface_leds <= led_signal;
	userlogic_reset <= userlogic_reset_signal;
	userlogic_data_in <= sram_data_in;
	-- main data receiving process
	process (reset, clk, sram_rd, sram_wr) 
		variable data_var : std_logic_vector(7 downto 0);
		variable wr_was_low : boolean := false;
		variable sram_control_region_active : boolean;
		variable middleware_data_out : std_logic_vector(7 downto 0);
	begin
        if rising_edge(clk) then
			if reset = '1' then
                led_signal <= (others => '0');
                userlogic_reset_signal <= '1';
                middleware_data_out := (others => '0');
            else
				sram_control_region_active := (unsigned(sram_address) <= unsigned(control_region));
				if sram_rd = '1' or sram_wr = '1' then -- or wr_was_low then
					-- writing to an address
					-- only respond when sram_wr goes high again
					if sram_wr = '1' then
						-- control region
						if sram_control_region_active then
							case unsigned(sram_address) is
							when LED =>
								data_var := std_logic_vector(sram_data_in);
								led_signal <= data_var(3 downto 0);
							when USERLOGIC_CONTROL =>
								data_var := std_logic_vector(sram_data_in);
								userlogic_reset_signal <= data_var(0);
							when others =>
							end case;
						end if;
					-- otherwise reading
					else
						-- control region
						if sram_control_region_active then
							-- write unaffected as zero
							middleware_data_out := (others => '0');
							-- middleware
							case unsigned(sram_address) is
							when LED =>
								middleware_data_out(3 downto 0) := led_signal;
							when USERLOGIC_CONTROL =>
								middleware_data_out(0) := userlogic_reset_signal;
							when others =>
								middleware_data_out(7 downto 0) := sram_address(7 downto 0);
							end case;
							-- userlogic
							if sram_control_region_active then
								sram_data_out <= middleware_data_out;
							else
								sram_data_out <= userlogic_data_out;
							end if;
						end if;
					end if;
				end if;
				if sram_control_region_active then
					userlogic_wr <= sram_wr;
					userlogic_rd <= sram_rd;
					userlogic_address <= std_logic_vector(unsigned(sram_address) - control_region-1);
				else
					userlogic_wr <= '0';
					userlogic_rd <= '0';
					userlogic_address <= (others => '0');
				end if;
			end if;
		end if;
	end process;
end rtl;
