import os
import wandb
import optuna
import subprocess
from datetime import datetime
from pathlib import Path

from optuna_utils.run_ghdl_simulation import run_ghdl_simulation


def clean_key(key):
    key_mapping = {
        "Slice_LUTs": "luts",
        "LUT_as_Memory": "luts_mem",
        "Block_RAM_Tile": "brams",
        "DSPs": "dsps",
    }
    cleaned_key = key.strip("| ").replace(" ", "_")
    return key_mapping.get(cleaned_key, cleaned_key)


def analyze_resource_utilization(report_path: str):
    keywords = ["| Slice LUTs", "|   LUT as Memory", "| Block RAM Tile", "| DSPs"]
    if Path(report_path).exists():
        with open(report_path, "r") as f:
            lines = f.readlines()
            report_info = {}
            for line in lines:
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split("|")
                        used_value = (
                            float(parts[2].strip()) if parts[2].strip() != "" else 0
                        )
                        total_value = (
                            float(parts[4].strip()) if parts[4].strip() != "" else 0
                        )
                        utils_value = (
                            float(parts[5].strip()) if parts[5].strip() != "" else 0
                        )
                        cleaned_keyword = clean_key(keyword)

                        report_info[cleaned_keyword + "_used"] = used_value
                        report_info[cleaned_keyword + "_total"] = total_value
                        report_info[cleaned_keyword + "_used_util"] = utils_value
            return report_info
    else:
        print(f"File not found: {report_path}")
        return None


def analyze_power_consumption(report_path: str):
    keywords = ["Total On-Chip Power (W)", "Dynamic (W)", "Device Static (W)"]
    power_values = {}

    if Path(report_path).exists():
        with open(report_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split("|")
                        value = float(parts[2].strip()) * 1000  # W → mW
                        if keyword == keywords[0]:
                            cleaned_keyword = "total_power(mW)"
                        elif keyword == keywords[1]:
                            cleaned_keyword = "dynamic_power(mW)"
                        elif keyword == keywords[2]:
                            cleaned_keyword = "static_power(mW)"
                        power_values[cleaned_keyword] = value
        return power_values
    else:
        print(f"File not found: {report_path}")
        return None


def run_resource_estimation(
    tmp_dir: str,
    tcl_path: str,
    report_dir,
    top_module: str,
    vhd_dir: str,
    data_dir: str,
    const_dir: str,
    firmware_dir: str,
):
    subprocess.run(["cp", "-r", tcl_path, tmp_dir])
    subprocess.run(["cp", "-r", vhd_dir, tmp_dir])
    subprocess.run(["cp", "-r", data_dir, tmp_dir])
    subprocess.run(["cp", "-r", const_dir, tmp_dir])
    subprocess.run(
        f"cp -r {firmware_dir}/* {os.path.join(tmp_dir, 'source')}", shell=True
    )
    tb_path = os.path.join(tmp_dir, "source", top_module, f"{top_module}_tb.vhd")
    absolute_data_path = os.path.join(tmp_dir, "data")
    subprocess.run(
        ["sed", "-i", f"s|./data|{absolute_data_path}|g", tb_path], check=True
    )
    vivado_cmd = (
        f"bash -c 'source /tools/Xilinx/Vivado/2019.2/settings64.sh && "
        f"vivado -mode tcl -nolog -nojournal -source resource_estimation.tcl -tclargs {report_dir}'"
    )
    subprocess.run(vivado_cmd, shell=True, cwd=tmp_dir)


def run_power_estimation(
    tmp_dir: str,
    power_tcl_path: str,
    report_dir: str,
    saif_path: str,
    top_module: str,
    time_fs: float,
):
    subprocess.run(["cp", "-r", power_tcl_path, tmp_dir])
    vivado_cmd = (
        f"bash -c 'source /tools/Xilinx/Vivado/2019.2/settings64.sh && "
        f"vivado -mode tcl -nolog -nojournal -source power_estimation.tcl -tclargs {saif_path} {report_dir} {time_fs} {top_module}'"
    )
    subprocess.run(vivado_cmd, shell=True, cwd=tmp_dir)


def vivado_runner(base_dir: str, top_module: str, trial: object):

    # Setup dirs
    data_dir = os.path.join(base_dir, "data")
    vhd_dir = os.path.join(base_dir, "source")
    const_dir = os.path.join(base_dir, "constraints")
    firmware_dir = os.path.join(base_dir, "firmware")
    makefile_path = os.path.join(base_dir, "makefile")

    tmp_dir = os.path.join(base_dir, "tmp_vivado_proj")
    subprocess.run(["rm", "-rf", tmp_dir])
    os.makedirs(tmp_dir, exist_ok=True)
    report_dir = os.path.abspath(os.path.join(base_dir, "vivado_report"))
    os.makedirs(report_dir, exist_ok=True)

    # [1] Run GHDL simulation
    try:
        gdhl_start = datetime.now()
        time_fs = run_ghdl_simulation(
            base_dir=base_dir,
            top_module=top_module,
            vhd_dir=vhd_dir,
            data_dir=data_dir,
            makefile_path=makefile_path,
        )
        gdhl_end = datetime.now()
        gdhl_time_diff = gdhl_end - gdhl_start
        gdhl_time_diff = round(gdhl_time_diff.total_seconds() / 60, 2)
        time_ms = time_fs / 1e12  # fs → ms
        wandb.log(
            {
                "time_used(ms)": time_ms,  # fs → ms
                "ghdl_simulation_time(min)": gdhl_time_diff,
            }
        )
    except Exception as e:
        print(f"[Error] GHDL simulation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # [2] Estimate Resource
    try:
        resource_est_start = datetime.now()
        run_resource_estimation(
            tmp_dir=tmp_dir,
            tcl_path="scripts/optuna/amd/resource_estimation.tcl",
            report_dir=report_dir,
            top_module=top_module,
            vhd_dir=vhd_dir,
            data_dir=data_dir,
            const_dir=const_dir,
            firmware_dir=firmware_dir,
        )
        resource_est_end = datetime.now()
        resource_est_diff = resource_est_end - resource_est_start
        resource_est_time = round(resource_est_diff.total_seconds() / 60, 2)
        wandb.log({"resource_estimation_time(min)": resource_est_time})

        report_path = os.path.join(report_dir, "utilization_report.txt")
        if os.path.exists(report_path):
            res_info = analyze_resource_utilization(report_path)
            if not res_info:
                print("[Error] Failed to parse resource report.")
                raise optuna.exceptions.TrialPruned()
            else:
                wandb.log(res_info)
                if (
                    res_info["luts_used"] > 8000
                    or res_info["luts_mem_used"] > 2400
                    or res_info["brams_used"] > 10
                    or res_info["dsps_used"] > 20
                ):
                    print(f"[Trial {trial.number}] Pruned due to resource overflow.")
                    raise optuna.exceptions.TrialPruned()
        else:
            print(f"[Error] Resource report not found: {report_path}")
            raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"[Error] Resource estimation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # [3] Power estimation
    try:
        power_est_start = datetime.now()
        run_power_estimation(
            tmp_dir=tmp_dir,
            power_tcl_path="scripts/optuna/amd/power_estimation.tcl",
            report_dir=report_dir,
            saif_path=os.path.join(tmp_dir, "sim_wave.saif"),
            top_module=top_module,
            time_fs=time_fs,
        )
        power_est_end = datetime.now()
        power_est_diff = power_est_end - power_est_start
        power_est_time = round(power_est_diff.total_seconds() / 60, 2)
        wandb.log({"power_estimation_time(min)": power_est_time})

        report_path = os.path.join(report_dir, "power_report.txt")
        power_info = analyze_power_consumption(report_path)
        total_power = power_info["total_power(mW)"]
        dynamic_power = power_info["dynamic_power(mW)"]
        static_power = power_info["static_power(mW)"]

        energy_used = round(total_power * time_ms / 1000, 3)
        wandb.log(
            {
                "total_power(mW)": total_power,
                "dynamic_power(mW)": dynamic_power,
                "static_power(mW)": static_power,
                "energy_used(mJ)": energy_used,
            }
        )
    except Exception as e:
        print(f"[Error] Power estimation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    hw_metrics = {
        "power": total_power if total_power is not None else None,
        "latency": time_ms if time_ms is not None else None,
        "energy": energy_used if energy_used is not None else None,
    }
    return hw_metrics
