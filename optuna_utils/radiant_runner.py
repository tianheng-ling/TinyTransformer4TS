import os
import re
import wandb
import optuna
import subprocess
from pathlib import Path
from datetime import datetime

from optuna_utils.run_ghdl_simulation import run_ghdl_simulation


def analyze_resource_utilization(report_path: str):
    report_path = Path(report_path)
    report_info = {}
    if report_path.suffix == ".par":
        with open(report_path, "r") as f:
            for line in f:
                if m := re.match(r"\s+(\S*)\s+(\d+)/(\d+)\s+\d+% used", line):
                    key, used, total = m.groups()
                    used_util = int(used) / int(total)
                    if key in ["LUT", "DSP", "EBR"]:
                        key = key.lower() + "s"
                        report_info[f"{key}_used"] = int(used)
                        report_info[f"{key}_used_util"] = round(used_util * 100, 2)
                        report_info["dsps_total"] = total
        return report_info
    elif report_path.suffix == ".mrp":
        with open(report_path, "rt") as f:
            mrp_report = f.read()
            keywords = ["Number of LUT4s:", "Number of DSPs:", "Number of EBRs:"]
            for line in mrp_report.split("\n"):
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split(":")
                        key = parts[0].strip()

                        used_value = int(parts[1].split("out of")[0].strip())
                        total_value = int(
                            parts[1].split("out of")[1].split("(")[0].strip()
                        )
                        used_util = used_value / total_value

                        if key == "Number of LUT4s":
                            key = "LUT"
                        elif key == "Number of DSPs":
                            key = "DSP"
                        elif key == "Number of EBRs":
                            key = "EBR"
                        key = key.lower() + "s"
                        report_info[key + "_used"] = used_value
                        report_info[key + "_used_util"] = round(used_util * 100, 2)
                        report_info[key + "_total"] = total_value
        return report_info

    elif report_path.suffix == ".srp":
        with open(report_path, "rt") as f:
            report_path = f.read()

            keywords = [
                "Device EBR Count ..............:",
                "Used EBR Count ................:",
                "Number of EBR Blocks Needed ...:",
                "Device Register Count .........:",
                "Number of registers needed ....:",
                "Number of DSP Blocks:",
            ]

            tmp_results = {}
            tmp_results["luts_used"] = 0
            tmp_results["luts_total"] = 5280
            tmp_results["ebrs_used"] = 0
            tmp_results["ebrs_total"] = 20
            tmp_results["ebrs_missing"] = 0
            tmp_results["dsps_used"] = 0
            tmp_results["dsps_total"] = 8

            for line in report_path.split("\n"):
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split(":")
                        key = parts[0].strip()

                        if key == "Device EBR Count ..............":
                            key = "ebrs_total"
                        elif key == "Used EBR Count ................":
                            key = "ebrs_used"
                        elif key == "Number of EBR Blocks Needed ...":
                            key = "ebrs_missing"
                        elif key == "Device Register Count .........":
                            key = "luts_total"
                        elif key == "Number of registers needed ....":
                            key = "luts_used"
                        elif key == "### Number of DSP Blocks":
                            key = "dsps_used"

                        value = int(parts[1].strip())
                        tmp_results[key] = value

            report_info["luts_used"] = tmp_results["luts_used"]
            report_info["luts_used_util"] = round(
                tmp_results["luts_used"] / tmp_results["luts_total"] * 100, 2
            )

            report_info["ebrs_used"] = (
                tmp_results["ebrs_used"] + tmp_results["ebrs_missing"]
            )
            report_info["ebrs_used_util"] = round(
                (tmp_results["ebrs_used"] + tmp_results["ebrs_missing"])
                / tmp_results["ebrs_total"]
                * 100,
                2,
            )
            report_info["dsps_used"] = tmp_results["dsps_used"]
            report_info["dsps_used_util"] = round(
                tmp_results["dsps_used"] / tmp_results["dsps_total"] * 100, 2
            )

        return report_info

    else:
        print(f"[Error] Unsupported file type: {report_path}")
        return None


def analyze_power_consumption(report_path: str):

    if Path(report_path).exists():
        report_info = {}
        with open(report_path, "rt") as f:
            power_report = f.read()

            match = re.search(
                r"Total Power Est.\ Design  : (.+) W, (.+) W, (.+) W", power_report
            )
            static, dynamic, total = match.groups()
            report_info["static_power(mW)"] = float(static) * 1000
            report_info["dynamic_power(mW)"] = float(dynamic) * 1000
            report_info["total_power(mW)"] = float(total) * 1000

        return report_info
    else:
        print(f"File not found: {report_path}")
        return None


def run_resource_estimation(
    tmp_dir: str,
    firmware_dir: str,
    report_dir: str,
    vhd_dir: str,
    const_dir: str,
    data_dir: str,
    tcl_path: str,
    env: dict,
):

    subprocess.run(["cp", "-r", vhd_dir, tmp_dir])
    subprocess.run(["cp", "-r", const_dir, tmp_dir])
    subprocess.run(["cp", "-r", data_dir, tmp_dir])
    subprocess.run(f"cp -r {firmware_dir}/* {tmp_dir}/source", shell=True)
    subprocess.run(["cp", "-r", tcl_path, tmp_dir])
    subprocess.run(
        ["pnmainc", "resource_estimation.tcl", tmp_dir, "env5se_top_reconfig"],
        env=env,
        cwd=tmp_dir,
    )

    report_priority = [
        ("par", "radiant_project_impl_1.par"),
        ("mrp", "radiant_project_impl_1.mrp"),
        ("srp", "radiant_project_impl_1.srp"),
    ]

    for ext, filename in report_priority:
        report_path = os.path.join(tmp_dir, "impl_1", filename)
        if os.path.exists(report_path):
            dst_path = os.path.join(report_dir, f"utilization_report.{ext}")
            subprocess.run(["cp", report_path, dst_path])
            break
    else:
        print("[Error] No .par/.mrp/.srp file found.")
        raise optuna.exceptions.TrialPruned()


def run_power_simulation(tmp_dir: str, report_dir: str, time_fs: int, env: dict):
    power_sim_dir = os.path.join(tmp_dir, "power_simulation")
    os.makedirs(power_sim_dir, exist_ok=True)

    power_sim_tcl = "scripts/optuna/lattice/power_simulation.tcl"
    power_est_tcl = "scripts/optuna/lattice/power_estimation.tcl"
    subprocess.run(["cp", "-r", power_sim_tcl, tmp_dir])
    subprocess.run(["cp", "-r", power_est_tcl, tmp_dir])

    tcl_file_path = os.path.join(tmp_dir, "power_simulation.tcl")
    vo_path = os.path.join(tmp_dir, "impl_1/radiant_project_impl_1_vo.vo")
    if os.path.exists(vo_path):
        subprocess.run(
            [
                "sed",
                "-i",
                f"s/run 2989125 ns/run {time_fs} fs/g",
                tcl_file_path,
            ]
        )
        radiant_root = f"/home/tianhengling/lscc/radiant/2023.2"
        vsim_bin = os.path.join(radiant_root, "modeltech/linuxloem/vsim")
        subprocess.run(
            [vsim_bin, "-c", "-do", "power_simulation.tcl"], env=env, cwd=tmp_dir
        )
        subprocess.run(["pnmainc", "power_estimation.tcl"], env=env, cwd=tmp_dir)
        subprocess.run(["cp", os.path.join(tmp_dir, "power_report.txt"), report_dir])
    else:
        raise FileNotFoundError(".vo file missing — cannot run power simulation")


def get_radiant_env():
    radiant_home = "/home/tianhengling/lscc/radiant/2023.2"
    env = os.environ.copy()
    env["RADIANT_HOME"] = radiant_home
    env["bindir"] = f"{radiant_home}/bin/lin64"
    env["PATH"] = f"{radiant_home}/bin/lin64:" + env.get("PATH", "")
    env["LM_LICENSE_FILE"] = f"{radiant_home}/license/license.dat"
    env["LD_LIBRARY_PATH"] = (
        f"{radiant_home}/bin/lin64:"
        f"{radiant_home}/ispfpga/bin/lin64:" + env.get("LD_LIBRARY_PATH", "")
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_DEBUG_PLUGINS"] = "1"
    return env


def radiant_runner(base_dir: str, top_module: str, trial: object):

    # Setup dirs
    data_dir = os.path.join(base_dir, "data")
    vhd_dir = os.path.join(base_dir, "source")
    const_dir = os.path.join(base_dir, "constraints")
    firmware_dir = os.path.join(base_dir, "firmware")
    makefile_path = os.path.join(base_dir, "makefile")

    tmp_dir = os.path.join(base_dir, "tmp_radiant_proj")
    subprocess.run(["rm", "-rf", tmp_dir])
    os.makedirs(tmp_dir, exist_ok=True)
    report_dir = os.path.join(base_dir, "radiant_report")
    os.makedirs(report_dir, exist_ok=True)

    # [1] Run GHDL simulation WARNING: Just for speedup experiments
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
        gdhl_duration = gdhl_end - gdhl_start
        gdhl_time = round(gdhl_duration.total_seconds() / 60, 2)
        time_ms = time_fs / 1e12  # fs → ms
        wandb.log(
            {
                "time_used(ms)": time_ms,  # fs → ms
                "ghdl_time(min)": gdhl_time,
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
            firmware_dir=firmware_dir,
            report_dir=report_dir,
            vhd_dir=vhd_dir,
            const_dir=const_dir,
            data_dir=data_dir,
            tcl_path="scripts/optuna/lattice/resource_estimation.tcl",
            env=get_radiant_env(),
        )
        resource_est_end = datetime.now()
        resource_est_duration = resource_est_end - resource_est_start
        resource_est_time = round(resource_est_duration.total_seconds() / 60, 2)
        wandb.log({"resource_estimation_time(min)": resource_est_time})

        for ext in ["par", "mrp", "srp"]:
            report_path = os.path.join(report_dir, f"utilization_report.{ext}")
            if os.path.exists(report_path):
                break
        res_info = analyze_resource_utilization(report_path=report_path)
        if res_info is not None:
            wandb.log(res_info)
            if (
                res_info["luts_used"] > 5280
                or res_info["dsps_used"] > 8
                or res_info["ebrs_used"] > 30
            ):
                print(f"[Trial {trial.number}] Pruned due to resource overflow.")
                raise optuna.exceptions.TrialPruned()
        else:
            print("[Error] Failed to parse resource report.")
            raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"[Error] Resource estimation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # [3] Power estimation
    try:
        power_est_start = datetime.now()
        run_power_simulation(
            tmp_dir=tmp_dir,
            report_dir=report_dir,
            time_fs=time_fs,
            env=get_radiant_env(),
        )
        power_est_end = datetime.now()
        power_est_duration = power_est_end - power_est_start
        power_est_time = round(power_est_duration.total_seconds() / 60, 2)
        wandb.log({"power_estimation_time(min)": power_est_time})

        power_info = analyze_power_consumption(
            report_path=os.path.join(report_dir, "power_report.txt")
        )
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
        "power": total_power,
        "latency": time_ms,
        "energy": energy_used,
    }

    return hw_metrics
