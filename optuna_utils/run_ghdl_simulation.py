import os
import subprocess
from pathlib import Path


def run_ghdl_simulation(
    base_dir: str,
    top_module: str,
    vhd_dir: str,
    data_dir: str,
    makefile_path: str,
):
    tmp_ghdl_dir = os.path.join(base_dir, "tmp_ghdl_proj")
    ghdl_report_dir = os.path.join(base_dir, "ghdl_report")
    os.makedirs(tmp_ghdl_dir, exist_ok=True)
    os.makedirs(ghdl_report_dir, exist_ok=True)

    subprocess.run(["rm", "-rf", tmp_ghdl_dir])
    subprocess.run(["mkdir", tmp_ghdl_dir])
    subprocess.run(["cp", "-r", vhd_dir, tmp_ghdl_dir])
    subprocess.run(["cp", "-r", data_dir, tmp_ghdl_dir])
    subprocess.run(["cp", makefile_path, tmp_ghdl_dir])

    tmp_ghdl_source_dir = os.path.join(tmp_ghdl_dir, "source")

    for dirpath, _, _ in os.walk(tmp_ghdl_source_dir):
        if Path(dirpath).parent == Path(tmp_ghdl_source_dir):
            module = os.path.basename(dirpath)
            subprocess.run(["make", f"TESTBENCH={module}"], cwd=tmp_ghdl_dir)
            subprocess.run(
                [
                    "cp",
                    os.path.join(tmp_ghdl_dir, ".simulation", "make_output.txt"),
                    os.path.join(ghdl_report_dir, f"ghdl_{module}_output.txt"),
                ]
            )

    report_path = os.path.join(ghdl_report_dir, f"ghdl_{top_module}_output.txt")
    with open(report_path, "r") as f:
        for line in f:
            if "Time taken for processing" in line:
                try:
                    time_fs = float(line.split("=")[1].split("fs")[0].strip())
                    return time_fs
                except Exception as e:
                    print("[Error] Failed to parse simulation time:", e)
                    break
    print("[Warning] Using fallback time = 1 ms")
    return 1.0
