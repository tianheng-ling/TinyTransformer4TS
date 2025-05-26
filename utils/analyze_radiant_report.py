import re
import os
import pandas as pd
from pathlib import Path

from utils.set_paths import parse_path, get_paths_list


def clean_key(key):
    return key.strip(" ...")[0]


def analyze_lattice_resource_utilization(source_path: str):

    paths_list = get_paths_list(source_path)
    df_list = []

    for path in paths_list:

        report_info = {}
        report_info.update(parse_path(path))

        par_report_path = os.path.join(
            path + "/hw/lattice/radiant_report/", "resource_report.par"
        )

        with open(par_report_path, "rt") as f:
            par_report = f.read()
        for line in par_report.split("\n"):
            if m := re.match("\s+(\S*)\s+(\d+)/(\d+)\s+\d+% used", line):
                key, used, total = m.groups()
                used_relative = int(used) / int(total)
                if key in ["LUT", "DSP", "EBR"]:
                    key = key.lower() + "s"
                    report_info[key + "_used"] = int(used)
                    report_info[key + "_used_util"] = round(used_relative * 100, 4)
                report_info["dsps_total"] = 8
                report_info["luts_total"] = 5280
                report_info["ebr_total"] = 30
        df_list.append(pd.DataFrame([report_info]))
    df = pd.concat(df_list, ignore_index=True)
    return df


def analyze_lattice_power_consumption(source_path: str):

    paths_list = get_paths_list(source_path)
    df_list = []

    for path in paths_list:

        report_info = {}
        report_info.update(parse_path(path))
        par_report_path = os.path.join(
            path + "/hw/lattice/radiant_report/", "power_estimation_report.txt"
        )

        with open(par_report_path, "rt") as f:
            power_report = f.read()

            match = re.search(
                r"Total Power Est.\ Design  : (.+) W, (.+) W, (.+) W", power_report
            )
            static, dynamic, total = match.groups()
            report_info["static_power(mW)"] = float(static) * 1000
            report_info["dynamic_power(mW)"] = float(dynamic) * 1000
            report_info["total_power(mW)"] = float(total) * 1000

        df_list.append(pd.DataFrame([report_info]))
    df = pd.concat(df_list, ignore_index=True)
    return df


def analyze_lattice_timing(source_path: str):

    paths_list = get_paths_list(source_path)
    df_list = []

    keywords = ["Time taken for processing"]
    df_list = []
    for path in paths_list:

        # Step 1: get timing report
        report_path = os.path.join(
            path + "/hw/lattice/ghdl_report/", "ghdl_network_output.txt"
        )
        if Path(report_path).exists():
            with open(report_path, "r") as f:
                lines = f.readlines()
        else:
            print(f"File not found: {report_path}")
            continue

        # Step 2: parse the report
        report_info = {}
        report_info.update(parse_path(path))  # include the info from the path
        for line in lines:
            for keyword in keywords:
                if keyword in line:
                    parts = line.split("=")
                    value = float(parts[1].strip().split(" ")[0])
                    value = value / 1000000000000
                    report_info[keyword] = value

        df_list.append(pd.DataFrame([report_info]))
    df = pd.concat(df_list, ignore_index=True)
    df["total_time(ms)"] = df["Time taken for processing"] * 1000
    df.drop(columns=["Time taken for processing"], inplace=True)
    return df


def calculate_lattice_energy_consumption(df_power, df_timing, source_path):

    paths_list = get_paths_list(source_path)
    parsed_info = parse_path(paths_list[0])
    df_energy = pd.merge(
        df_power,
        df_timing,
        on=list(parsed_info.keys()),
    )
    df_energy["total_energy(mJ)"] = (
        df_energy["total_power(mW)"] * df_energy["total_time(ms)"]
    )
    df_energy["total_energy(mJ)"] = df_energy["total_energy(mJ)"].round(3)
    return df_energy
