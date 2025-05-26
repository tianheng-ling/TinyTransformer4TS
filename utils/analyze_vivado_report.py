import os
import pandas as pd
from pathlib import Path

from utils.set_paths import parse_path, get_paths_list


def clean_key(key):
    key_mapping = {
        "Slice_LUTs": "luts",
        "LUT_as_Memory": "luts_mem",
        "Block_RAM_Tile": "brams",
        "DSPs": "dsps",
        "Total_On-Chip_Power_(W)": "total_power(mW)",
        "Dynamic_(W)": "dynamic_power(mW)",
        "Device_Static_(W)": "static_power(mW)",
        "Time_taken_for_processing": "total_time(ms)",
    }
    cleaned_key = key.strip("| ").replace(" ", "_")
    return key_mapping.get(cleaned_key, cleaned_key)


def analyze_amd_resource_utilization(source_path: str):

    paths_list = get_paths_list(source_path)

    keywords = ["| Slice LUTs", "|   LUT as Memory", "| Block RAM Tile", "| DSPs"]
    df_list = []

    for path in paths_list:

        # Step 1: get resource utilization report
        report_path = os.path.join(
            path + "/hw/amd/vivado_report/", "utilization_report.txt"
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
        df_list.append(pd.DataFrame([report_info]))
    df = pd.concat(df_list, ignore_index=True)
    return df


def analyze_amd_power_consumption(source_path: str):

    paths_list = get_paths_list(source_path)
    keywords = ["Total On-Chip Power (W)", "Dynamic (W)", "Device Static (W)"]
    df_list = []

    for path in paths_list:

        # Step 1: get power consump report
        report_path = os.path.join(path + "/hw/amd/vivado_report/", "power_report.txt")
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
                    parts = line.split("|")
                    value = float(parts[2].strip())
                    cleaned_keyword = clean_key(keyword)
                    report_info[cleaned_keyword] = value
        df_list.append(pd.DataFrame([report_info]))

    df = pd.concat(df_list, ignore_index=True)

    # convert power values to mW
    power_columns = ["total_power(mW)", "dynamic_power(mW)", "static_power(mW)"]
    df.loc[:, power_columns] *= 1000

    return df


def analyze_amd_timing(source_path: str):
    paths_list = get_paths_list(source_path)

    keywords = ["Time taken for processing"]
    df_list = []
    for path in paths_list:

        # Step 1: get timing report
        report_path = os.path.join(
            path + "/hw/amd/ghdl_report/", "ghdl_network_output.txt"
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
                    cleaned_keyword = clean_key(keyword)
                    report_info[cleaned_keyword] = value

        df_list.append(pd.DataFrame([report_info]))
    df = pd.concat(df_list, ignore_index=True)
    df["total_time(ms)"] = df["total_time(ms)"] * 1000
    return df


def calculate_amd_energy_consumption(df_power, df_timing, source_path):

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
