"""Finds and reads the raw Online Retail file (.xlsx or .csv)."""
import os

import pandas as pd

from src.config import BASE_DIR, DATA_DIR


def find_raw_file():
    """Look for a file named online_retail*.xlsx / .xls / .csv.

    Searched in: RetailPulse/data, RetailPulse, and the folder above it.
    """
    for folder in (DATA_DIR, BASE_DIR, os.path.dirname(BASE_DIR)):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            low = name.lower()
            if low.startswith("online_retail") and low.endswith(
                (".xlsx", ".xls", ".csv")
            ):
                return os.path.join(folder, name)
    return None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _read_excel(path):
    # Online Retail II has several sheets; join them all.
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def load_raw(path=None) -> pd.DataFrame:
    path = path or find_raw_file()
    if path is None:
        raise FileNotFoundError(
            "Raw data not found. Put your file (for example "
            "online_retail.csv.xlsx) inside the RetailPulse/data folder."
        )

    print(f"Reading raw data: {path}")
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return _read_excel(path)
        except Exception:
            return _read_csv(path)  # file may really be a csv
    try:
        return _read_csv(path)
    except Exception:
        return _read_excel(path)  # file may really be an excel
