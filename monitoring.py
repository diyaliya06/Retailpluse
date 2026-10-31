"""Step 6: data drift detection (older half of the data vs. newer half)."""
import json

import pandas as pd
from scipy.stats import ks_2samp

from src.config import art

COLUMNS = ["Quantity", "Price", "Revenue"]


def drift_report(df: pd.DataFrame, threshold: float = 0.10) -> dict:
    split = df["InvoiceDate"].median()
    before = df[df["InvoiceDate"] <= split]
    after = df[df["InvoiceDate"] > split]

    report = {}
    for col in COLUMNS:
        b, a = float(before[col].mean()), float(after[col].mean())
        rel = abs(a - b) / abs(b) if b else 0.0
        ks_p = float(ks_2samp(before[col], after[col]).pvalue)
        report[col] = {
            "before_mean": b,
            "after_mean": a,
            "relative_change": rel,
            "ks_pvalue": ks_p,
            "drift": bool(rel > threshold),
        }
    return report


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = pd.read_csv(art("cleaned_transactions.csv"),
                         parse_dates=["InvoiceDate"])
    report = drift_report(df)
    with open(art("drift_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    flagged = [k for k, v in report.items() if v["drift"]]
    print("Saved drift_report.json | drift detected in:", flagged or "none")
    return report


if __name__ == "__main__":
    run()
