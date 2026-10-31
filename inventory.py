"""Step 5: inventory optimisation (safety stock + reorder quantity)."""
import math

import numpy as np
import pandas as pd

from src.config import LEAD_TIME_DAYS, SERVICE_Z, art


def build_inventory(forecast: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rmse = (metrics[metrics["Selected"]]
            .set_index("StockCode")["RMSE"].to_dict())

    rows = []
    for code, g in forecast.groupby("StockCode"):
        avg = float(g["yhat"].mean())
        std = float(rmse.get(code, np.nan))
        if np.isnan(std):
            std = float(np.nan_to_num(g["yhat"].std()))

        safety = SERVICE_Z * std * math.sqrt(LEAD_TIME_DAYS)
        expected_30 = avg * 30
        rows.append({
            "StockCode": code,
            "AvgDailyForecast": round(avg, 3),
            "ForecastStd": round(std, 3),
            "SafetyStock": round(safety, 2),
            "ReorderPoint": round(avg * LEAD_TIME_DAYS + safety, 2),
            "Expected30DayDemand": round(expected_30, 2),
            "SuggestedReorderQty": int(math.ceil(expected_30 + safety)),
        })

    inv = pd.DataFrame(rows)
    pct = inv["Expected30DayDemand"].rank(pct=True, method="first")
    inv["Priority"] = np.where(pct > 0.8, "High",
                               np.where(pct > 0.5, "Medium", "Low"))
    return inv.sort_values("Expected30DayDemand", ascending=False).reset_index(drop=True)


def run() -> pd.DataFrame:
    forecast = pd.read_csv(art("product_forecast.csv"))
    metrics = pd.read_csv(art("forecast_metrics.csv"))
    inv = build_inventory(forecast, metrics)
    inv.to_csv(art("inventory_recommendations.csv"), index=False)
    print(f"Saved inventory_recommendations.csv ({len(inv)} products)")
    return inv


if __name__ == "__main__":
    run()
