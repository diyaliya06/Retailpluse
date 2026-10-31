"""Step 4: demand forecasting for the top products.

Models compared per product on a hold-out period (best RMSE wins):
  - Prophet      (used when installed)
  - GBM          (lag-feature Gradient Boosting; lightweight stand-in for LSTM)
  - Seasonal     (day-of-week average baseline)
  - Hybrid       (average of Prophet/Seasonal and GBM)
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config import FORECAST_HORIZON, RANDOM_STATE, TOP_PRODUCTS, art

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:  # prophet not installed or failed to load
    HAS_PROPHET = False

LAGS = [1, 2, 3, 7, 14]


def _daily(df: pd.DataFrame, product: str) -> pd.Series:
    s = (
        df[df["StockCode"] == product]
        .set_index("InvoiceDate")["Quantity"]
        .resample("D")
        .sum()
    )
    return s.astype(float)


def _future_index(train: pd.Series, horizon: int):
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1),
                         periods=horizon, freq="D")


def _frame(idx, yhat, sd) -> pd.DataFrame:
    yhat = np.maximum(np.asarray(yhat, dtype=float), 0.0)
    return pd.DataFrame({
        "ds": idx,
        "yhat": yhat,
        "yhat_lower": np.maximum(yhat - 1.28 * sd, 0.0),
        "yhat_upper": yhat + 1.28 * sd,
    })


def _seasonal(train: pd.Series, horizon: int) -> pd.DataFrame:
    recent = train[-56:]
    dow_mean = recent.groupby(recent.index.dayofweek).mean()
    idx = _future_index(train, horizon)
    yhat = [dow_mean.get(d.dayofweek, recent.mean()) for d in idx]
    sd = float(np.nan_to_num(recent.std()))
    return _frame(idx, yhat, sd)


def _gbm(train: pd.Series, horizon: int) -> pd.DataFrame:
    vals = train.values.astype(float)
    idx = train.index
    rows, targets = [], []
    for i in range(max(LAGS), len(vals)):
        rows.append([vals[i - l] for l in LAGS]
                    + [idx[i].dayofweek, vals[max(0, i - 7):i].mean()])
        targets.append(vals[i])
    if len(rows) < 10:
        return _seasonal(train, horizon)

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(rows, targets)
    sd = float(np.std(np.array(targets) - model.predict(rows)))

    hist, preds = list(vals), []
    future = _future_index(train, horizon)
    for d in future:
        feats = [hist[-l] for l in LAGS] + [d.dayofweek, float(np.mean(hist[-7:]))]
        p = max(float(model.predict([feats])[0]), 0.0)
        preds.append(p)
        hist.append(p)
    return _frame(future, preds, sd)


def _prophet(train: pd.Series, horizon: int) -> pd.DataFrame:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(train) >= 365,
    )
    model.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = model.predict(model.make_future_dataframe(periods=horizon, freq="D"))
    fc = fc.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        fc[col] = fc[col].clip(lower=0)
    return fc


def _blend(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = a.copy()
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        out[col] = (a[col].values + b[col].values) / 2
    return out


def _run_models(train: pd.Series, horizon: int) -> dict:
    results = {"Seasonal": _seasonal(train, horizon), "GBM": _gbm(train, horizon)}
    if HAS_PROPHET:
        try:
            results["Prophet"] = _prophet(train, horizon)
        except Exception as exc:
            print(f"  Prophet failed: {exc}")
    base = results.get("Prophet", results["Seasonal"])
    results["Hybrid"] = _blend(base, results["GBM"])
    return results


def _metrics(actual: np.ndarray, pred: np.ndarray):
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = actual != 0
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else float("nan")
    return mape, rmse


def create_forecast(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(art("cleaned_transactions.csv"),
                         parse_dates=["InvoiceDate"])

    print(f"Forecasting engine: {'Prophet + GBM + Hybrid' if HAS_PROPHET else 'GBM + Seasonal + Hybrid (Prophet not installed)'}")

    top = df.groupby("StockCode")["Revenue"].sum().nlargest(TOP_PRODUCTS).index
    forecasts, metric_rows = [], []

    for product in top:
        series = _daily(df, product)
        if len(series) < 45:
            print(f"  skip {product}: only {len(series)} days of history")
            continue

        hold = min(FORECAST_HORIZON, len(series) // 4)
        train, test = series[:-hold], series[-hold:]
        scores = {}
        for name, fc in _run_models(train, hold).items():
            scores[name] = _metrics(test.values, fc["yhat"].values[:hold])

        best = min(scores, key=lambda n: scores[n][1])
        for name, (mape, rmse) in scores.items():
            metric_rows.append({
                "StockCode": product, "Model": name,
                "MAPE": round(mape, 2) if not np.isnan(mape) else None,
                "RMSE": round(rmse, 4), "Selected": name == best,
            })

        final = _run_models(series, FORECAST_HORIZON)[best]
        final["StockCode"] = str(product)
        final["Model"] = best
        forecasts.append(final)
        print(f"  {product}: best model = {best} (RMSE {scores[best][1]:.3f})")

    if not forecasts:
        raise RuntimeError("No product had enough daily history (45+ days) to forecast.")

    result = pd.concat(forecasts, ignore_index=True)[
        ["StockCode", "ds", "yhat", "yhat_lower", "yhat_upper", "Model"]
    ]
    metrics = pd.DataFrame(metric_rows)
    result.to_csv(art("product_forecast.csv"), index=False)
    metrics.to_csv(art("forecast_metrics.csv"), index=False)
    print("Saved product_forecast.csv and forecast_metrics.csv")
    return result, metrics


if __name__ == "__main__":
    create_forecast()
