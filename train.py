"""Runs the whole RetailPulse pipeline end to end.

    python -m src.train        (from the RetailPulse folder)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import churn, forecasting, inventory, monitoring, preprocessing, rfm  # noqa: E402
from src.config import BASE_DIR  # noqa: E402


def _log_to_mlflow(churn_metrics, forecast_metrics, drift):
    """Optional experiment tracking. Never stops the pipeline if MLflow fails."""
    try:
        import mlflow

        mlflow.set_tracking_uri(Path(BASE_DIR, "mlruns").as_uri())
        mlflow.set_experiment("RetailPulse")
        with mlflow.start_run(run_name="pipeline"):
            if churn_metrics.get("accuracy") is not None:
                mlflow.log_metric("churn_accuracy", churn_metrics["accuracy"])
            if churn_metrics.get("roc_auc") is not None:
                mlflow.log_metric("churn_auc", churn_metrics["roc_auc"])
            best = forecast_metrics[forecast_metrics["Selected"]]
            mlflow.log_metric("forecast_mean_rmse", float(best["RMSE"].mean()))
            mlflow.log_metric("drifted_features",
                              sum(v["drift"] for v in drift.values()))
        print("Logged run to MLflow (folder: mlruns)")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


def main():
    start = time.time()

    print("\n[1/6] Cleaning data")
    df = preprocessing.run()

    print("\n[2/6] RFM segmentation")
    rfm.run(df)

    print("\n[3/6] Churn model")
    churn_metrics = churn.train_churn_model(df)

    print("\n[4/6] Demand forecasting")
    _, forecast_metrics = forecasting.create_forecast(df)

    print("\n[5/6] Inventory optimisation")
    inventory.run()

    print("\n[6/6] Drift monitoring")
    drift = monitoring.run(df)

    _log_to_mlflow(churn_metrics, forecast_metrics, drift)
    print(f"\nDone in {time.time() - start:.0f}s. Now run: "
          "python -m streamlit run src/app.py")


if __name__ == "__main__":
    main()
