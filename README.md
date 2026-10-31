# RetailPulse - AI-Powered Customer Analytics & Demand Forecasting

Pipeline: Raw data -> Cleaning -> RFM + K-Means -> Churn model -> Forecasting
(Prophet / GBM / Hybrid) -> Inventory optimisation -> Drift monitoring -> Streamlit dashboard.

## Folder layout

```
RetailPulse/
  data/                  <- put online_retail.csv.xlsx (or .csv) here
  artifacts/             <- generated outputs (csv, json, model)
  src/
    config.py  loader.py  preprocessing.py  rfm.py  churn.py
    forecasting.py  inventory.py  monitoring.py  train.py  app.py
  requirements.txt  Dockerfile  README.md
```

## Easiest way (Windows)

1. Copy your data file (online_retail.csv.xlsx) into the `data` folder.
2. Double-click `run.bat`.

## Run manually (VS Code terminal)

```
cd D:\project1\RetailPulse
pip install -r requirements.txt
python -m src.train
python -m streamlit run src/app.py
```

Open http://localhost:8501

Always run the commands from inside the `RetailPulse` folder.

## Docker

```
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

## Notes

- Churn label = "no purchase in the last N days"; features only use data before
  that cutoff, so the model does not simply learn the label from Recency.
- If Prophet is not installed, forecasting still runs with GBM + Seasonal + Hybrid.
- View MLflow runs with `mlflow ui` (run from the RetailPulse folder).
