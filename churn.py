"""Step 3: churn prediction.

Label design (no leakage):
  - Pick a cutoff date `window` days before the last transaction.
  - Features (Recency, Frequency, Monetary, ...) use only data BEFORE the cutoff.
  - Churn = 1 if the customer made NO purchase after the cutoff.
The trained model then scores every customer using today's data.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import RANDOM_STATE, art

FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]


def _features(tx: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    f = (
        tx.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            FirstDate=("InvoiceDate", "min"),
        )
        .reset_index()
    )
    f["AvgOrderValue"] = f["Monetary"] / f["Frequency"]
    f["TenureDays"] = (snapshot - f["FirstDate"]).dt.days
    return f.drop(columns="FirstDate")


def _build_training_set(df: pd.DataFrame):
    max_date = df["InvoiceDate"].max()
    span = (max_date - df["InvoiceDate"].min()).days

    fallback = (None, None)
    for window in (90, 60, 45, 30, 21, 14):
        if span < window * 2:
            continue
        cutoff = max_date - pd.Timedelta(days=window)
        past = df[df["InvoiceDate"] <= cutoff]
        future = df[df["InvoiceDate"] > cutoff]
        if past.empty:
            continue
        train = _features(past, cutoff + pd.Timedelta(days=1))
        active = set(future["CustomerID"])
        train["Churn"] = (~train["CustomerID"].isin(active)).astype(int)
        counts = train["Churn"].value_counts()
        if len(counts) < 2 or counts.min() < 5:
            continue
        rate = counts[1] / len(train)
        if 0.15 <= rate <= 0.70:      # a healthy, learnable churn rate
            return train, window
        if fallback[0] is None:       # otherwise keep the first usable window
            fallback = (train, window)
    return fallback


def train_churn_model(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(art("cleaned_transactions.csv"),
                         parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(art("rfm_customers.csv"))

    features = FEATURES
    train, window = _build_training_set(df)

    if train is None:
        print("WARNING: not enough history for a time-based churn label. "
              "Using a Recency-median proxy label (Recency not used as a feature).")
        snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        train = _features(df, snapshot)
        train["Churn"] = (train["Recency"] > train["Recency"].median()).astype(int)
        features = [f for f in FEATURES if f != "Recency"]
        window = None

    X, y = train[features], train["Churn"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=RANDOM_STATE,
    )

    # Honest evaluation: stratified cross-validation on the whole training set
    n_splits = int(min(5, y.value_counts().min()))
    auc = accuracy = None
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        cv_pred = (cv_proba >= 0.5).astype(int)
        accuracy = float(accuracy_score(y, cv_pred))
        auc = float(roc_auc_score(y, cv_proba))
        print(f"Churn model ({n_splits}-fold CV): accuracy {accuracy:.3f}, AUC {auc:.3f}")
        print(f"Churn rate in training data: {y.mean():.1%}")
        print(classification_report(y, cv_pred, zero_division=0))

    model.fit(X, y)

    # Score every customer with their current data
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    current = _features(df, snapshot)
    proba = model.predict_proba(current[features])[:, list(model.classes_).index(1)]

    out = current[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()
    out["Churn"] = (proba >= 0.5).astype(int)
    out["ChurnProbability"] = proba.round(4)
    out = out.merge(rfm[["CustomerID", "Segment"]], on="CustomerID", how="left")
    out.to_csv(art("churn_predictions.csv"), index=False)

    importance = (
        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    importance.to_csv(art("churn_feature_importance.csv"), index=False)

    joblib.dump(model, art("churn_model.pkl"))

    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc,
        "label_window_days": window,
        "features": features,
        "training_rows": int(len(X)),
        "churn_rate_in_training": float(y.mean()),
        "evaluation": f"{n_splits}-fold cross-validation" if n_splits >= 2 else "not enough churners",
    }
    with open(art("churn_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Saved churn_predictions.csv, churn_feature_importance.csv, churn_model.pkl")
    return metrics


if __name__ == "__main__":
    train_churn_model()
