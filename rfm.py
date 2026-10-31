"""Step 2: RFM features + K-Means clustering + named customer segments."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, art


def _score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """1-5 quintile score (5 = best)."""
    ranks = series.rank(method="first", ascending=higher_is_better)
    return pd.qcut(ranks, 5, labels=False) + 1


def _segment(row) -> str:
    r, f = row["R"], row["F"]
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 3:
        return "Potential Customers"
    if f >= 3 or row["M"] >= 3:
        return "At Risk Customers"
    return "Low Value Customers"


def build_rfm(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    if len(rfm) < 10:
        raise ValueError("Need at least 10 customers for RFM segmentation.")

    # Rule-based segments from RFM quintile scores
    rfm["R"] = _score(rfm["Recency"], higher_is_better=False)  # recent = high
    rfm["F"] = _score(rfm["Frequency"], higher_is_better=True)
    rfm["M"] = _score(rfm["Monetary"], higher_is_better=True)
    rfm["Segment"] = rfm.apply(_segment, axis=1)

    # K-Means clusters on log-scaled RFM
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    X = StandardScaler().fit_transform(X)
    k = min(n_clusters, len(rfm))
    rfm["Cluster"] = KMeans(
        n_clusters=k, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(X)

    return rfm[["CustomerID", "Recency", "Frequency", "Monetary",
                "Cluster", "Segment"]]


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(art("cleaned_transactions.csv"),
                         parse_dates=["InvoiceDate"])
    rfm = build_rfm(df)
    rfm.to_csv(art("rfm_customers.csv"), index=False)
    print(f"Saved artifacts/rfm_customers.csv ({len(rfm)} customers)")
    return rfm


if __name__ == "__main__":
    run()
