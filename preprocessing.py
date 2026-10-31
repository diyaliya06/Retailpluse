"""Step 1: clean the raw transactions and save artifacts/cleaned_transactions.csv."""
import pandas as pd

from src.config import art
from src.loader import load_raw

COLUMN_MAP = {
    "InvoiceNo": "Invoice",
    "Invoice": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "UnitPrice": "Price",
    "Price": "Price",
    "Customer ID": "CustomerID",
    "CustomerID": "CustomerID",
    "Country": "Country",
}

REQUIRED = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "CustomerID", "Country",
]

# Non-product lines found in the Online Retail dataset
NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "BANK CHARGES", "C2", "CRUK", "PADS", "AMAZONFEE",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_MAP[str(c).strip()] for c in df.columns
              if str(c).strip() in COLUMN_MAP}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in raw data: {missing}. Found: {list(df.columns)}"
        )
    return df[REQUIRED].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    start = len(df)

    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df = df[~df["Invoice"].str.upper().str.startswith("C")]  # cancellations

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df.dropna(subset=["Quantity", "Price", "InvoiceDate", "CustomerID"])

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    df["Description"] = df["Description"].fillna("Unknown").astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unknown")

    df = df.drop_duplicates()
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Outlier removal: drop the top 0.1% of order-line revenue
    df = df[df["Revenue"] <= df["Revenue"].quantile(0.999)]

    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    print(f"Cleaning: {start:,} rows -> {len(df):,} rows")
    return df


def run() -> pd.DataFrame:
    df = clean(load_raw())
    df.to_csv(art("cleaned_transactions.csv"), index=False)
    print("Saved artifacts/cleaned_transactions.csv")
    return df


if __name__ == "__main__":
    run()
