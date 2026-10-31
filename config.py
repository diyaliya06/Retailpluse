"""Central paths and settings for RetailPulse.

All paths are built from this file's location, so the project works no matter
which folder you launch it from.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)


def art(name: str) -> str:
    """Full path of a file inside the artifacts folder."""
    return os.path.join(ART_DIR, name)


# Forecasting / inventory settings
FORECAST_HORIZON = 30      # days to forecast
TOP_PRODUCTS = 10          # how many top-revenue products to forecast
LEAD_TIME_DAYS = 7         # supplier lead time used for safety stock
SERVICE_Z = 1.65           # 95% service level

RANDOM_STATE = 42
