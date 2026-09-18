"""Shared pytest fixtures — tiny synthetic deliveries, no real data needed."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def synthetic_deliveries() -> pd.DataFrame:
    """120 rows across 3 riders and 10 dates; clean-ish but with a few
    deliberate artifacts (NaN token, whitespace, prefix junk) that the
    real dataset also carries."""
    rows = []
    for i in range(120):
        rows.append({
            "ID": f"ORD{i:04d}",
            "rider_id": f"CITY{i % 3}DEL01",
            "rider_age": float(20 + (i % 15)),
            "rider_rating": 4.0 + (i % 10) / 10.0,
            "rest_lat": 12.9 + (i % 7) * 0.01,
            "rest_lon": 77.5 + (i % 5) * 0.01,
            "dest_lat": 12.9 + (i % 7) * 0.01 + 0.02,
            "dest_lon": 77.5 + (i % 5) * 0.01 + 0.02,
            "order_date": f"{(i % 10) + 1:02d}-03-2022",   # string dates, raw %d-%m-%Y format
            "time_ordered": f"{10 + (i % 5)}:30:00",
            "time_picked": f"{10 + (i % 5)}:45:00",
            "weather": ["Sunny", "Stormy", "Cloudy"][i % 3],
            "road_traffic_density": ["Low", "Medium", "High", "Jam"][i % 4],
            "vehicle_condition": i % 3,
            "type_of_order": ["Snack", "Meal", "Drinks"][i % 3],
            "type_of_vehicle": ["motorcycle", "scooter"][i % 2],
            "multiple_deliveries": i % 3,
            "festival": "No",
            "city": ["Urban", "Metropolitian", "Semi-Urban"][i % 3],
            "time_taken_min": float(20 + (i % 25) + (i % 4) * 2),
        })
    df = pd.DataFrame(rows)
    # plant raw-dump artifacts the way the raw csv carries them.
    # NOTE: the real file's 'NaN ' sentinel is parsed as null by read_csv
    # itself, so a true NaN here reproduces the actual ingestion path (the
    # sentinel-string mapping is unit-tested separately in test_clean_token).
    df.loc[0, "rider_age"] = np.nan              # missing value
    df.loc[1, "city"] = "Urban "                 # trailing space
    df.loc[2, "weather"] = "conditions Stormy"  # prefix junk
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    df = pd.read_csv(buf)                        # dtypes as the raw file arrives
    return df


@pytest.fixture(scope="session")
def tiny_config() -> dict:
    """Config fixture for tests: minimal knobs, tmp-safe defaults."""
    from src.config import load_config

    cfg = load_config()
    cfg["training"]["cv"]["folds"] = 2
    cfg["training"]["quantiles"] = [0.1, 0.5, 0.9]
    return cfg
