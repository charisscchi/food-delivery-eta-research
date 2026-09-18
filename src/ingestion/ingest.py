"""Stage 1 — Ingestion.

Reads the raw Kaggle-mirror csv, strips the dataset's prefix artifacts
("conditions Sunny", "(min) 24"), normalises the "NaN " sentinel token to
real nulls, applies the canonical rename map and writes a typed parquet.

Kept deliberately boring and defensive: every column is coerced with
errors->'coerce' so the validation stage (not ingestion) decides which
rows survive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get, load_config
from src.constants import (
    NA_TOKEN,
    RAW_TIME_TAKEN_PREFIX,
    RAW_WEATHER_PREFIX,
    RAW_COLUMN_RENAME,
)

log = logging.getLogger("ingest")


def _strip_prefix(value: object, prefix: str) -> object:
    """'conditions Sunny' -> 'Sunny'; '(min) 24' -> '24'. Non-strings pass through."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return value


def _clean_token(value: object) -> object:
    """Trim whitespace and map the dump's 'NaN' sentinel to a true null."""
    if isinstance(value, str):
        s = value.strip()
        return np.nan if s == NA_TOKEN else s
    return value


def load_raw_csv(raw_path: str | Path) -> pd.DataFrame:
    """Read the raw csv, rename to canonical columns, strip prefix junk, coerce types."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw csv not found at {raw_path}. Run scripts/fetch_data.py first."
        )

    df = pd.read_csv(raw_path)

    missing = [c for c in RAW_COLUMN_RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"Raw csv missing expected columns: {missing}")

    df = df.rename(columns=RAW_COLUMN_RENAME)

    # strip embedded prefixes before any type coercion
    df["weather_raw"] = df["weather_raw"].map(
        lambda v: _strip_prefix(v, RAW_WEATHER_PREFIX)
    )
    df["time_taken_raw"] = df["time_taken_raw"].map(
        lambda v: _strip_prefix(v, RAW_TIME_TAKEN_PREFIX)
    )
    # whitespace + 'NaN' sentinel -> real null, everywhere
    df = df.map(_clean_token)

    # types
    df["order_date"] = pd.to_datetime(df["order_date_raw"], format="%d-%m-%Y", errors="coerce")
    df["rider_age"] = pd.to_numeric(df["rider_age"], errors="coerce")
    df["rider_rating"] = pd.to_numeric(df["rider_rating"], errors="coerce")
    for c in ["rest_lat", "rest_lon", "dest_lat", "dest_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["vehicle_condition"] = pd.to_numeric(df["vehicle_condition"], errors="coerce")
    df[TARGET := "time_taken_min"] = pd.to_numeric(df["time_taken_raw"], errors="coerce")

    # categorical casing normalisation (dataset mixes 'Urban ' / 'Metropolitian')
    for c in ["road_traffic_density", "city", "type_of_order", "type_of_vehicle", "festival"]:
        df[c] = df[c].astype("string").str.strip()

    df = df.rename(columns={"weather_raw": "weather"})

    # raw leftovers are superseded by their coerced twins; drop so the
    # interim schema matches INTERIM_COLUMNS exactly (validation gate)
    df = df.drop(columns=["order_date_raw", "time_taken_raw"])

    log.info("Ingested %s rows from %s", len(df), raw_path.name)
    return df


def run(stage: str = "ingest", cfg: dict | None = None) -> Path:
    """Ingestion entry point: raw csv -> typed interim parquet."""
    cfg = cfg or load_config()
    interim_path = Path(get(cfg, "paths.interim_data"))

    df = load_raw_csv(get(cfg, "paths.raw_data"))

    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_path, index=False)
    log.info("Wrote interim parquet: %s (%d rows)", interim_path, len(df))
    return interim_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
