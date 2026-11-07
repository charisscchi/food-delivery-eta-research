"""Stage 3 — Feature engineering.

Three groups of features, mirroring how delivery-time models are built
on Swiggy's ds_time_prediction team:

  1. Geo        haversine_km, lat/lon diffs (rider has to physically cover this)
  2. Temporal   order_hour, day_of_week, weekend, lunch/dinner peaks, pickup delay
  3. Rider      rider_age, rating, multiple_deliveries + rider_hist_median_min
                (a *target-encoded* historical speed, fit on TRAIN only and
                 joined onto test/new — the classic leakage trap is handled
                 by fit/apply separation, see fit_rider_history).

Also owns the TIME-BASED splits (train / test / new-data window on
Order_Date — never random) and the late-label definition
(time_taken_min > train P75).

Outputs in data/processed/:
  train.parquet, test.parquet, new_data.parquet, splits.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get, load_config
from src.constants import (
    CATEGORICAL_FEATURES,
    LATE_LABEL_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
)

log = logging.getLogger("features")

EARTH_RADIUS_KM = 6371.0


# ------------------------------------------------------------------ geo math
def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorised haversine distance in km between restaurant and drop location."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------- row-level feats
def build_core_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Row-level features; uses NO target information, so safe at inference."""
    cfg = cfg or load_config()
    df = df.copy()

    # ---- geo -------------------------------------------------------------
    dist = haversine_km(df["rest_lat"], df["rest_lon"], df["dest_lat"], df["dest_lon"])
    clip_km = get(cfg, "features.haversine.clip_km", 60)
    df["haversine_km"] = dist.clip(upper=clip_km).fillna(0.0)
    df["lat_diff"] = (df["dest_lat"] - df["rest_lat"]).fillna(0.0)
    df["lon_diff"] = (df["dest_lon"] - df["rest_lon"]).fillna(0.0)

    # ---- temporal --------------------------------------------------------
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    def _parse_time(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, format="%H:%M:%S", errors="coerce").dt.hour

    ordered_h = _parse_time(df["time_ordered"])
    picked_h = _parse_time(df["time_picked"])
    # fallback for '24:HH:MM' style artifacts (dataset has a few)
    mask24 = df["time_ordered"].astype("string").str.startswith("24")
    if mask24.any():
        alt = pd.to_datetime(
            "00" + df["time_ordered"].astype("string").str[2:], format="%H:%M:%S", errors="coerce"
        ).dt.hour
        ordered_h = ordered_h.mask(mask24, alt)

    df["order_hour"] = ordered_h.fillna(df["order_hour"] if "order_hour" in df else 12.0)
    df["day_of_week"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.day_of_week.isin([5, 6]).astype(int)

    # lunch (11-14) and dinner (18-22) demand peaks
    df["is_lunch_peak"] = df["order_hour"].between(11, 14).astype(int)
    df["is_dinner_peak"] = df["order_hour"].between(18, 22).astype(int)

    # pickup delay: minutes from order to pickup, clipped
    ordered_dt = pd.to_datetime(df["time_ordered"], format="%H:%M:%S", errors="coerce")
    picked_dt = pd.to_datetime(df["time_picked"], format="%H:%M:%S", errors="coerce")
    delay = (picked_dt - ordered_dt).dt.total_seconds() / 60.0
    # negative delays (clock skew) and overnight pickups -> clip to [0, cap]
    clip_min = get(cfg, "features.pickup_delay.clip_min", 180)
    df["pickup_delay_min"] = delay.clip(lower=0, upper=clip_min).fillna(10.0)

    # ---- rider -----------------------------------------------------------
    # NOTE: rider_age / rider_rating medians are NOT imputed here — that would
    # fit the imputer on test/new rows too (leakage). The split happens first
    # in run(); train medians are joined in _impute_train_medians below.
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0).clip(lower=0)

    # rider's home city (parsed from rider_id prefix, e.g. INDORES13DEL02 -> city code)
    df["rider_city"] = (
        df["rider_id"].astype("string").str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    )

    return df


# --------------------------------------------------- rider history (leak-safe)
def fit_rider_history(train_df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.Series, float]:
    """Fit per-rider historical delivery median on TRAIN rows only.

    Returns (rider -> median map, global median fallback). Never call on
    test/new data — that would leak the target.
    """
    cfg = cfg or load_config()
    min_orders = get(cfg, "features.rider_history.min_orders", 5)

    grouped = train_df.groupby("rider_id")[TARGET_COL].agg(["median", "count"])
    eligible = grouped[grouped["count"] >= min_orders]
    rider_map = eligible["median"]
    global_median = float(train_df[TARGET_COL].median())
    log.info(
        "Rider history: %d/%d riders eligible (>= %d train orders); global fallback median=%.1f min",
        len(rider_map), grouped.shape[0], min_orders, global_median,
    )
    return rider_map, global_median


def apply_rider_history(df: pd.DataFrame, rider_map: pd.Series, global_median: float) -> pd.DataFrame:
    """Join the train-fit rider history onto any dataframe, with fallback."""
    df = df.copy()
    df["rider_hist_median_min"] = (
        df["rider_id"].map(rider_map).fillna(global_median).astype(float)
    )
    return df


def fit_train_medians(train_df: pd.DataFrame) -> dict[str, float]:
    """Train-only medians for rider_age / rider_rating (imputation stats)."""
    return {
        "rider_age": float(train_df["rider_age"].median()),
        "rider_rating": float(train_df["rider_rating"].median()),
    }


def apply_train_medians(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Apply train-fit medians to any split (leak-safe imputation)."""
    df = df.copy()
    for col, med in medians.items():
        df[col] = df[col].fillna(med)
    return df


# ----------------------------------------------------------------- splitting
def make_splits(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """TIME-BASED split on order_date: train (earliest 70%) / test (next 15%) /
    new_data (last 15%, simulating a fresh production batch).

    Rows with unparseable dates are dropped (validation gates caught most).
    """
    cfg = cfg or load_config()
    train_share = get(cfg, "features.splits.train_share", 0.70)
    test_share = get(cfg, "features.splits.test_share", 0.15)

    df = df.dropna(subset=["order_date"]).sort_values(["order_date", "ID"]).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_share)
    test_end = int(n * (train_share + test_share))

    train, test, new = df.iloc[:train_end], df.iloc[train_end:test_end], df.iloc[test_end:]

    # enforce strict date monotonicity across split boundaries
    splits_info = {
        "n_train": len(train), "n_test": len(test), "n_new": len(new),
        "train_date_range": [str(train["order_date"].min().date()), str(train["order_date"].max().date())],
        "test_date_range": [str(test["order_date"].min().date()), str(test["order_date"].max().date())],
        "new_date_range": [str(new["order_date"].min().date()), str(new["order_date"].max().date())],
        "late_threshold_min": None,  # filled by caller
        "method": "time-based on order_date (no random leakage)",
    }
    assert train["order_date"].max() <= test["order_date"].min() + pd.Timedelta(days=1)
    assert test["order_date"].max() <= new["order_date"].min() + pd.Timedelta(days=1)
    return train, test, new, splits_info


# ---------------------------------------------------------------------- entry
def run(stage: str = "features", cfg: dict | None = None) -> dict:
    """Feature entry point: interim parquet -> processed train/test/new parquets + splits.json."""
    cfg = cfg or load_config()
    interim_path = Path(get(cfg, "paths.interim_data"))
    processed_dir = Path(get(cfg, "paths.processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(interim_path)

    # drop rows without target or date — they cannot train or evaluate
    df = df.dropna(subset=[TARGET_COL, "order_date"]).reset_index(drop=True)

    df = build_core_features(df, cfg)

    # split FIRST, then fit rider history + imputation medians on train only
    train, test, new, splits_info = make_splits(df, cfg)

    rider_map, global_median = fit_rider_history(train, cfg)
    medians = fit_train_medians(train)
    for split_df_name, split_df in (("train", train), ("test", test), ("new", new)):
        split_df = apply_rider_history(split_df, rider_map, global_median)
        split_df = apply_train_medians(split_df, medians)
        if split_df_name == "train":
            train = split_df
        elif split_df_name == "test":
            test = split_df
        else:
            new = split_df

    # late label: train P75 as the threshold (applied to all splits)
    late_threshold = float(train[TARGET_COL].quantile(0.75))
    for split_df in (train, test, new):
        split_df[LATE_LABEL_COL] = (split_df[TARGET_COL] > late_threshold).astype(int)
    splits_info["late_threshold_min"] = round(late_threshold, 2)

    # impute remaining categoricals as 'unknown' so downstream encoders never see NaN
    for split_df in (train, test, new):
        for c in CATEGORICAL_FEATURES:
            if c in split_df.columns:
                split_df[c] = split_df[c].astype("string").fillna("unknown").astype(str)

    train.to_parquet(processed_dir / "train.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    new.to_parquet(processed_dir / "new_data.parquet", index=False)
    with open(processed_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)

    log.info(
        "Features written: train=%d test=%d new=%d | late threshold=%.1f min | splits.json saved",
        len(train), len(test), len(new), late_threshold,
    )
    return splits_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
