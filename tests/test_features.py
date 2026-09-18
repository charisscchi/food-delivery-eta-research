"""Feature-engineering tests: haversine, splits monotonicity, leak-safe rider history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LATE_LABEL_COL, TARGET_COL
from src.features.build_features import (
    apply_rider_history,
    build_core_features,
    fit_rider_history,
    haversine_km,
    make_splits,
)


def test_haversine_known_distance():
    # ~111 km per degree of latitude at the equator; identical points -> 0
    s = pd.Series([0.0, 5.0])
    d = haversine_km(s, pd.Series([0.0, 5.0]), pd.Series([1.0, 5.0]), pd.Series([0.0, 5.0]))
    assert abs(d.iloc[0] - 111.19) < 0.5
    assert d.iloc[1] == 0.0


def test_build_core_features_adds_expected_columns(synthetic_deliveries):
    df = build_core_features(synthetic_deliveries.copy())
    for col in ["haversine_km", "order_hour", "is_weekend", "pickup_delay_min", "rider_city"]:
        assert col in df.columns
    # no nulls introduced in the new feature columns
    assert df["haversine_km"].notna().all()
    assert df["order_hour"].notna().all()


def test_rider_history_is_leak_safe(synthetic_deliveries, tiny_config):
    """Rider stats must be computed from train only; unseen riders get the global median."""
    df = build_core_features(synthetic_deliveries.copy())
    rider_map, global_median = fit_rider_history(df, tiny_config)
    # all 3 synthetic riders have 40 orders each -> eligible
    assert len(rider_map) == 3
    applied = apply_rider_history(df, rider_map, global_median)
    # an unseen rider falls back to global median
    unseen = pd.DataFrame({"rider_id": ["NEWDEL99"]})
    applied_unseen = apply_rider_history(unseen, rider_map, global_median)
    assert applied_unseen["rider_hist_median_min"].iloc[0] == global_median


def test_splits_time_monotone_no_leakage(synthetic_deliveries, tiny_config):
    """Train dates must never exceed test dates; test must never exceed new dates."""
    df = build_core_features(synthetic_deliveries.copy())
    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["order_date"])
    train, test, new, info = make_splits(df, tiny_config)
    assert train["order_date"].max() <= test["order_date"].max()
    assert test["order_date"].max() <= new["order_date"].max()
    assert info["n_train"] + info["n_test"] + info["n_new"] == len(df)
    # strict: no overlap via ID
    ids = pd.concat([train["ID"], test["ID"], new["ID"]])
    assert ids.duplicated().sum() == 0
