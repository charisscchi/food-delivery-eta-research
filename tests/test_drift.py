"""Drift (PSI) tests: identical -> 0, shifted -> large, gate behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.drift import compute_drift_table, drift_gate, psi_categorical, psi_numeric


def test_psi_identical_is_zero():
    x = np.random.default_rng(42).normal(30, 8, 2000)
    assert psi_numeric(x, x) < 1e-9


def test_psi_shifted_mean_is_large():
    rng = np.random.default_rng(42)
    base = rng.normal(30, 8, 5000)
    shifted = rng.normal(45, 8, 5000)  # +15 min mean shift
    assert psi_numeric(base, shifted) > 0.2


def test_psi_categorical_shift():
    e = ["Sunny"] * 900 + ["Stormy"] * 100
    a = ["Sunny"] * 500 + ["Stormy"] * 500
    assert psi_categorical(e, a) > 0.2


def test_drift_table_and_gate(tmp_path, tiny_config):
    rng = np.random.default_rng(0)
    n = 3000
    ref = pd.DataFrame({
        "rider_age": rng.normal(30, 5, n),
        "rider_rating": rng.normal(4.7, 0.3, n),
        "haversine_km": rng.gamma(2, 2, n),
        "order_hour": rng.integers(8, 23, n).astype(float),
        "pickup_delay_min": rng.normal(10, 4, n),
        "multiple_deliveries": rng.integers(0, 4, n).astype(float),
        "rider_hist_median_min": rng.normal(30, 6, n),
        "time_taken_min": rng.normal(30, 8, n),
        "weather": rng.choice(["Sunny", "Stormy", "Cloudy"], n, p=[0.7, 0.1, 0.2]),
        "road_traffic_density": rng.choice(["Low", "Medium", "High"], n, p=[0.3, 0.4, 0.3]),
        "city": rng.choice(["Urban", "Semi-Urban"], n, p=[0.6, 0.4]),
        "type_of_vehicle": rng.choice(["motorcycle", "scooter"], n),
        "type_of_order": rng.choice(["Meal", "Snack"], n),
        "festival": ["No"] * n,
    })
    # mild shift: everything from the same-ish distribution, weather mix shifted hard
    batch = ref.sample(frac=0.4, random_state=7).copy()
    batch["weather"] = np.random.default_rng(1).choice(["Sunny", "Stormy"], len(batch), p=[0.2, 0.8])

    table = compute_drift_table(ref, batch, tiny_config)
    assert "psi" in table.columns and len(table) > 0
    assert set(table["status"]).issubset({"OK", "WARN", "FAIL"})

    gate = drift_gate(table, tiny_config)
    assert gate["passed"] is False                      # weather PSI should FAIL
    assert "weather" in gate["failing_features"]

    # and a clean batch passes
    clean_batch = ref.sample(frac=0.4, random_state=8).copy()
    clean_table = compute_drift_table(ref, clean_batch, tiny_config)
    assert drift_gate(clean_table, tiny_config)["passed"] is True
