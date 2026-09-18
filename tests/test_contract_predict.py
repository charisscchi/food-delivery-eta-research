"""Model-contract + prediction tests: rejection, fallback, threshold rule, monotone quantiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.model_contract import ModelContract, summarize_batch
from src.training.train import choose_alert_threshold


# --------------------------------------------------------------- contract
def _normalize_raw_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the fixture's planted raw-dump artifacts (ingestion does this in production)."""
    df = df.copy()
    df["weather"] = df["weather"].astype(str).str.replace("conditions ", "", regex=False).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["rider_age"] = pd.to_numeric(df["rider_age"], errors="coerce").fillna(30.0)
    return df


def test_contract_rejects_missing_critical(synthetic_deliveries):
    contract = ModelContract()
    df = _normalize_raw_artifacts(synthetic_deliveries)
    df.loc[0, "rider_id"] = np.nan          # critical missing -> REJECTED
    df.loc[1, ["rest_lat", "rest_lon"]] = 0.0  # (0,0) coords -> REJECTED
    clean, rejected = contract.validate_row_batch(df)
    assert len(rejected) == 2
    assert (rejected["contract_status"] == "REJECTED").all()
    assert len(clean) == len(df) - 2


def test_contract_fallbacks_repair_out_of_range(synthetic_deliveries):
    contract = ModelContract()
    df = _normalize_raw_artifacts(synthetic_deliveries)
    df.loc[0, "rider_id"] = np.nan          # ensure row 0 is dropped before the age median is taken
    df.loc[1, "rider_age"] = 150.0          # out of range -> repaired to in-range median
    df.loc[2, "weather"] = "Snowing"        # unseen category -> 'unknown'
    clean, rejected = contract.validate_row_batch(df)
    assert len(rejected) == 1
    row1 = clean.loc[1]
    row2 = clean.loc[2]
    expected_fill = float(df.loc[2:, "rider_age"].median())   # median of in-range ages
    assert row1["rider_age"] == expected_fill
    assert row2["weather"] == "unknown"
    assert row1["contract_status"] == "FALLBACK"
    assert row2["contract_status"] == "FALLBACK"
    assert "numeric_fallback:rider_age" in row1["contract_notes"]
    assert "categorical_fallback:weather" in row2["contract_notes"]
    assert (clean.loc[3:, "contract_status"] == "OK").all()


def test_contract_ok_rows_untouched(synthetic_deliveries):
    contract = ModelContract()
    clean, _ = contract.validate_row_batch(_normalize_raw_artifacts(synthetic_deliveries))
    ok = clean[clean["contract_status"] == "OK"]
    assert len(ok) == len(clean)  # fully normalized batch passes clean
    assert ok["contract_notes"].eq("").all()


# ------------------------------------------------------------ alert threshold
def test_choose_alert_threshold_finds_precision_floor():
    rng = np.random.default_rng(42)
    n = 4000
    # scores correlated with the label so a precision-0.8 threshold exists
    scores = np.clip(rng.beta(2, 2, n), 0, 1)
    labels = (scores + rng.normal(0, 0.15, n) > 0.65).astype(int)
    t, m = choose_alert_threshold(scores, labels, min_precision=0.80, max_threshold=0.95)
    assert m["satisfied"] is True
    assert m["precision"] >= 0.80 - 1e-9
    # smallest satisfying threshold -> recall should be meaningful
    assert m["recall"] > 0.05


def test_choose_alert_threshold_unreachable_falls_back_honestly():
    # labels independent of scores -> no threshold reaches 0.9 precision
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 3000)
    labels = rng.integers(0, 2, 3000)
    t, m = choose_alert_threshold(scores, labels, min_precision=0.90, max_threshold=0.95)
    assert m["satisfied"] is False
    assert t == 0.95


# ------------------------------------------------------------------ summary
def test_summarize_batch(synthetic_deliveries):
    df = synthetic_deliveries.copy()
    df["contract_status"] = "OK"
    df["late_flag"] = df["time_taken_min"] > df["time_taken_min"].quantile(0.9)
    s = summarize_batch(df)
    assert s["n_scored"] == 120
    assert s["n_ok"] == 120
    assert 0 <= s["share_late_flagged"] <= 1


def test_pinball_loss_zero_for_perfect_prediction():
    from src.training.train import pinball_loss
    y = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y, y.copy(), 0.5) == pytest.approx(0.0)
    # asymmetric penalty: under-prediction at q=0.9 costs q*gap
    assert pinball_loss(np.array([10.0]), np.array([20.0]), 0.9) == pytest.approx(0.1 * 10)
    assert pinball_loss(np.array([20.0]), np.array([10.0]), 0.9) == pytest.approx(0.9 * 10)
