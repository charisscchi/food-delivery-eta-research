"""Drift monitoring — PSI (Population Stability Index).

Gate pattern from Swiggy DSP on-call workflows: before a fresh batch is
scored, its feature distribution is compared against the training
distribution. Any feature with PSI > threshold blocks scoring (exit 2)
with a full drift report, instead of silently serving degraded scores.

PSI interpretation (industry standard):
    < 0.10  stable
  0.10-0.20 moderate shift — WARN
    > 0.20  significant shift — FAIL (block scoring)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import get, load_config

log = logging.getLogger("drift")


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI for a numeric feature using quantile bins fit on the expected (train) distribution."""
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    if len(expected) < 50 or len(actual) < 50:
        return 0.0  # too little data to judge drift — do not block on it

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # degenerate (constant) feature
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0]
    a_counts = np.histogram(actual, bins=edges)[0]
    e_share = np.maximum(e_counts / len(expected), eps)
    a_share = np.maximum(a_counts / len(actual), eps)
    psi = float(np.sum((a_share - e_share) * np.log(a_share / e_share)))
    return psi


def psi_categorical(expected: Sequence, actual: Sequence, eps: float = 1e-4) -> float:
    """PSI for a categorical feature over its category shares."""
    e = pd.Series(list(expected)).dropna()
    a = pd.Series(list(actual)).dropna()
    if len(e) < 50 or len(a) < 50:
        return 0.0
    cats = sorted(set(e.unique()) | set(a.unique()))
    e_share = np.array([(e == c).mean() for c in cats], dtype=float)
    a_share = np.array([(a == c).mean() for c in cats], dtype=float)
    e_share = np.maximum(e_share, eps)
    a_share = np.maximum(a_share, eps)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def compute_drift_table(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """PSI for every monitored feature: reference (train) vs batch (new data)."""
    cfg = cfg or load_config()
    numeric_feats = get(cfg, "drift.numeric_features", [])
    cat_feats = get(cfg, "drift.categorical_features", [])
    bins = get(cfg, "drift.psi_bins", 10)
    fail_at = get(cfg, "drift.psi_fail", 0.20)
    warn_at = get(cfg, "drift.psi_warn", 0.10)

    rows = []
    for f in numeric_feats:
        if f in reference.columns and f in batch.columns:
            psi = psi_numeric(reference[f].to_numpy(), batch[f].to_numpy(), bins=bins)
            rows.append({"feature": f, "type": "numeric", "psi": round(psi, 4),
                         "status": "FAIL" if psi > fail_at else "WARN" if psi > warn_at else "OK"})
    for f in cat_feats:
        if f in reference.columns and f in batch.columns:
            psi = psi_categorical(reference[f].tolist(), batch[f].tolist())
            rows.append({"feature": f, "type": "categorical", "psi": round(psi, 4),
                         "status": "FAIL" if psi > fail_at else "WARN" if psi > warn_at else "OK"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_gate(drift_table: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Evaluate the drift gate: FAIL if any feature PSI > psi_fail."""
    cfg = cfg or load_config()
    fail_at = get(cfg, "drift.psi_fail", 0.20)
    n_fail = int((drift_table["psi"] > fail_at).sum())
    n_warn = int(((drift_table["psi"] > get(cfg, "drift.psi_warn", 0.10)) & (drift_table["psi"] <= fail_at)).sum())
    return {
        "passed": n_fail == 0,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "fail_threshold": fail_at,
        "failing_features": drift_table.loc[drift_table["psi"] > fail_at, "feature"].tolist(),
        "warning_features": drift_table.loc[
            (drift_table["psi"] > get(cfg, "drift.psi_warn", 0.10)) & (drift_table["psi"] <= fail_at), "feature"
        ].tolist(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
