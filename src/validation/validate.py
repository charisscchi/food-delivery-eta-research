"""Stage 2 — Data validation.

Gate-based quality checks on the interim dataset, in the spirit of the
data-quality gates used on Swiggy's DS pipelines: every gate has a
severity (FAIL stops the pipeline, WARN is logged for the run report)
and the full result set is written to reports/data_quality/.

Gates:
  row_count_min       FAIL   enough data to train on
  column_count        FAIL   schema did not silently change
  id_uniqueness       FAIL   no duplicate order ids
  target_parse_rate   FAIL   % rows with parseable Time_taken
  target_range        FAIL   delivery minutes within sane bounds
  date_parse_rate     FAIL   % rows with parseable Order_Date
  null_share_numeric  WARN   per-column null share
  duplicate_rows      WARN   exact row dupes (deduped downstream)
  coordinate_sanity   WARN   % rows with (0,0) / missing coords
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get, load_config
from src.constants import ID_COL, INTERIM_COLUMNS, TARGET_COL

log = logging.getLogger("validation")

FAIL = "FAIL"
WARN = "WARN"
PASS = "PASS"


def _gate(name: str, severity: str, passed: bool, observed: object, expected: object = None) -> dict:
    return {
        "gate": name,
        "severity": severity,
        "status": PASS if passed else severity,
        "observed": observed,
        "expected": expected,
    }


def run_checks(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """Run all validation gates; returns the gate-result list (never raises)."""
    cfg = cfg or load_config()
    gates_cfg = {g["gate"]: g for g in get(cfg, "validation.gates", [])}
    results: list[dict] = []

    # ---- row count ------------------------------------------------------
    g = gates_cfg.get("row_count_min", {})
    min_rows = g.get("value", 40000)
    results.append(_gate("row_count_min", g.get("severity", FAIL), len(df) >= min_rows, len(df), min_rows))

    # ---- column count ---------------------------------------------------
    g = gates_cfg.get("column_count", {})
    expected_cols = g.get("value", len(INTERIM_COLUMNS))
    results.append(_gate("column_count", g.get("severity", FAIL), df.shape[1] == expected_cols, df.shape[1], expected_cols))

    # ---- id uniqueness --------------------------------------------------
    n_dupes = int(df[ID_COL].duplicated().sum())
    g = gates_cfg.get("id_uniqueness", {})
    results.append(_gate("id_uniqueness", g.get("severity", FAIL), n_dupes == 0, n_dupes, 0))

    # ---- target parse rate + range -------------------------------------
    g = gates_cfg.get("target_parse_rate", {})
    parse_rate = float(df[TARGET_COL].notna().mean())
    results.append(_gate("target_parse_rate", g.get("severity", FAIL), parse_rate >= g.get("value", 0.995), round(parse_rate, 6), g.get("value", 0.995)))

    g = gates_cfg.get("target_range", {})
    lo, hi = g.get("value", [5, 120])
    in_range = df[TARGET_COL].dropna().between(lo, hi)
    # we tolerate a small % outside range (clipped in features), but >5% is a schema shift
    share_out = float(1 - in_range.mean()) if len(in_range) else 1.0
    results.append(_gate("target_range", g.get("severity", FAIL), share_out <= 0.05, round(share_out, 6), "<=0.05 outside"))

    # ---- date parse rate ------------------------------------------------
    g = gates_cfg.get("date_parse_rate", {})
    date_rate = float(df["order_date"].notna().mean())
    results.append(_gate("date_parse_rate", g.get("severity", FAIL), date_rate >= g.get("value", 0.999), round(date_rate, 6), g.get("value", 0.999)))

    # ---- null share per numeric column (WARN) ---------------------------
    g = gates_cfg.get("null_share_numeric", {})
    null_threshold = g.get("value", 0.30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nulls = {c: round(float(df[c].isna().mean()), 4) for c in numeric_cols}
    worst_col, worst_share = max(nulls.items(), key=lambda kv: kv[1]) if nulls else (None, 0.0)
    results.append(_gate("null_share_numeric", g.get("severity", WARN), worst_share <= null_threshold, {"worst_col": worst_col, "max_share": worst_share, "per_col": nulls}, f"<= {null_threshold}"))

    # ---- exact duplicate rows (WARN) -------------------------------------
    g = gates_cfg.get("duplicate_rows", {})
    n_exact = int(df.duplicated().sum())
    results.append(_gate("duplicate_rows", g.get("severity", WARN), n_exact <= g.get("value", 1000), n_exact, g.get("value", 1000)))

    # ---- coordinate sanity (WARN) ----------------------------------------
    g = gates_cfg.get("coordinate_sanity", {})
    coord_cols = ["rest_lat", "rest_lon", "dest_lat", "dest_lon"]
    bad_coord = (
        df[coord_cols].isna().any(axis=1)
        | ((df["rest_lat"] == 0) & (df["rest_lon"] == 0))
        | ((df["dest_lat"] == 0) & (df["dest_lon"] == 0))
    )
    bad_share = float(bad_coord.mean())
    results.append(_gate("coordinate_sanity", g.get("severity", WARN), bad_share <= g.get("value", 0.10), round(bad_share, 6), g.get("value", 0.10)))

    return results


def run(stage: str = "validation", cfg: dict | None = None) -> dict:
    """Validation entry point: interim parquet -> gate report json + hard stop on FAIL."""
    cfg = cfg or load_config()
    interim_path = Path(get(cfg, "paths.interim_data"))
    df = pd.read_parquet(interim_path)

    results = run_checks(df, cfg)

    report = {
        "stage": "validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "gates": results,
        "n_fail": sum(1 for r in results if r["status"] == FAIL),
        "n_warn": sum(1 for r in results if r["status"] == WARN),
    }

    report_dir = Path(get(cfg, "paths.reports_dir")) / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "ingestion_quality_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    for r in results:
        log.info("[%s] %-22s observed=%s expected=%s", r["status"], r["gate"], r["observed"] if not isinstance(r["observed"], dict) else "see report", r["expected"])

    if report["n_fail"] > 0:
        failed = [r["gate"] for r in results if r["status"] == FAIL]
        raise DataQualityError(f"Validation FAILED {report['n_fail']} gate(s): {failed}. Report: {out}")

    log.info("Validation passed: %d PASS, %d WARN -> %s", sum(1 for r in results if r["status"] == PASS), report["n_warn"], out)
    return report


class DataQualityError(RuntimeError):
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
