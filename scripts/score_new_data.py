"""Score a fresh batch of orders (new-data window) — production simulation.

This is the "tested on new data" step: the held-out last 15% of dates
is treated as a fresh production batch and pushed through the exact
path production would take:

    1. drift gate    PSI of every monitored feature vs training data.
                     FAIL -> exit 2 with drift report (no scoring).
    2. contract      required inputs, ranges, allowed categoricals;
                     repairable rows get fallbacks, others rejected.
    3. score         quantile ETA + promise time + late-risk.
    4. report        honest metrics vs the (delayed) ground truth +
                     alerts.csv for ops.

Outputs in reports/new_data_scoring/:
    predictions.parquet, alerts.csv, scoring_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import get, load_config  # noqa: E402
from src.constants import LATE_LABEL_COL, TARGET_COL  # noqa: E402
from src.evaluation.drift import compute_drift_table, drift_gate  # noqa: E402
from src.inference.model_contract import ModelContract, attach_predictions, summarize_batch  # noqa: E402
from src.training.train import pinball_loss  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("score_new_data")


def main() -> int:
    cfg = load_config()
    bundle_path = Path(get(cfg, "paths.models_dir")) / "champion" / "quantile_bundle.joblib"
    if not bundle_path.exists():
        log.error("Champion bundle missing (%s). Run scripts/run_pipeline.py first.", bundle_path)
        return 1
    bundle = joblib.load(bundle_path)

    processed_dir = Path(get(cfg, "paths.processed_dir"))
    train = pd.read_parquet(processed_dir / "train.parquet")
    new = pd.read_parquet(processed_dir / "new_data.parquet")
    log.info("Fresh batch: %d orders | training reference: %d orders", len(new), len(train))

    out_dir = Path(get(cfg, "paths.reports_dir")) / "new_data_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------ 1. drift gate
    log.info("[1/4] Drift gate — PSI vs training reference")
    drift_table = compute_drift_table(train, new, cfg)
    gate = drift_gate(drift_table, cfg)
    drift_table.to_csv(out_dir / "drift_table.csv", index=False)
    log.info(
        "Drift: %d OK / %d WARN / %d FAIL (threshold PSI > %.2f)",
        int((drift_table["status"] == "OK").sum()), gate["n_warn"], gate["n_fail"], gate["fail_threshold"],
    )
    if not gate["passed"]:
        log.error("DRIFT GATE FAILED for features %s — scoring blocked (see drift_table.csv).",
                  gate["failing_features"])
        report = {
            "run": "score_new_data",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "drift_gate": gate,
            "status": "BLOCKED_BY_DRIFT",
            "n_scored": 0,
        }
        with open(out_dir / "scoring_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return 2

    # ------------------------------------------------------ 2. contract
    log.info("[2/4] Model contract — validate & fallback")
    contract = ModelContract()
    # overwrite default fallbacks with train-fit global stats from the bundle
    clean, rejected = contract.validate_row_batch(new)
    log.info("Contract: %d OK-or-repaired / %d rejected", len(clean), len(rejected))
    if rejected is not None and len(rejected):
        rejected.to_parquet(out_dir / "rejected_rows.parquet", index=False)

    # ------------------------------------------------------ 3. score
    log.info("[3/4] Scoring batch (quantiles + late-risk + promise)")
    scored = attach_predictions(clean, bundle, contract, cfg)
    scored.to_parquet(out_dir / "predictions.parquet", index=False)

    # ------------------------------------------------------ 4. report vs actuals
    log.info("[4/4] Scoring report — honest metrics vs delayed ground truth")
    quantiles = bundle["config_snapshot"]["quantiles"]
    y = scored[TARGET_COL].to_numpy()

    per_q_pinball = {}
    for q in quantiles:
        pred = scored[f"p{int(q*100)}_min"].to_numpy()
        per_q_pinball[f"q{int(q*100)}"] = round(pinball_loss(y, pred, q), 3)
    p10 = scored[f"p{int(quantiles[0]*100)}_min"].to_numpy()
    p50 = scored[f"p{int(quantiles[1]*100)}_min"].to_numpy()
    p90 = scored[f"p{int(quantiles[2]*100)}_min"].to_numpy()
    coverage = float(((p10 <= y) & (y <= p90)).mean())
    mae = float(np.mean(np.abs(y - p50)))
    promise = scored["promise_min"].to_numpy()
    broken = float((y > promise).mean())

    late_auc = None
    late_prec = None
    if scored[LATE_LABEL_COL].nunique() == 2:
        late_auc = round(float(roc_auc_score(scored[LATE_LABEL_COL], scored["late_risk"])), 4)
        flagged = scored["late_flag"]
        late_prec = round(float(scored.loc[flagged, LATE_LABEL_COL].mean()), 4) if flagged.any() else 0.0

    drift_dict = drift_table.to_dict(orient="records")

    # full-batch accounting: contract rejections happen BEFORE scoring, so
    # summarize_batch (scored frame only) can't see them — merge explicitly
    scored_summary = summarize_batch(scored)
    scored_summary["n_rejected"] = int(len(rejected))
    scored_summary["n_batch"] = int(len(new))

    report = {
        "run": "score_new_data",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCORED",
        "n_batch": int(len(new)),
        "n_scored": int(len(scored)),
        "batch_summary": scored_summary,
        "drift_gate": {**gate, "table": drift_dict},
        "quantile_metrics_vs_actuals": {
            "pinball": per_q_pinball,
            "p10_p90_coverage": round(coverage, 4),
            "p50_mae_min": round(mae, 2),
        },
        "promise_time": {
            "avg_promise_min": round(float(promise.mean()), 2),
            "avg_actual_min": round(float(y.mean()), 2),
            "broken_promise_rate": round(broken, 4),
        },
        "late_risk_vs_actuals": {
            "test_auc": late_auc,
            "precision_at_alert_threshold": late_prec,
            "n_alerts": int(scored["late_flag"].sum()),
            "alert_threshold": round(float(bundle["alert_threshold"]), 3),
        },
    }
    with open(out_dir / "scoring_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ------------------------------------------------------ alerts csv
    max_alerts = get(cfg, "new_data.max_alerts_csv", 50)
    alerts = (
        scored[scored["late_flag"]]
        .sort_values("late_risk", ascending=False)
        .loc[:, [
            "ID", "order_date", "rider_id", "weather", "road_traffic_density",
            "haversine_km", "time_taken_min", "p50_min", "p90_min", "promise_min",
            "late_risk", "contract_status",
        ]]
        .head(max_alerts)
    )
    alerts.to_csv(out_dir / "alerts.csv", index=False)

    log.info("=" * 70)
    log.info("SCORING REPORT  | batch=%d | P50 MAE=%.1f min | coverage=%.1f%% | broken promises=%.1f%%",
             len(scored), mae, coverage * 100, broken * 100)
    if late_auc is not None:
        log.info("LATE-RISK       | AUC=%.3f | precision@threshold=%.3f | alerts=%d",
                 late_auc, late_prec or 0.0, report["late_risk_vs_actuals"]["n_alerts"])
    log.info("Artifacts -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
