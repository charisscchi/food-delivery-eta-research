"""Stage 5 — Evaluation (test set).

Beyond point metrics, this is the eval suite a Swiggy DS reviewer
would look for:

  * pinball loss per quantile (train vs test — overfitting gap)
  * P10–P90 interval coverage (should be ~80%)
  * quantile crossing violations (p10<=p50<=p90)
  * decile calibration of late-risk scores vs actual late rate
  * per-slice AUC (weather / traffic / city) — where the model is weak
  * promise-time recommendation quality: share of orders where actual
    time exceeded the padded P90 promise (broken-promise rate)
  * feature importance (gain-based where available)

Writes reports/metrics/model_metrics.json + reports/figures/*.png.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: VSCode terminal / CI safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import get, load_config
from src.constants import LATE_LABEL_COL, TARGET_COL
from src.training.train import _align_columns, _clean_for_lgbm, _matrix, pinball_loss

log = logging.getLogger("evaluation")


def _load_bundle(cfg: dict) -> dict:
    bundle_path = Path(get(cfg, "paths.models_dir")) / "champion" / "quantile_bundle.joblib"
    return joblib.load(bundle_path)


def _feature_matrix(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = _matrix(df)
    return _align_columns(X, bundle["feature_columns"])


def decile_calibration(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Mean score vs actual late-rate per score decile — the calibration table."""
    df = pd.DataFrame({"score": scores, "label": labels})
    df["decile"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop") + 1
    g = df.groupby("decile").agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        actual_late_rate=("label", "mean"),
    ).reset_index()
    return g


def feature_importance(bundle: dict) -> pd.DataFrame | None:
    """Gain-based importance from the champion's q50 model if available."""
    model = bundle["quantile_models"].get(0.50)
    if model is None:
        return None
    try:
        imp = model.feature_importances_  # LightGBM / XGBoost
        return (
            pd.DataFrame({"feature": bundle["feature_columns"], "importance": imp})
            .sort_values("importance", ascending=False)
            .head(15)
        )
    except AttributeError:
        return None


def run(stage: str = "evaluation", cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    bundle = _load_bundle(cfg)

    processed_dir = Path(get(cfg, "paths.processed_dir"))
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    X_test = _feature_matrix(test, bundle)
    y_test = test[TARGET_COL].to_numpy()
    X_train = _feature_matrix(train, bundle)
    y_train = train[TARGET_COL].to_numpy()

    quantiles = bundle["config_snapshot"]["quantiles"]
    alert_threshold = bundle["alert_threshold"]

    # -------------------------------------------------- quantile metrics
    test_preds = {q: bundle["quantile_models"][q].predict(_clean_for_lgbm(X_test)) for q in quantiles}
    train_preds = {q: bundle["quantile_models"][q].predict(_clean_for_lgbm(X_train)) for q in quantiles}

    pinball_test = {f"q{int(q*100)}": round(pinball_loss(y_test, test_preds[q], q), 3) for q in quantiles}
    pinball_train = {f"q{int(q*100)}": round(pinball_loss(y_train, train_preds[q], q), 3) for q in quantiles}
    overfit_gap = {k: round(pinball_train[k] - pinball_test[k], 3) for k in pinball_test}

    p10, p50, p90 = test_preds[quantiles[0]], test_preds[quantiles[1]], test_preds[quantiles[2]]
    coverage_80 = float(((p10 <= y_test) & (y_test <= p90)).mean())
    crossing = float(((p10 > p50) | (p50 > p90)).mean())
    mae_p50 = float(np.mean(np.abs(y_test - p50)))

    # promise time: padded P90 rounded up to bucket, e.g. 41 -> 45
    bucket = get(cfg, "evaluation.promise_time.bucket_minutes", 5)
    promise = np.ceil(p90 / bucket) * bucket
    broken_promise_rate = float((y_test > promise).mean())
    avg_promise = float(promise.mean())
    avg_actual = float(y_test.mean())

    # -------------------------------------------------- late-risk metrics
    late_model = bundle["late_model"]
    late_scores_test = late_model.predict_proba(_clean_for_lgbm(X_test))[:, 1]
    late_auc = float(roc_auc_score(test[LATE_LABEL_COL], late_scores_test))
    flagged = late_scores_test >= alert_threshold
    flag_precision = float(test.loc[flagged, LATE_LABEL_COL].mean()) if flagged.any() else 0.0
    flag_recall = float((test.loc[flagged, LATE_LABEL_COL].sum()) / max(test[LATE_LABEL_COL].sum(), 1))

    calib = decile_calibration(late_scores_test, test[LATE_LABEL_COL].to_numpy())

    # per-slice AUC on weather / traffic / city
    slice_aucs: dict[str, dict[str, float]] = {}
    for col in ["weather", "road_traffic_density", "city"]:
        aucs = {}
        for val, idx in test.groupby(col).indices.items():
            labels_slice = test[LATE_LABEL_COL].to_numpy()[idx]
            if len(np.unique(labels_slice)) < 2:
                continue
            aucs[str(val)] = round(float(roc_auc_score(labels_slice, late_scores_test[idx])), 3)
        slice_aucs[col] = aucs

    imp = feature_importance(bundle)

    # -------------------------------------------------- figures
    fig_dir = Path(get(cfg, "paths.reports_dir")) / get(cfg, "evaluation.figures_dir", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. calibration plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(calib["mean_score"], calib["actual_late_rate"], marker="o", label="Actual late rate")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted late-risk score (decile)")
    ax.set_ylabel("Actual late rate")
    ax.set_title("Late-risk calibration — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "late_risk_calibration.png", dpi=150)
    plt.close(fig)

    # 2. quantile fan chart on a time-ordered sample of 150 orders
    sample_idx = np.linspace(0, len(y_test) - 1, 150).astype(int)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(
        range(len(sample_idx)), p10[sample_idx], p90[sample_idx],
        alpha=0.25, color="tab:blue", label="P10–P90 interval",
    )
    ax.plot(range(len(sample_idx)), p50[sample_idx], color="tab:blue", lw=1.2, label="P50 prediction")
    ax.scatter(range(len(sample_idx)), y_test[sample_idx], s=6, color="tab:red", alpha=0.55, label="Actual")
    ax.set_xlabel("Test orders (time-ordered sample)")
    ax.set_ylabel("Delivery time (min)")
    ax.set_title(f"Quantile predictions vs actuals — test set (80% interval coverage: {coverage_80:.0%})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "quantile_fan_chart.png", dpi=150)
    plt.close(fig)

    # 3. feature importance
    if imp is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="tab:blue")
        ax.set_xlabel("Importance (gain)")
        ax.set_title("Top features — champion q50 model")
        fig.tight_layout()
        fig.savefig(fig_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    # 4. error distribution of P50
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_test - p50, bins=60, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Actual - P50 (min)")
    ax.set_ylabel("Orders")
    ax.set_title(f"P50 error distribution — test MAE {mae_p50:.1f} min")
    fig.tight_layout()
    fig.savefig(fig_dir / "p50_error_dist.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------- metrics report
    metrics = {
        "stage": "evaluation",
        "champion": bundle["champion"],
        "quantiles": quantiles,
        "pinball_test": pinball_test,
        "pinball_train": pinball_train,
        "overfit_gap_min": overfit_gap,
        "pinball_gap_threshold": get(cfg, "evaluation.overfitting.pinball_gap_warn", 0.30),
        "p10_p90_coverage": round(coverage_80, 4),
        "quantile_crossing_share": round(crossing, 4),
        "p50_mae_min": round(mae_p50, 2),
        "promise_time": {
            "bucket_minutes": bucket,
            "avg_promise_min": round(avg_promise, 2),
            "avg_actual_min": round(avg_actual, 2),
            "broken_promise_rate": round(broken_promise_rate, 4),
            "padding_min": round(avg_promise - avg_actual, 2),
        },
        "late_risk": {
            "alert_threshold": round(alert_threshold, 3),
            "test_auc": round(late_auc, 4),
            "test_precision_at_threshold": round(flag_precision, 4),
            "test_recall_at_threshold": round(flag_recall, 4),
            "n_flagged_test": int(flagged.sum()),
        },
        "decile_calibration": calib.to_dict(orient="records"),
        "slice_aucs": slice_aucs,
        "feature_importance_top": imp.to_dict(orient="records") if imp is not None else None,
        "overfitting_flags": {
            "pinball_gap_exceeds": any(
                v > get(cfg, "evaluation.overfitting.pinball_gap_warn", 0.30) for v in overfit_gap.values()
            ),
            "auc_gap_exceeds": None,  # filled below
        },
    }

    # late AUC overfitting check
    from sklearn.base import clone as _clone
    train_late_scores = late_model.predict_proba(_clean_for_lgbm(X_train))[:, 1]
    late_train_auc = float(roc_auc_score(train[LATE_LABEL_COL], train_late_scores))
    metrics["overfitting_flags"]["auc_gap_exceeds"] = bool(
        late_train_auc - late_auc > get(cfg, "evaluation.overfitting.auc_gap_warn", 0.05)
    )
    metrics["late_risk"]["train_auc"] = round(late_train_auc, 4)

    metrics_dir = Path(get(cfg, "paths.reports_dir")) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Evaluation complete: champion=%s | P50 MAE=%.1f min | coverage=%.1f%% | late AUC=%.3f",
             bundle["champion"], mae_p50, coverage_80 * 100, late_auc)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
