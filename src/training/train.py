"""Stage 4 — Training.

Quantile regression candidates race under 4-fold TimeSeriesSplit CV,
scored by pinball (quantile) loss — the honest metric for probabilistic
forecasts. Champion = lowest mean CV pinball. Every candidate + the
champion bundle is logged to MLflow (file store at ./mlruns), following
the experiment-tracking pattern used across Swiggy DS repos.

Also trains the late-risk classifier (is_late) with an alert-precision
threshold rule: smallest threshold achieving precision >= 0.80 on train
CV out-of-fold predictions (guarantee for ops alerts).

Output: models/champion/{quantile_bundle.joblib, late_risk_bundle.joblib, training_report.json}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import get, load_config
from src.constants import (
    CATEGORICAL_FEATURES,
    LATE_LABEL_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
)

log = logging.getLogger("training")


# ----------------------------------------------------------------- helpers
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss in minutes — lower is better."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix X (numeric + one-hot categoricals via pandas)."""
    X = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dummy_na=False,
    )
    return X


def _align_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex to the trained column set; unseen categoricals in new data vanish silently."""
    return X.reindex(columns=columns, fill_value=0)


def make_candidate(name: str, params: dict, quantile: float):
    """Instantiate a quantile regressor for one candidate + quantile.

    Objective-like keys in `params` (set in the constructor below) are
    dropped so config and code never fight over the same kwarg.
    """
    params = {k: v for k, v in params.items() if k not in ("objective", "alpha", "quantile_alpha")}
    if name == "hist_gbr_quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=42, **params)
    if name == "lightgbm_quantile":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None
        return LGBMRegressor(objective="quantile", alpha=quantile, random_state=42, **params)
    if name == "xgboost_quantile":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=quantile, random_state=42, **params)
    raise ValueError(f"Unknown candidate: {name}")


def _clean_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf and object cols that crash LightGBM/XGBoost native."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


# ------------------------------------------------------------ main training
def run(stage: str = "training", cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed = get(cfg, "project.random_seed", 42)
    quantiles = get(cfg, "training.quantiles", [0.10, 0.50, 0.90])
    n_folds = get(cfg, "training.cv.folds", 4)

    processed_dir = Path(get(cfg, "paths.processed_dir"))
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    log.info("Train rows: %d | Test rows: %d", len(train), len(test))

    X_train_full = _matrix(train)
    y_train = train[TARGET_COL].to_numpy()
    # test matrix must be aligned to the TRAIN dummified column set — a
    # separately-dummified frame has fewer categories (unseen-in-test cols
    # missing) and native boosters reject the shape mismatch
    X_test_full = _align_columns(_matrix(test), X_train_full.columns)
    y_test = test[TARGET_COL].to_numpy()
    feature_columns = X_train_full.columns.tolist()

    # ---------------------------------------------------- MLflow experiment
    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep run logs clean
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_dir = Path(get(cfg, "paths.mlflow_dir"))
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    # MLflow 3.x: file store is maintenance-mode; sqlite is the local standard
    db_path = (mlflow_dir / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("delivery-eta-quantile")
    client = MlflowClient()

    # one parent run for the whole training stage
    parent_run = mlflow.start_run(run_name="training-stage")
    parent_run_id = parent_run.info.run_id
    t0 = time.time()

    # TimeSeriesSplit on date-ordered train
    splitter = TimeSeriesSplit(n_splits=n_folds)
    date_order = np.argsort(train["order_date"].to_numpy())

    candidate_names = [
        name for name, spec in get(cfg, "training.candidates", {}).items() if spec.get("enabled", True)
    ]
    log.info("Candidates: %s | quantiles: %s | folds: %d", candidate_names, quantiles, n_folds)

    cv_results: dict[str, dict] = {}

    for cand_name in candidate_names:
        params = get(cfg, f"training.candidates.{cand_name}.params", {})

        # per-quantile CV
        cand_cv: dict[float, list[float]] = {q: [] for q in quantiles}
        unavailable = False
        for q in quantiles:
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(date_order.reshape(-1, 1))):
                model = make_candidate(cand_name, params, q)
                if model is None:
                    unavailable = True
                    break
                Xtr, Xva = X_train_full.iloc[tr_idx], X_train_full.iloc[va_idx]
                ytr, yva = y_train[tr_idx], y_train[va_idx]
                model.fit(_clean_for_lgbm(Xtr), ytr)
                pred = model.predict(_clean_for_lgbm(Xva))
                cand_cv[q].append(pinball_loss(yva, pred, q))
            if unavailable:
                break

        if unavailable:
            log.warning("Candidate %s unavailable on this runtime — skipped gracefully", cand_name)
            continue

        mean_pinball_per_q = {q: float(np.mean(scores)) for q, scores in cand_cv.items()}
        mean_pinball = float(np.mean(list(mean_pinball_per_q.values())))
        cv_results[cand_name] = {"mean_pinball": mean_pinball, "per_quantile": mean_pinball_per_q}

        # log candidate to MLflow
        with mlflow.start_run(run_name=f"candidate-{cand_name}", nested=True) as crun:
            mlflow.log_params({"candidate": cand_name, "quantiles": str(quantiles), "n_folds": n_folds, **_stringify(params)})
            mlflow.log_metrics({f"cv_pinball_q{int(q*100)}": v for q, v in mean_pinball_per_q.items()})
            mlflow.log_metric("cv_pinball_mean", mean_pinball)

        log.info("Candidate %s | CV mean pinball = %.3f min | per-q: %s",
                 cand_name, mean_pinball, {f"q{int(q*100)}": round(v, 3) for q, v in mean_pinball_per_q.items()})

    if not cv_results:
        raise RuntimeError("No quantile candidates ran — check candidate config / library availability.")

    champion_name = min(cv_results, key=lambda k: cv_results[k]["mean_pinball"])
    log.info("CHAMPION: %s (CV mean pinball %.3f min)", champion_name, cv_results[champion_name]["mean_pinball"])

    # ------------------------------------------------ champion: fit + test
    champion_params = get(cfg, f"training.candidates.{champion_name}.params", {})
    champion_models: dict[float, object] = {}
    test_pinball: dict[float, float] = {}
    p10_p90_coverage = None

    X_train_c = _clean_for_lgbm(X_train_full)
    X_test_c = _clean_for_lgbm(X_test_full)

    for q in quantiles:
        m = make_candidate(champion_name, champion_params, q)
        m.fit(X_train_c, y_train)
        champion_models[q] = m
        test_pinball[q] = pinball_loss(y_test, m.predict(X_test_c), q)

    # 80% interval coverage: fraction of test rows where p10 <= y <= p90
    preds_test = pd.DataFrame({f"q{int(q*100)}": m.predict(X_test_c) for q, m in champion_models.items()})
    p10_p90_coverage = float(
        ((preds_test["q10"] <= y_test) & (y_test <= preds_test["q90"])).mean()
    )
    # monotonicity guard: p10 <= p50 <= p90 should hold for ~all rows
    mono_violation_share = float(
        ((preds_test["q10"] > preds_test["q50"]) | (preds_test["q50"] > preds_test["q90"])).mean()
    )

    # train pinball (overfitting gap)
    train_pinball = {q: pinball_loss(y_train, champion_models[q].predict(X_train_c), q) for q in quantiles}
    overfit_gap = {f"q{int(q*100)}": round(train_pinball[q] - test_pinball[q], 3) for q in quantiles}

    with mlflow.start_run(run_name=f"champion-{champion_name}", nested=True):
        mlflow.log_params({"candidate": champion_name, "champion": True, **_stringify(champion_params)})
        mlflow.log_metrics({f"test_pinball_q{int(q*100)}": v for q, v in test_pinball.items()})
        mlflow.log_metric("p10_p90_coverage", p10_p90_coverage)

    # ------------------------------------------------ late-risk classifier
    late_threshold_rule = get(cfg, "training.late_risk_classifier.threshold_rule", {})
    y_late = train[LATE_LABEL_COL].to_numpy()
    y_late_test = test[LATE_LABEL_COL].to_numpy()

    lgbm_params = get(cfg, "training.late_risk_classifier.lightgbm_binary.params", {})
    late_model = None
    try:
        from lightgbm import LGBMClassifier
        late_model = LGBMClassifier(random_state=seed, **lgbm_params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        late_model = HistGradientBoostingClassifier(random_state=seed)

    # out-of-fold predictions on train for an honest threshold choice
    oof = np.zeros(len(train))
    for tr_idx, va_idx in splitter.split(date_order.reshape(-1, 1)):
        from sklearn.base import clone
        fold_model = clone(late_model)
        fold_model.fit(_clean_for_lgbm(X_train_full.iloc[tr_idx]), y_late[tr_idx])
        oof[va_idx] = fold_model.predict_proba(_clean_for_lgbm(X_train_full.iloc[va_idx]))[:, 1]

    train_auc_oof = float(roc_auc_score(y_late, oof))
    threshold, threshold_metrics = choose_alert_threshold(
        oof, y_late,
        min_precision=get(late_threshold_rule, "value", 0.80),
        max_threshold=get(late_threshold_rule, "max_threshold", 0.95),
    )
    log.info("Late-risk: OOF AUC=%.3f | alert threshold=%.2f (precision=%.3f, recall=%.3f)",
             train_auc_oof, threshold, threshold_metrics["precision"], threshold_metrics["recall"])

    late_model.fit(_clean_for_lgbm(X_train_full), y_late)
    late_test_proba = late_model.predict_proba(_clean_for_lgbm(X_test_full))[:, 1]
    late_test_auc = float(roc_auc_score(y_late_test, late_test_proba))
    late_test_ap = float(average_precision_score(y_late_test, late_test_proba))

    with mlflow.start_run(run_name="late-risk-classifier", nested=True):
        mlflow.log_param("model", type(late_model).__name__)
        mlflow.log_metric("train_oof_auc", train_auc_oof)
        mlflow.log_metric("test_auc", late_test_auc)
        mlflow.log_metric("test_pr_auc", late_test_ap)
        mlflow.log_metric("alert_threshold", threshold)
        mlflow.log_metric("threshold_precision", threshold_metrics["precision"])
        mlflow.log_metric("threshold_recall", threshold_metrics["recall"])

    # ------------------------------------------------------------- persist
    bundle = {
        "champion": champion_name,
        "quantile_models": champion_models,
        "feature_columns": feature_columns,
        "candidate_cv": cv_results,
        "late_model": late_model,
        "alert_threshold": threshold,
        "late_threshold_rule": late_threshold_rule,
        "train_late_share": float(y_late.mean()),
        "config_snapshot": {
            "quantiles": quantiles,
            "cv_folds": n_folds,
            "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        },
    }
    models_dir = Path(get(cfg, "paths.models_dir")) / "champion"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "quantile_bundle.joblib")

    training_report = {
        "stage": "training",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidates": cv_results,
        "champion": champion_name,
        "champion_params": champion_params,
        "quantiles": quantiles,
        "cv_folds": n_folds,
        "test_pinball": {f"q{int(q*100)}": round(v, 3) for q, v in test_pinball.items()},
        "train_pinball": {f"q{int(q*100)}": round(v,  3) for q, v in train_pinball.items()},
        "overfit_gap_min": overfit_gap,
        "p10_p90_coverage": round(p10_p90_coverage, 4),
        "quantile_monotonicity_violations": round(mono_violation_share, 4),
        "late_risk": {
            "model": type(late_model).__name__,
            "train_oof_auc": round(train_auc_oof, 4),
            "test_auc": round(late_test_auc, 4),
            "test_pr_auc": round(late_test_ap, 4),
            "alert_threshold": round(threshold, 3),
            "threshold_precision_on_train_oof": round(threshold_metrics["precision"], 4),
            "threshold_recall_on_train_oof": round(threshold_metrics["recall"], 4),
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "mlflow_parent_run_id": parent_run_id,
    }
    with open(models_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    mlflow.end_run()
    log.info("Champion bundle saved to %s", models_dir)
    return training_report


def choose_alert_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    min_precision: float = 0.80,
    max_threshold: float = 0.95,
) -> tuple[float, dict]:
    """Smallest score threshold whose precision on the given scores >= min_precision.

    Ops guarantee: at most 1 in 5 alerts is a false alarm. If unreachable,
    returns max_threshold with its metrics (honest, reported).
    """
    best = None
    for t in np.linspace(0.01, max_threshold, 95):
        flagged = scores >= t
        n_flag = int(flagged.sum())
        if n_flag == 0:
            continue
        precision = float((labels[flagged] == 1).mean())
        if precision >= min_precision:
            recall = float((labels[flagged] == 1).sum() / max(int((labels == 1).sum()), 1))
            best = (float(t), n_flag, precision, recall)
            break
    if best is None:
        t = float(max_threshold)
        flagged = scores >= t
        n_flag = int(flagged.sum())
        precision = float((labels[flagged] == 1).mean()) if n_flag else 0.0
        recall = float((labels[flagged] == 1).sum() / max(int((labels == 1).sum()), 1))
        return t, {"threshold": t, "n_flagged": n_flag, "precision": precision, "recall": recall, "satisfied": False}
    t, n_flag, precision, recall = best
    return t, {"threshold": t, "n_flagged": n_flag, "precision": precision, "recall": recall, "satisfied": True}


def _stringify(d: dict) -> dict:
    return {k: (",".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in d.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
