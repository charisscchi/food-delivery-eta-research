# Food Delivery ETA & Late-Risk — Quantile ML Pipeline

**P10 / P50 / P90 delivery-time forecasts + a late-risk alerting model, engineered the way production DS teams build it: config-driven stages, gate-based data validation, leak-safe feature engineering, candidate racing under time-series CV, MLflow experiment tracking, a model contract with deterministic fallbacks, and drift-gated scoring on a held-out "fresh production batch".**

- **Problem**: "How long will this order take?" is not a point estimate — customers need a promise (P90), ops need risk flags, and riders get judged on the median. This project predicts the **whole distribution** of delivery time.
- **Data**: 45,593 real food-delivery orders (rider attributes, restaurant→drop geo, weather, traffic, vehicle, festival, city) — open Kaggle-origin dataset, see [Data card](#data-card).
- **Runs 100% locally** in VSCode — no cloud, no GPU, no API keys.

---

## Why this project (the Swiggy angle)

This mirrors how Swiggy's DS org actually ships models (structure borrowed from their `ds_time_prediction`-style repos):

| Swiggy DS practice | Where it lives here |
|---|---|
| Config-driven stages (`config.yaml` is the single source of truth) | `configs/config.yaml` |
| Data-quality **gates** (FAIL stops the pipeline, WARN reports) | `src/validation/validate.py` |
| Time-based splits + target-encoded rider history **fit on train only** (no leakage) | `src/features/build_features.py` |
| Candidate racing: 3 engines × 3 quantiles, champion by CV pinball loss | `src/training/train.py` |
| MLflow experiment tracking (Swiggy literally maintains an MLflow fork) | `mlruns/mlflow.db` |
| **Model contract**: required inputs, valid ranges, fallback rules — DS owns the contract | `src/inference/model_contract.py` |
| PSI drift gate before scoring a fresh batch | `src/evaluation/drift.py`, `scripts/score_new_data.py` |
| Batch scoring + ops alerts (like their daily `run_daily.py` pattern) | `scripts/score_new_data.py` → `reports/new_data_scoring/alerts.csv` |

## Architecture

```
data/raw/train.csv
      │
      ▼
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
│ 1. INGEST       │──▶│ 2. VALIDATE     │──▶│ 3. FEATURES              │
│ parse & type,   │   │ 9 quality gates │   │ geo (haversine), temporal│
│ strip prefixes, │   │ FAIL/WARN       │   │ rider history (train-only│
│ NaN sentinels   │   │                 │   │ target encoding), splits │
└─────────────────┘   └──────────────────┘   │ 70/15/15 by Order_Date   │
                                             └───────────┬──────────────┘
                                                         ▼
┌──────────────────────────────┐   ┌───────────────────────────────────┐
│ 5. EVALUATE                  │◀──│ 4. TRAIN                          │
│ pinball (train vs test),     │   │ 3 candidates × 3 quantiles        │
│ 80% interval coverage,       │   │ 4-fold TimeSeriesSplit CV         │
│ calibration, per-slice AUC,  │   │ champion = lowest CV pinball       │
│ promise-time quality, figs   │   │ + late-risk classifier (P80 alert │
└──────────────┬───────────────┘   │ precision rule) — all in MLflow   │
               │                   └───────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6. SCORE NEW DATA (production simulation)                          │
│ PSI drift gate → model contract (fallbacks/rejects) → quantile     │
│ scores + promise + late-risk → honest metrics vs delayed ground     │
│ truth → ops alerts csv                                            │
└────────────────────────────────────────────────────────────────────┘
```

## Quickstart (VSCode, local)

```powershell
# 1. create the environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. fetch the raw data (idempotent) — or place Kaggle's train.csv at data/raw/
python scripts/fetch_data.py

# 3. run the full pipeline (ingest → validate → features → train → evaluate)
python scripts/run_pipeline.py
#    partial: python scripts/run_pipeline.py --stages ingest,validate

# 4. score the held-out "fresh production batch" (drift gate + contract + alerts)
python scripts/score_new_data.py

# 5. tests
python -m pytest tests/ -q

# 6. explore
#    notebooks/01_eda.ipynb  (run after stages ingest,validate)
#    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Results

All numbers below are from the committed run artifacts in `reports/` (champion: **lightgbm_quantile**, selected by 4-fold time-series CV over HistGBR and XGBoost — full candidate race in `models/champion/training_report.json`, every run logged to MLflow).

### Quantile head

| Metric | Test | New-data batch |
|---|---|---|
| Pinball loss P10 / P50 / P90 (min) | 0.61 / 1.59 / 0.67 | 0.60 / 1.58 / 0.65 |
| P50 MAE (minutes) | **3.18** | **3.15** |
| P10–P90 interval coverage (nominal 80%) | 70.4% | 70.8% |
| Quantile crossing violations | 0.53% | — |
| Broken-promise rate (actual > padded P90 promise) | **1.7%** | **1.6%** |
| Avg promise vs avg actual (padding) | 33.4 vs 26.5 min (+6.9) | 33.3 vs 26.4 min |

### Late-risk classifier (is_late := time > train-P75 = 32 min)

| Metric | Train (OOF) | Test | New-data batch |
|---|---|---|---|
| ROC-AUC | 0.801 | 0.974 | 0.971 |
| Precision @ alert threshold | 0.802 | 0.799 | 0.779 |
| Recall @ alert threshold | 0.698 | 0.901 | — |

Alert threshold 0.30, chosen on train out-of-fold predictions as the smallest threshold holding precision ≥ 0.80 (the ops guarantee: ≤ 1 in 5 alerts is a false alarm).

### Candidate race (CV mean pinball, minutes — lower wins)

| Candidate | P10 | P50 | P90 | Mean |
|---|---|---|---|---|
| lightgbm_quantile **(champion)** | 0.658 | 1.605 | 0.714 | **0.992** |
| xgboost_quantile | 0.652 | 1.621 | 0.718 | 0.997 |
| hist_gbr_quantile | 0.664 | 1.635 | 0.719 | 1.006 |

### Drift gate on the new-data batch

All 14 monitored features stable — max PSI 0.011 (`rider_hist_median_min`), gate threshold 0.20. Full table in `reports/new_data_scoring/drift_table.csv`.

### What the model learned (top features)

`haversine_km` > `rider_age` > `rider_hist_median_min` (train-only target encoding) > `rider_rating` > `order_hour` — geo and rider history dominate, exactly what a delivery-time model should lean on. Per-slice AUCs are stable across weather/traffic/city (0.93–0.99); the weakest slice is `weather=unknown` (AUC 0.74, ~1% of orders) — see `reports/metrics/model_metrics.json`.

### Calibration (deciles, test set)

Bottom-4 deciles: 0% actual late rate. Decile 8: 52% predicted vs 52% actual. Decile 10: 99% predicted vs 99% actual. Monotone and tight — the late-risk score is directly usable as an ops priority queue.

## Repo structure

```
delivery-eta-quantile/
├── configs/config.yaml          # every knob: gates, candidates, splits, drift, contract
├── data/                        # raw/ interim/ processed/ (gitignored, regenerated)
├── models/champion/             # quantile_bundle.joblib + training_report.json (gitignored)
├── mlruns/                      # MLflow sqlite store (gitignored)
├── notebooks/01_eda.ipynb       # structural EDA on the interim parquet
├── reports/                     # COMMITTED: quality report, metrics, figures, alerts
│   ├── data_quality/ingestion_quality_report.json
│   ├── metrics/model_metrics.json
│   ├── figures/*.png
│   ├── new_data_scoring/{predictions.parquet, alerts.csv, scoring_report.json, drift_table.csv}
│   └── run_summary.json
├── scripts/
│   ├── fetch_data.py            # idempotent downloader
│   ├── run_pipeline.py          # stage orchestrator (--stages subset)
│   └── score_new_data.py        # production simulation: drift → contract → score → alerts
├── src/
│   ├── config.py, constants.py
│   ├── ingestion/ingest.py      # prefix strip, NaN sentinel, typing
│   ├── validation/validate.py   # 9 gates, FAIL/WARN severities
│   ├── features/build_features.py  # geo/temporal/rider features + time splits
│   ├── training/train.py        # candidate race + champion + late-risk + MLflow
│   ├── evaluation/evaluate.py    # pinball, coverage, calibration, slices, figures
│   ├── evaluation/drift.py      # PSI numeric + categorical + gate
│   └── inference/model_contract.py  # contract + single scoring path
└── tests/                       # 18 tests: ingest, features/leakage, drift, contract, threshold
```

## Model card

- **Inputs** (contract-enforced): rider id/age/rating, restaurant & drop lat/lon, order date + times, weather, traffic density, vehicle condition/type, order type, multiple deliveries, festival, city.
- **Outputs**: `p10_min`, `p50_min`, `p90_min` (monotonicity-repaired), `promise_min` (padded, bucketed P90), `late_risk` (P(time > train-P75)), `late_flag`.
- **Fallbacks**: out-of-range numerics → in-range median + FALLBACK status; unknown categoricals → `'unknown'`; missing critical fields (coords, rider id, date) → REJECTED row, never a crash.
- **Retrain trigger**: PSI drift gate on new batches (any feature > 0.20 blocks scoring and surfaces the report).
- **Known limits**: single-city-family dataset; no live traffic feed; rider history is a median (not EWMA); promise bucketing is a business knob.

## Data card

- **Source**: Kaggle "Food Delivery Time Prediction" train set (45,593 orders × 20 cols), fetched from a public GitHub mirror listed in `configs/config.yaml`.
- **License**: dataset used for learning/portfolio purposes; code is MIT (see LICENSE).

## What I'd build next

1. Hyperparameter search per quantile (optuna) instead of fixed candidate grids.
2. Calibration of P90 via conformal prediction (coverage guarantee without distribution assumptions).
3. Rider-history as EWMA with recency decay, and a "new rider" segment feature.
4. SHAP-based per-slice error explanation for the ops report.
