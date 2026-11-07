# Research Experiment 2
# Can conformal calibration improve food delivery ETA intervals?

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.training.train import (
    _matrix,
    _align_columns,
    _clean_for_lgbm,
)

# Load the original trained model
bundle = joblib.load(
    ROOT / "models/champion/quantile_bundle.joblib"
)

models = bundle["quantile_models"]

# The original test set becomes our calibration set.
calibration = pd.read_parquet(
    ROOT / "data/processed/test.parquet"
)

# Reserve the later data for final evaluation.
final_test = pd.read_parquet(
    ROOT / "data/processed/new_data.parquet"
)


def predict_interval(data):
    X = _matrix(data)
    X = _align_columns(X, bundle["feature_columns"])
    X = _clean_for_lgbm(X)

    lower = models[0.1].predict(X)
    upper = models[0.9].predict(X)

    return lower, upper


# Step 1: Predictions on calibration data
cal_lower, cal_upper = predict_interval(calibration)
cal_actual = calibration["time_taken_min"].to_numpy()

# Step 2: Calculate conformal adjustment
scores = np.maximum(
    cal_lower - cal_actual,
    cal_actual - cal_upper
)

alpha = 0.20
n = len(scores)

level = np.ceil((n + 1) * (1 - alpha)) / n
correction = np.quantile(scores, level, method="higher")

print(f"Calibration orders: {n}")
print(f"Conformal adjustment: {correction:.2f} minutes")

# Step 3: Predictions on untouched final data
lower, upper = predict_interval(final_test)
actual = final_test["time_taken_min"].to_numpy()

# Original intervals
baseline_lower = lower
baseline_upper = upper

# Calibrated intervals
adjusted_lower = lower - correction
adjusted_upper = upper + correction


def evaluate(y, lo, hi):
    coverage = np.mean((y >= lo) & (y <= hi)) * 100
    width = np.mean(hi - lo)

    return round(coverage, 2), round(width, 2)


# Step 4: Compare results overall and by group
rows = []

groups = [
    ("overall", "All", np.ones(len(final_test), dtype=bool))
]

for column in ["weather", "road_traffic_density"]:
    for name, positions in final_test.groupby(
        column, dropna=False
    ).indices.items():

        mask = np.zeros(len(final_test), dtype=bool)
        mask[positions] = True
        groups.append((column, str(name), mask))

for category, name, mask in groups:
    base_cov, base_width = evaluate(
        actual[mask],
        baseline_lower[mask],
        baseline_upper[mask]
    )

    new_cov, new_width = evaluate(
        actual[mask],
        adjusted_lower[mask],
        adjusted_upper[mask]
    )

    rows.append({
        "category": category,
        "group": name,
        "orders": int(mask.sum()),
        "baseline_coverage_pct": base_cov,
        "calibrated_coverage_pct": new_cov,
        "baseline_width_min": base_width,
        "calibrated_width_min": new_width,
    })

results = pd.DataFrame(rows)

print("\nCalibration experiment results:")
print(results.to_string(index=False))

# Step 5: Save results
output = (
    ROOT / "reports/metrics/calibration_comparison.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output, index=False)
print("\nSaved to:", output)
