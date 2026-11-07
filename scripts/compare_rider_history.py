# Research Experiment 5
# Evaluate model performance without two potentially
# problematic features.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.training.train import (
    _matrix,
    _align_columns,
    _clean_for_lgbm,
    make_candidate,
)

# Load original train/test data and feature definitions
train = pd.read_parquet(
    ROOT / "data/processed/train.parquet"
)
test = pd.read_parquet(
    ROOT / "data/processed/test.parquet"
)

bundle = joblib.load(
    ROOT / "models/champion/quantile_bundle.joblib"
)

columns = bundle["feature_columns"]

X_train = _align_columns(_matrix(train), columns)
X_test = _align_columns(_matrix(test), columns)

# Remove the two features we want to investigate
excluded = [
    "pickup_delay_min",
    "rider_hist_median_min",
]

X_train = X_train.drop(columns=excluded)
X_test = X_test.drop(columns=excluded)

X_train = _clean_for_lgbm(X_train)
X_test = _clean_for_lgbm(X_test)

y_train = train["time_taken_min"].to_numpy()
y_test = test["time_taken_min"].to_numpy()

print("Features remaining:", len(X_train.columns))

# Use the same model type and parameters
cfg = load_config()

params = cfg["training"]["candidates"][
    "lightgbm_quantile"
]["params"]

predictions = {}

for q in [0.1, 0.5, 0.9]:
    print(f"Training P{int(q * 100)} model...")

    model = make_candidate(
        "lightgbm_quantile", params, q
    )
    model.fit(X_train, y_train)

    predictions[q] = model.predict(X_test)

# Evaluate performance
lower = predictions[0.1]
median = predictions[0.5]
upper = predictions[0.9]

mae = np.mean(np.abs(y_test - median))
coverage = np.mean(
    (y_test >= lower) & (y_test <= upper)
) * 100
width = np.mean(upper - lower)

result = pd.DataFrame([{
    "model": "Without pickup delay and rider history",
    "mae_min": round(mae, 2),
    "coverage_pct": round(coverage, 2),
    "avg_width_min": round(width, 2),
}])

print("\nNew model results:")
print(result.to_string(index=False))

# Combine with the previous ablation results
previous = pd.read_csv(
    ROOT / "reports/metrics/pickup_delay_comparison.csv"
)

comparison = pd.concat(
    [previous, result],
    ignore_index=True
)

print("\nAll three models:")
print(comparison.to_string(index=False))

output = ROOT / "reports/metrics/feature_ablation_results.csv"
comparison.to_csv(output, index=False)

print("\nSaved:", output)
