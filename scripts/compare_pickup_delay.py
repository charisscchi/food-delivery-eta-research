# Research Experiment 4
# How much does pickup-delay information affect ETA predictions?

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import get, load_config
from src.training.train import (
    _matrix,
    _align_columns,
    _clean_for_lgbm,
    make_candidate,
)

# 1. Load the same data used in the original experiment
train = pd.read_parquet(
    ROOT / "data/processed/train.parquet"
)
test = pd.read_parquet(
    ROOT / "data/processed/test.parquet"
)

bundle = joblib.load(
    ROOT / "models/champion/quantile_bundle.joblib"
)

y_train = train["time_taken_min"].to_numpy()
y_test = test["time_taken_min"].to_numpy()

# 2. Build the original feature matrices
X_train = _matrix(train)
X_test = _align_columns(
    _matrix(test),
    bundle["feature_columns"]
)

X_train = _clean_for_lgbm(X_train)
X_test = _clean_for_lgbm(X_test)

# 3. Remove pickup delay from both train and test data
feature_to_remove = "pickup_delay_min"

assert feature_to_remove in X_train.columns

X_train_reduced = X_train.drop(columns=[feature_to_remove])
X_test_reduced = X_test.drop(columns=[feature_to_remove])

print("Original features:", X_train.shape[1])
print("Reduced features:", X_train_reduced.shape[1])

# 4. Predict using the existing original model
quantiles = [0.1, 0.5, 0.9]

original_predictions = {
    q: bundle["quantile_models"][q].predict(X_test)
    for q in quantiles
}

# 5. Train a new LightGBM model without pickup delay
cfg = load_config()

params = get(
    cfg,
    "training.candidates.lightgbm_quantile.params",
    {}
)

reduced_predictions = {}

for q in quantiles:
    print(f"Training reduced model: P{int(q * 100)}")

    model = make_candidate(
        "lightgbm_quantile", params, q
    )

    model.fit(X_train_reduced, y_train)

    reduced_predictions[q] = model.predict(X_test_reduced)

# 6. Evaluate both models on the SAME test observations
def evaluate(predictions, actual):
    lower = predictions[0.1]
    median = predictions[0.5]
    upper = predictions[0.9]

    return {
        "mae_min": round(
            float(np.mean(np.abs(actual - median))), 2
        ),
        "coverage_pct": round(
            float(np.mean(
                (actual >= lower) & (actual <= upper)
            ) * 100), 2
        ),
        "avg_width_min": round(
            float(np.mean(upper - lower)), 2
        ),
    }

original = evaluate(original_predictions, y_test)
reduced = evaluate(reduced_predictions, y_test)

results = pd.DataFrame([
    {"model": "Original", **original},
    {"model": "Without pickup delay", **reduced},
])

print("\nFeature ablation comparison:")
print(results.to_string(index=False))

# 7. Save the experiment results
output = (
    ROOT / "reports/metrics/pickup_delay_comparison.csv"
)

output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)

print("\nSaved to:", output)
