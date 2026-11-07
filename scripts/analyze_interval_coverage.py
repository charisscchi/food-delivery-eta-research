# Research Experiment 1
# Does weather or traffic affect delivery ETA reliability?

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

# 1. Load the existing test dataset and trained model
test = pd.read_parquet(
    ROOT / "data/processed/test.parquet"
)

bundle = joblib.load(
    ROOT / "models/champion/quantile_bundle.joblib"
)

# 2. Prepare the same inputs used by the original model
X = _matrix(test)
X = _align_columns(X, bundle["feature_columns"])
X = _clean_for_lgbm(X)

# 3. Generate delivery-time predictions
models = bundle["quantile_models"]

p10 = models[0.1].predict(X)
p50 = models[0.5].predict(X)
p90 = models[0.9].predict(X)

# 4. Measure prediction reliability for each order
results = test[
    ["weather", "road_traffic_density", "time_taken_min"]
].copy()

actual = results["time_taken_min"].to_numpy()

results["covered"] = (actual >= p10) & (actual <= p90)
results["absolute_error"] = np.abs(actual - p50)
results["interval_width"] = p90 - p10

# 5. Compare weather and traffic groups
tables = []

for feature in ["weather", "road_traffic_density"]:
    summary = (
        results.groupby(feature, dropna=False)
        .agg(
            orders=("covered", "size"),
            coverage=("covered", "mean"),
            mae_min=("absolute_error", "mean"),
            avg_width_min=("interval_width", "mean"),
        )
        .reset_index()
        .rename(columns={feature: "group"})
    )

    summary.insert(0, "category", feature)
    summary["coverage_pct"] = (
        summary.pop("coverage") * 100
    ).round(2)

    tables.append(summary)

final = pd.concat(tables, ignore_index=True)

# 6. Save research results
output = ROOT / "reports/metrics/weather_traffic_analysis.csv"
final.round(2).to_csv(output, index=False)

print("\nOverall coverage:", round(results["covered"].mean() * 100, 2), "%")
print("\nWeather and traffic comparison:")
print(final.round(2).to_string(index=False))
print("\nSaved results to:", output)
