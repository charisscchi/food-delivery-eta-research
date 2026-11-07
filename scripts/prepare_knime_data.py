"""
Prepare training and test CSV files for the
Food Delivery ETA KNIME beginner workflow.

Uses the original project's processed data splits.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "knime_demo"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Simple, interpretable features for the KNIME demo.
# Excludes pickup delay and rider history.
COLUMNS = [
    "haversine_km",
    "rider_age",
    "rider_rating",
    "order_hour",
    "weather",
    "road_traffic_density",
    "vehicle_condition",
    "multiple_deliveries",
    "time_taken_min",
]

for split in ["train", "test"]:
    source_file = SOURCE / f"{split}.parquet"

    if not source_file.exists():
        raise FileNotFoundError(
            f"Missing {source_file}. Run the original "
            "data preparation pipeline first."
        )

    data = pd.read_parquet(source_file)

    missing = [
        column for column in COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {split}: {missing}"
        )

    demo = data[COLUMNS].dropna().copy()

    output_file = OUTPUT / f"knime_{split}.csv"
    demo.to_csv(output_file, index=False)

    print(
        f"{split}: {len(demo):,} rows "
        f"saved to {output_file}"
    )

print("\nKNIME data preparation complete!")
