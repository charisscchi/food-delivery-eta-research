# Research Experiment 3
# Visualize delivery ETA calibration results

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

data = pd.read_csv(
    ROOT / "reports/metrics/calibration_comparison.csv"
)

output_dir = ROOT / "reports/figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Overall coverage comparison
overall = data[data["category"] == "overall"].iloc[0]

labels = ["Original model", "Calibrated model"]
coverage = [
    overall["baseline_coverage_pct"],
    overall["calibrated_coverage_pct"],
]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, coverage, width=0.5)

ax.axhline(
    80, color="black", linestyle="--",
    label="Target: 80%"
)

for bar, value in zip(bars, coverage):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

ax.set_ylim(0, 100)
ax.set_ylabel("Prediction interval coverage (%)")
ax.set_title("Does Calibration Improve Delivery ETA Reliability?")
ax.legend()
fig.tight_layout()

fig.savefig(
    output_dir / "eta_coverage_comparison.png",
    dpi=200
)
plt.close(fig)

# Figure 2: Compare weather and traffic groups
groups = data[
    (data["category"] != "overall")
    & (data["group"] != "unknown")
].copy()

groups["label"] = groups.apply(
    lambda row: (
        ("Weather: " if row["category"] == "weather"
         else "Traffic: ") + row["group"]
    ),
    axis=1
)

y = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    y - 0.2,
    groups["baseline_coverage_pct"],
    height=0.4,
    label="Original"
)

ax.barh(
    y + 0.2,
    groups["calibrated_coverage_pct"],
    height=0.4,
    label="Calibrated"
)

ax.axvline(
    80, color="black", linestyle="--",
    label="80% target"
)

ax.set_yticks(y)
ax.set_yticklabels(groups["label"])
ax.invert_yaxis()
ax.set_xlim(0, 100)

ax.set_xlabel("Prediction interval coverage (%)")
ax.set_title("ETA Reliability Across Weather and Traffic")
ax.legend()

fig.tight_layout()
fig.savefig(
    output_dir / "eta_coverage_by_condition.png",
    dpi=200
)
plt.close(fig)

print("Success! Saved two research figures:")
print(output_dir / "eta_coverage_comparison.png")
print(output_dir / "eta_coverage_by_condition.png")
