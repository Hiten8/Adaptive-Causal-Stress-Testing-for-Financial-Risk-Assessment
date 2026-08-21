"""
Adaptive Regime-Aware Window Generator

Generates rolling windows together with regime-aware metadata.
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/regime_labels.csv")

WINDOW_FOLDER = Path("data/windows")

METADATA_FOLDER = Path("data/metadata")

WINDOW_FOLDER.mkdir(parents=True, exist_ok=True)

METADATA_FOLDER.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 120   # months

STEP_SIZE = 1

# Load Data
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# Generate Windows
metadata = []
window_id = 0
for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
    end = start + WINDOW_SIZE
    window = df.iloc[start:end].copy()

    # Window Information
    start_date = window["Date"].iloc[0]
    end_date = window["Date"].iloc[-1]

    # Dominant Regime
    dominant_regime = int(
        window["Regime"].mode()[0]
    )

    # Regime Distribution
    regime_distribution = (
        window["Regime"]
        .value_counts(normalize=True)
    )

    # Confidence
    avg_confidence = window["Confidence"].mean()
    avg_entropy = window["Entropy"].mean()

    # Save Window
    filename = f"window_{window_id:04d}.csv"
    window.to_csv(
        WINDOW_FOLDER / filename,
        index=False
    )

    # Metadata
    row = {
        "WindowID": window_id,
        "File": filename,
        "StartDate": start_date,
        "EndDate": end_date,
        "DominantRegime": dominant_regime,
        "AverageConfidence": avg_confidence,
        "AverageEntropy": avg_entropy
    }

    # Regime percentages
    for regime in sorted(df["Regime"].unique()):
        row[f"Regime_{regime}"] = (
            regime_distribution.get(regime, 0)
        )

    metadata.append(row)
    window_id += 1

# Save Metadata
metadata = pd.DataFrame(metadata)
metadata.to_csv(
    METADATA_FOLDER / "window_metadata.csv",
    index=False
)

print("=" * 60)
print("Adaptive Windows Generated")
print("=" * 60)
print()
print(metadata.head())
print()
print(f"Total Windows : {len(metadata)}")