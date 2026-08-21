"""
Infer economic regimes using the trained HMM.

Outputs:
- Regime labels
- Regime probabilities
- Regime confidence
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

INPUT_FILE = Path("data/processed/regime_features.csv")

MODEL_FOLDER = Path("models/regime_models")

OUTPUT_FILE = Path("data/processed/regime_labels.csv")

# Load Data
df = pd.read_csv(INPUT_FILE)
dates = df["Date"]
X = df.drop(columns=["Date"])

# Load Model
model = joblib.load(MODEL_FOLDER / "best_hmm.pkl")
scaler = joblib.load(MODEL_FOLDER / "regime_scaler.pkl")
X_scaled = scaler.transform(X)

# Predict
regimes = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence = probabilities.max(axis=1)
entropy = -np.sum(
    probabilities * np.log(probabilities + 1e-12),
    axis=1
)

# Save
output = df.copy()
output["Regime"] = regimes
for i in range(probabilities.shape[1]):
    output[f"RegimeProb_{i}"] = probabilities[:, i]

output["Confidence"] = confidence
output["Entropy"] = entropy

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("Regime Inference Complete")
print("=" * 60)
print(output.head())
print()
print("Regime Distribution")
print(output["Regime"].value_counts())