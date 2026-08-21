"""
Automatically determine the optimal number of economic regimes
using Gaussian Hidden Markov Models (HMM).

The script:
1. Loads regime features
2. Standardizes the data
3. Trains HMMs with different numbers of regimes
4. Uses multiple random initializations for robustness
5. Computes AIC and BIC
6. Selects the optimal model using BIC
7. Saves the best HMM model
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

INPUT_FILE = Path("data/processed/regime_features.csv")

REPORT_FOLDER = Path("results/reports")
MODEL_FOLDER = Path("models/regime_models")

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

MIN_REGIMES = 2
MAX_REGIMES = 6

N_INITIALIZATIONS = 10
MAX_ITERATIONS = 500

COVARIANCE_TYPE = "full"

# Load Data
print("=" * 70)
print("Loading Regime Feature Dataset")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
X = df.drop(columns=["Date"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples = X_scaled.shape[0]
n_features = X_scaled.shape[1]

def count_parameters(model, n_features):
    """
    Approximate number of free parameters
    in a Gaussian Hidden Markov Model.
    """

    k = model.n_components
    transition = k * (k - 1)
    initial = k - 1
    means = k * n_features
    covariances = k * n_features * (n_features + 1) / 2

    return transition + initial + means + covariances

# Model Selection
results = []
overall_best_model = None
overall_best_bic = np.inf

print("\nSearching for Optimal Number of Regimes...\n")

for n_regimes in range(MIN_REGIMES, MAX_REGIMES + 1):
    print("-" * 60)
    print(f"Testing {n_regimes} Regimes")
    print("-" * 60)
    best_model = None
    best_loglik = -np.inf

    # Multiple Random Initializations
    for seed in range(N_INITIALIZATIONS):
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=COVARIANCE_TYPE,
            n_iter=MAX_ITERATIONS,
            random_state=seed
        )

        try:
            model.fit(X_scaled)
            loglik = model.score(X_scaled)
            if loglik > best_loglik:
                best_loglik = loglik
                best_model = model

        except Exception:
            continue

    if best_model is None:
        print("Model training failed.\n")
        continue

    n_params = count_parameters(best_model, n_features)
    aic = -2 * best_loglik + 2 * n_params
    bic = -2 * best_loglik + np.log(n_samples) * n_params

    results.append({

        "Regimes": n_regimes,

        "LogLikelihood": best_loglik,

        "AIC": aic,

        "BIC": bic

    })

    print(f"Best Log Likelihood : {best_loglik:.2f}")
    print(f"AIC                : {aic:.2f}")
    print(f"BIC                : {bic:.2f}")
    print()

    if bic < overall_best_bic:
        overall_best_bic = bic
        overall_best_model = best_model

# Results
results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_FOLDER / "regime_selection.csv",
    index=False
)

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    results_df["Regimes"],
    results_df["AIC"],
    marker="o",
    linewidth=2,
    label="AIC"
)

plt.plot(
    results_df["Regimes"],
    results_df["BIC"],
    marker="s",
    linewidth=2,
    label="BIC"
)

plt.xlabel("Number of Regimes")
plt.ylabel("Criterion Value")
plt.title("HMM Model Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    REPORT_FOLDER / "regime_selection.png",
    dpi=300
)

plt.close()

# Save Best Model
joblib.dump(
    overall_best_model,
    MODEL_FOLDER / "best_hmm.pkl"
)

# Save Scaler
joblib.dump(
    scaler,
    MODEL_FOLDER / "regime_scaler.pkl"
)

# Print Best Model
best_row = results_df.loc[results_df["BIC"].idxmin()]

print("\n")
print("=" * 70)
print("Optimal Number of Regimes")
print("=" * 70)
print(best_row)
print("\n")
print(f"Best HMM saved to:\n{MODEL_FOLDER / 'best_hmm.pkl'}")
print(f"Scaler saved to:\n{MODEL_FOLDER / 'regime_scaler.pkl'}")
print("\nFinished.")