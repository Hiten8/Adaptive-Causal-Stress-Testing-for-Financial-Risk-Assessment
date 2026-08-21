"""
Regime Feature Engineering

Creates economically meaningful variables
for Hidden Markov Model regime detection.
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/pcmci_input.csv")

OUTPUT_FILE = Path("data/processed/regime_features.csv")

# Load
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# Feature Engineering
features = pd.DataFrame()
features["Date"] = df["Date"]

# Inflation Rate
features["Inflation"] = df["CPIAUCSL"].pct_change(12) * 100

# Industrial Production Growth
features["IndustrialProductionGrowth"] = (
    df["INDPRO"].pct_change(12) * 100
)

# Money Supply Growth
features["MoneySupplyGrowth"] = (
    df["M2SL"].pct_change() * 100
)

# Monthly Market Returns
features["SP500_Return"] = (
    df["SP500"].pct_change() * 100
)

features["NASDAQ_Return"] = (
    df["NASDAQ"].pct_change() * 100
)

# VIX Change
features["VIX_Change"] = (
    df["VIX"].pct_change() * 100
)

# Unemployment Change
features["Unemployment_Change"] = (
    df["UNRATE"].diff()
)

# Fed Funds
features["FedFunds"] = df["FEDFUNDS"]

# Yield Curve Spread
features["YieldCurveSpread"] = (
    df["GS10"]
    -
    df["TB3MS"]
)

# Remove Initial NaN
features.dropna(inplace=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("Regime Feature Engineering Complete")
print("=" * 60)
print(features.head())
print()
print(features.describe())