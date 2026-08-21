"""
Preprocess merged macro-financial dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("data/interim/merged_raw.csv")

OUTPUT_FILE = Path("data/processed/pcmci_input.csv")

REPORT_FOLDER = Path("results/reports")

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("="*60)
print("Initial Dataset")
print("="*60)

print(df.info())

df["Date"] = pd.to_datetime(df["Date"])

df.replace(".", np.nan, inplace=True)

for column in df.columns[1:]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.sort_values("Date", inplace=True)

df.drop_duplicates(subset="Date", inplace=True)

missing = df.isnull().sum()

missing_percent = 100 * missing / len(df)

report = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percent
})

print()
print("="*60)
print("Missing Value Report")
print("="*60)
print(report)

report.to_csv(
    REPORT_FOLDER / "missing_value_report.csv"
)

df.interpolate(
    method="linear",
    inplace=True
)

before = len(df)
df.dropna(inplace=True)
after = len(df)

print()
print(f"Rows Removed : {before-after}")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("="*60)
print("Processed Dataset Saved")
print("="*60)
print(df.head())
print()
print(df.info())