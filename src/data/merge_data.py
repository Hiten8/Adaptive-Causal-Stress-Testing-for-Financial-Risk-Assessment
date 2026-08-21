"""
Merge all FRED and Yahoo Finance datasets
into one monthly dataframe.
"""

from pathlib import Path
import pandas as pd

FRED_FOLDER = Path("data/raw/fred")
YAHOO_FOLDER = Path("data/raw/yahoo")

OUTPUT_FILE = Path("data/interim/merged_raw.csv")

START_DATE = "1995-01-01"
END_DATE = "2025-12-31"

def load_fred_csv(filepath):

    """
    Loads one FRED csv.

    Returns
    Date
    SeriesName
    """

    series_name = filepath.stem
    df = pd.read_csv(filepath)
    df.columns = ["Date", series_name]
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def load_yahoo_csv(filepath):

    """
    Loads one Yahoo csv.

    Uses only Adjusted Close
    (Close after auto_adjust=True)
    """

    series_name = filepath.stem
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]]
    df.rename(columns={"Close": series_name},
              inplace=True)

    return df

print("Loading FRED data...")

fred_files = sorted(FRED_FOLDER.glob("*.csv"))

master_df = None

for file in fred_files:
    df = load_fred_csv(file)
    if master_df is None:
        master_df = df

    else:
        master_df = master_df.merge(
            df,
            on="Date",
            how="outer"
        )

print("Loading Yahoo data...")

yahoo_files = sorted(YAHOO_FOLDER.glob("*.csv"))

for file in yahoo_files:
    df = load_yahoo_csv(file)
    master_df = master_df.merge(
        df,
        on="Date",
        how="outer"
    )

master_df = master_df[
    (master_df["Date"] >= START_DATE) &
    (master_df["Date"] <= END_DATE)
]

master_df.sort_values("Date", inplace=True)

master_df.reset_index(drop=True,
                      inplace=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("="*60)
print("Merged Dataset Created")
print("="*60)
print(master_df.head())
print()
print(master_df.info())
print()
print(f"Saved to:\n{OUTPUT_FILE}")