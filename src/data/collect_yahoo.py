"""
Download monthly financial market data from Yahoo Finance.
"""

from pathlib import Path
import yfinance as yf

START_DATE = "1959-01-01"
END_DATE = "2026-06-01"

OUTPUT_FOLDER = Path("data/raw/yahoo")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TICKERS = {

    "^GSPC": "SP500",

    "^IXIC": "NASDAQ",

    "^VIX": "VIX",

    "GLD": "GOLD",

    "CL=F": "WTI_OIL",

    "BZ=F": "BRENT_OIL"
}

def download_ticker(ticker, filename):

    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"No data found for {ticker}")
        return

    df.reset_index(inplace=True)
    output_path = OUTPUT_FOLDER / f"{filename}.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved -> {output_path}")

def main():

    print("=" * 50)
    print("Yahoo Finance Data Downloader")
    print("=" * 50)

    for ticker, filename in TICKERS.items():
        download_ticker(ticker, filename)

    print("\nDownload Complete!")


if __name__ == "__main__":
    main()