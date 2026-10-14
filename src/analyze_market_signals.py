
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "train.csv"


def main():
    print("=== MARKET SIGNAL ANALYSIS ===")

    df = pd.read_csv(DATA_PATH)

    # For analysis only: do not use a target-derived
    # rate_per_mile as an inference feature.
    df["rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    print("\n1. CORRELATION WITH RATE PER MILE")

    columns = [
        "rate_per_mile",
        "market_index",
        "quote_signal",
        "weight",
        "distance",
    ]

    print(
        df[columns]
        .corr()["rate_per_mile"]
        .round(4)
        .to_string()
    )

    print("\n2. MARKET INDEX QUARTILES")

    valid_market = df.dropna(subset=["market_index"]).copy()

    valid_market["market_group"] = pd.qcut(
        valid_market["market_index"],
        q=4,
        duplicates="drop",
    )

    print(
        valid_market.groupby(
            "market_group",
            observed=True,
        )
        .agg(
            count=("posted_rate", "count"),
            median_rpm=("rate_per_mile", "median"),
            mean_rpm=("rate_per_mile", "mean"),
        )
        .round(3)
        .to_string()
    )

    print("\n3. QUOTE SIGNAL QUARTILES")

    df["quote_group"] = pd.qcut(
        df["quote_signal"],
        q=4,
        duplicates="drop",
    )

    print(
        df.groupby(
            "quote_group",
            observed=True,
        )
        .agg(
            count=("posted_rate", "count"),
            median_rpm=("rate_per_mile", "median"),
            mean_rpm=("rate_per_mile", "mean"),
        )
        .round(3)
        .to_string()
    )

    print("\n4. HIGH-RATE VS NORMAL SIGNAL DISTRIBUTIONS")

    df["high_rate"] = df["rate_per_mile"] > 4

    print(
        df.groupby("high_rate")
        .agg(
            count=("posted_rate", "count"),
            median_market=("market_index", "median"),
            median_quote=("quote_signal", "median"),
            mean_market=("market_index", "mean"),
            mean_quote=("quote_signal", "mean"),
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
