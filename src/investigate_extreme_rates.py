
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "raw" / "train_test.csv"
)


def main():
    print("=== EXTREME FREIGHT RATE INVESTIGATION ===")

    df = pd.read_csv(DATA_PATH)

    # Freight rate per mile
    df["rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    # Mark unusually expensive shipments
    df["is_expensive"] = df["posted_rate"] > 5000

    print("\n1. EXPENSIVE SHIPMENT COUNTS")
    print(df["is_expensive"].value_counts())

    print("\n2. MARKET SIGNAL COMPARISON")

    comparison = df.groupby("is_expensive").agg(
        count=("posted_rate", "count"),
        mean_rate=("posted_rate", "mean"),
        mean_distance=("distance", "mean"),
        mean_market_index=("market_index", "mean"),
        mean_quote_signal=("quote_signal", "mean"),
        mean_rate_per_mile=("rate_per_mile", "mean"),
    )

    print(comparison.round(3).to_string())

    print("\n3. NUMERICAL CORRELATIONS")

    columns = [
        "posted_rate",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "rate_per_mile",
    ]

    print(
        df[columns]
        .corr(numeric_only=True)["posted_rate"]
        .sort_values(ascending=False)
        .round(3)
        .to_string()
    )

    print("\n4. TOP 10 EXPENSIVE SHIPMENTS")

    print(
        df.nlargest(10, "posted_rate")[
            [
                "load_id",
                "date",
                "distance",
                "equipment",
                "market_index",
                "quote_signal",
                "rate_per_mile",
                "posted_rate",
            ]
        ].round(3).to_string(index=False)
    )

    print("\n5. FREIGHT RATE PER MILE DISTRIBUTION")
    print(
        df["rate_per_mile"]
        .describe(percentiles=[0.50, 0.90, 0.95, 0.99])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
