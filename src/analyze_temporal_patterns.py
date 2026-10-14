
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "train.csv"


def main():
    print("=== TEMPORAL FREIGHT ANALYSIS ===")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    df["rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    df["high_rate"] = df["rate_per_mile"] > 4

    print("\n1. MONTHLY DATA DISTRIBUTION")

    monthly = df.groupby("month").agg(
        shipments=("load_id", "count"),
        mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        mean_distance=("distance", "mean"),
        mean_rate_per_mile=("rate_per_mile", "mean"),
        median_rate_per_mile=("rate_per_mile", "median"),
        high_rate_count=("high_rate", "sum"),
        high_rate_percentage=("high_rate", "mean"),
    )

    monthly["high_rate_percentage"] *= 100

    print(monthly.round(2).to_string())

    print("\n2. DEVELOPMENT TRAIN VS VALIDATION")

    df["period"] = df["date"].apply(
        lambda d: (
            "Jan-Jun Training"
            if d < pd.Timestamp("2025-07-01")
            else "Jul-Aug Validation"
        )
    )

    comparison = df.groupby("period").agg(
        shipments=("load_id", "count"),
        mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        mean_distance=("distance", "mean"),
        mean_rpm=("rate_per_mile", "mean"),
        high_rate_count=("high_rate", "sum"),
        high_rate_pct=("high_rate", "mean"),
    )

    comparison["high_rate_pct"] *= 100

    print(comparison.round(3).to_string())


if __name__ == "__main__":
    main()
