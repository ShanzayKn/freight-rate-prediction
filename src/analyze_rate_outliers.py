
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "processed" / "holdout.csv"

PREDICTIONS_PATH = (
    ROOT / "data" / "processed"
    / "catboost_holdout_predictions.csv"
)

HIGH_RATE_THRESHOLD = 4.0


def summarize_rates(df, label):
    df = df.copy()

    df["rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    df["high_rate"] = (
        df["rate_per_mile"] > HIGH_RATE_THRESHOLD
    )

    count = int(df["high_rate"].sum())
    percentage = df["high_rate"].mean() * 100

    print(f"\n=== {label} ===")
    print(f"Total shipments: {len(df):,}")
    print(f"High-rate shipments: {count:,}")
    print(f"Percentage: {percentage:.2f}%")

    print("\nHigh-rate shipments by equipment:")
    print(
        df.loc[df["high_rate"], "equipment"]
        .value_counts()
        .to_string()
    )

    print("\nHigh-rate shipments by month:")
    dates = pd.to_datetime(df["date"])

    print(
        df.assign(month=dates.dt.to_period("M"))
        .loc[lambda x: x["high_rate"]]
        .groupby("month")
        .size()
        .to_string()
    )

    return df


def main():
    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    train = summarize_rates(train, "TRAINING")
    holdout = summarize_rates(holdout, "HOLDOUT")

    print("\n=== HOLDOUT ERROR CONTRIBUTION ===")

    merged = holdout.merge(
        predictions[
            ["load_id", "predicted_rate"]
        ],
        on="load_id",
        validate="one_to_one",
    )

    merged["absolute_error"] = np.abs(
        merged["posted_rate"] - merged["predicted_rate"]
    )

    merged["squared_error"] = (
        merged["posted_rate"] - merged["predicted_rate"]
    ) ** 2

    summary = merged.groupby("high_rate").agg(
        count=("load_id", "count"),
        mae=("absolute_error", "mean"),
        total_absolute_error=("absolute_error", "sum"),
        total_squared_error=("squared_error", "sum"),
    )

    summary["share_absolute_error_pct"] = (
        summary["total_absolute_error"]
        / summary["total_absolute_error"].sum()
        * 100
    )

    summary["share_squared_error_pct"] = (
        summary["total_squared_error"]
        / summary["total_squared_error"].sum()
        * 100
    )

    print(summary.round(2).to_string())

    print("\n=== TOP 10 RATE-PER-MILE SHIPMENTS ===")

    print(
        merged.nlargest(10, "rate_per_mile")[
            [
                "load_id",
                "date",
                "equipment",
                "distance",
                "rate_per_mile",
                "posted_rate",
                "predicted_rate",
                "absolute_error",
            ]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
