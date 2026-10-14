
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    ROOT / "data" / "processed"
    / "catboost_holdout_predictions.csv"
)

HOLDOUT_PATH = (
    ROOT / "data" / "processed"
    / "holdout.csv"
)


def main():
    print("=== CATBOOST ERROR ANALYSIS ===")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    # Join original shipment features with predictions
    df = predictions.merge(
        holdout[
            ["load_id", "distance", "equipment", "weight", "date"]
        ],
        on="load_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(predictions) or df["distance"].isna().any():
        raise ValueError("Prediction and holdout records do not match.")

    # Signed error: positive means overprediction
    df["error"] = df["predicted_rate"] - df["actual_rate"]

    df["absolute_percentage_error"] = (
        df["absolute_error"] / df["actual_rate"] * 100
    )

    print("\n1. GENERAL ERROR STATISTICS")
    print(df["absolute_error"].describe().to_string())

    print("\n2. TOP 10 WORST PREDICTIONS")
    worst = df.nlargest(10, "absolute_error")

    print(
        worst[
            [
                "load_id",
                "distance",
                "equipment",
                "actual_rate",
                "predicted_rate",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    print("\n3. ERROR BY EQUIPMENT")
    by_equipment = df.groupby("equipment").agg(
        count=("load_id", "count"),
        mae=("absolute_error", "mean"),
        mean_actual_rate=("actual_rate", "mean"),
    )

    print(by_equipment.round(2).to_string())

    print("\n4. ERROR BY DISTANCE")

    df["distance_group"] = pd.cut(
        df["distance"],
        bins=[0, 500, 1000, 2000, np.inf],
        labels=[
            "0-500",
            "500-1000",
            "1000-2000",
            "2000+",
        ],
    )

    by_distance = df.groupby(
        "distance_group",
        observed=True,
    ).agg(
        count=("load_id", "count"),
        mae=("absolute_error", "mean"),
    )

    print(by_distance.round(2).to_string())

    print("\n5. ERROR BY PRICE RANGE")

    df["price_group"] = pd.cut(
        df["actual_rate"],
        bins=[0, 1000, 3000, 5000, np.inf],
        labels=[
            "Under $1,000",
            "$1,000-$3,000",
            "$3,000-$5,000",
            "Over $5,000",
        ],
    )

    by_price = df.groupby(
        "price_group",
        observed=True,
    ).agg(
        count=("load_id", "count"),
        mae=("absolute_error", "mean"),
    )

    print(by_price.round(2).to_string())

    print("\n6. OVERPREDICTION VS UNDERPREDICTION")
    print(f"Mean signed error: ${df['error'].mean():,.2f}")

    print(
        "Overpredictions:",
        int((df["error"] > 0).sum()),
    )

    print(
        "Underpredictions:",
        int((df["error"] < 0).sum()),
    )

    print("\n7. PERCENTAGE ERROR")
    print(
        "Median absolute percentage error:",
        f"{df['absolute_percentage_error'].median():.2f}%",
    )

    print(
        "90th percentile absolute error:",
        f"${df['absolute_error'].quantile(0.90):,.2f}",
    )


if __name__ == "__main__":
    main()
