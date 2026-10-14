
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "catboost_holdout_predictions.csv"
)

REPORTS = ROOT / "reports"


def main():
    print("=== MODEL PERFORMANCE EVALUATION ===")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "load_id", "actual_rate", "predicted_rate"
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required prediction columns")

    if df["load_id"].duplicated().any():
        raise ValueError("Duplicate load IDs detected")

    actual = df["actual_rate"].to_numpy(dtype=float)
    predicted = df["predicted_rate"].to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("Invalid actual rates detected")

    if not np.isfinite(predicted).all():
        raise ValueError("Invalid predictions detected")

    if (actual <= 0).any():
        raise ValueError("Actual freight rates must be positive")

    errors = actual - predicted
    absolute_errors = np.abs(errors)

    # Dollar-based metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Percentage-based metrics
    mape = np.mean(absolute_errors / actual) * 100
    wape = absolute_errors.sum() / actual.sum() * 100

    # Explanatory performance
    r2 = r2_score(actual, predicted)

    # Additional useful measurements
    median_ape = np.median(
        absolute_errors / actual
    ) * 100

    within_10_pct = np.mean(
        absolute_errors / actual <= 0.10
    ) * 100

    within_20_pct = np.mean(
        absolute_errors / actual <= 0.20
    ) * 100

    print(f"\nEvaluated shipments: {len(df):,}")

    print(f"\nMAE:              ${mae:,.2f}")
    print(f"RMSE:             ${rmse:,.2f}")
    print(f"MAPE:             {mape:.2f}%")
    print(f"WAPE:             {wape:.2f}%")
    print(f"R²:               {r2:.4f}")
    print(f"Median APE:       {median_ape:.2f}%")
    print(f"Within 10%:       {within_10_pct:.2f}%")
    print(f"Within 20%:       {within_20_pct:.2f}%")

    results = pd.DataFrame([{
        "evaluated_rows": len(df),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "wape_pct": wape,
        "r2": r2,
        "median_ape_pct": median_ape,
        "within_10_pct": within_10_pct,
        "within_20_pct": within_20_pct,
    }])

    REPORTS.mkdir(parents=True, exist_ok=True)

    output = REPORTS / "holdout_evaluation.csv"
    results.to_csv(output, index=False)

    print(f"\nSaved evaluation: {output}")


if __name__ == "__main__":
    main()
