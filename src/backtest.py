
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import (
    CATEGORICAL_FEATURES,
    FEATURE_SETS,
    select_features,
)
from experiment_tracking import save_experiment


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
REPORT_DIR = ROOT / "reports"

MODEL_CONFIG = {
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
}

# Each fold trains on all earlier available months.
VALIDATION_MONTHS = [
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
]


def main():
    print("=== TEMPORAL BACKTESTING ===")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    results = []

    for month in VALIDATION_MONTHS:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)

        train = df[df["date"] < start].copy()
        valid = df[
            (df["date"] >= start) & (df["date"] < end)
        ].copy()

        if train.empty or valid.empty:
            raise ValueError(f"Empty split for {month}")

        if train["date"].max() >= valid["date"].min():
            raise ValueError("Chronological split violation")

        print(
            f"\n=== {month}: "
            f"{len(train):,} train / {len(valid):,} validation ==="
        )

        for feature_set in FEATURE_SETS:
            X_train = select_features(train, feature_set)
            X_valid = select_features(valid, feature_set)

            model = CatBoostRegressor(**MODEL_CONFIG)

            model.fit(
                X_train,
                train["posted_rate"],
                cat_features=CATEGORICAL_FEATURES,
            )

            predictions = model.predict(X_valid)

            mae = mean_absolute_error(
                valid["posted_rate"], predictions
            )
            rmse = np.sqrt(
                mean_squared_error(
                    valid["posted_rate"], predictions
                )
            )

            results.append({
                "month": month,
                "feature_set": feature_set,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "mae": float(mae),
                "rmse": float(rmse),
            })

            print(
                f"{feature_set:8s} | "
                f"MAE ${mae:,.2f} | "
                f"RMSE ${rmse:,.2f}"
            )

            save_experiment(
                experiment_name=f"backtest_{month}_{feature_set}",
                model_config=MODEL_CONFIG,
                feature_set=feature_set,
                metrics={
                    "mae": float(mae),
                    "rmse": float(rmse),
                },
                train_rows=len(train),
                validation_rows=len(valid),
                train_period=(
                    f"{train['date'].min().date()} to "
                    f"{train['date'].max().date()}"
                ),
                validation_period=(
                    f"{valid['date'].min().date()} to "
                    f"{valid['date'].max().date()}"
                ),
            )

    results_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORT_DIR / "backtest_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n=== BACKTEST SUMMARY ===")

    summary = results_df.groupby("feature_set").agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        std_mae=("mae", "std"),
    ).sort_values("mean_mae")

    print(summary.round(2).to_string())

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
