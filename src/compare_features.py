
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Shared feature engineering module
from features import (
    CATEGORICAL_FEATURES,
    FEATURE_SETS,
    select_features,
)
from experiment_tracking import save_experiment
# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed"

REPORTS = ROOT / "reports"

TRAIN_PATH = DATA / "dev_train.csv"
VALIDATION_PATH = DATA / "dev_validation.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(y_true, predictions):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    return mae, rmse


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=== CATBOOST FEATURE COMPARISON ===")

    # --------------------------------------------------------
    # 1. Load development datasets
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_PATH)
    valid = pd.read_csv(VALIDATION_PATH)

    print(f"\nTraining records: {len(train):,}")
    print(f"Validation records: {len(valid):,}")

    # --------------------------------------------------------
    # 2. Verify chronological separation
    # --------------------------------------------------------

    train_dates = pd.to_datetime(
        train["date"],
        errors="raise",
    )

    valid_dates = pd.to_datetime(
        valid["date"],
        errors="raise",
    )

    if train_dates.max() >= valid_dates.min():
        raise ValueError(
            "Data leakage risk: training dates overlap "
            "with or follow validation dates."
        )

    print("\nChronological split verified.")

    # --------------------------------------------------------
    # 3. Separate target variables
    # --------------------------------------------------------

    y_train = train["posted_rate"]
    y_valid = valid["posted_rate"]

    results = []

    # --------------------------------------------------------
    # 4. Train and evaluate each feature configuration
    # --------------------------------------------------------

    for name in FEATURE_SETS:

        print(f"\nTraining {name}...")

        # Use our shared feature engineering module
        X_train = select_features(
            train,
            name,
        )

        X_valid = select_features(
            valid,
            name,
        )

        # Create a new model for each experiment
        model = CatBoostRegressor(
            **MODEL_CONFIG
        )

        # Train using development training data only
        model.fit(
            X_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES,
        )

        # Predict on development validation data
        predictions = model.predict(X_valid)

        # Calculate evaluation metrics
        mae, rmse = evaluate_model(
            y_valid,
            predictions,
        )

        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")

        results.append({
            "experiment": name,
            "features": len(X_train.columns),
            "mae": mae,
            "rmse": rmse,
        })
    
        # Record this experiment automatically
        record_path = save_experiment(
            experiment_name=f"catboost_{name}",
            model_config=MODEL_CONFIG,
            feature_set=name,
            metrics={
                "mae": float(mae),
                "rmse": float(rmse),
            },
            train_rows=len(X_train),
            validation_rows=len(X_valid),
            train_period=(
                f"{train_dates.min().date()} to "
                f"{train_dates.max().date()}"
            ),
            validation_period=(
                f"{valid_dates.min().date()} to "
                f"{valid_dates.max().date()}"
            ),
        )

        print(f"Experiment recorded: {record_path.name}")

    # --------------------------------------------------------
    # 5. Compare results
    # --------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "mae",
        ascending=True,
    ).reset_index(drop=True)

    print("\n=== FINAL COMPARISON ===")

    print(
        comparison.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # 6. Save experiment results
    # --------------------------------------------------------

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORTS / "feature_comparison.csv"

    comparison.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    # --------------------------------------------------------
    # 7. Display best experiment
    # --------------------------------------------------------

    best = comparison.iloc[0]

    print("\n=== BEST EXPERIMENT ===")

    print(f"Feature set: {best['experiment']}")
    print(f"Number of features: {int(best['features'])}")
    print(f"MAE: ${best['mae']:,.2f}")
    print(f"RMSE: ${best['rmse']:,.2f}")

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
