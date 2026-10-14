
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import shared feature engineering
from features import CATEGORICAL_FEATURES, select_features


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "processed" / "holdout.csv"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "posted_rate"

# Preserve our original common-feature model
FEATURE_SET = "basic"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 100,
    "thread_count": -1,
    "allow_writing_files": False,
}


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(y_true, predictions):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    return mae, rmse


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=== CATBOOST MODEL TRAINING ===")

    # --------------------------------------------------------
    # 1. Load prepared training and holdout datasets
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    print(f"\nTraining records: {len(train):,}")
    print(f"Holdout records: {len(holdout):,}")

    # --------------------------------------------------------
    # 2. Verify chronological separation
    # --------------------------------------------------------

    train_dates = pd.to_datetime(
        train["date"],
        errors="raise",
    )

    holdout_dates = pd.to_datetime(
        holdout["date"],
        errors="raise",
    )

    if train_dates.max() >= holdout_dates.min():
        raise ValueError(
            "Training and holdout periods overlap "
            "or are incorrectly ordered."
        )

    print("Chronological split verified.")

    # --------------------------------------------------------
    # 3. Prepare features using shared module
    # --------------------------------------------------------

    X_train = select_features(train, FEATURE_SET)
    y_train = train[TARGET]

    X_holdout = select_features(holdout, FEATURE_SET)
    y_holdout = holdout[TARGET]

    print(f"Feature set: {FEATURE_SET}")
    print(f"Number of features: {len(X_train.columns)}")

    # Confirm identical training and inference columns
    if list(X_train.columns) != list(X_holdout.columns):
        raise ValueError(
            "Training and holdout feature schemas do not match."
        )

    # --------------------------------------------------------
    # 4. Initialize CatBoost
    # --------------------------------------------------------

    model = CatBoostRegressor(**MODEL_CONFIG)

    # --------------------------------------------------------
    # 5. Train the model
    # --------------------------------------------------------

    print("\nTraining CatBoost model...")

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    # --------------------------------------------------------
    # 6. Generate holdout predictions
    # --------------------------------------------------------

    predictions = model.predict(X_holdout)

    # --------------------------------------------------------
    # 7. Evaluate model performance
    # --------------------------------------------------------

    mae, rmse = evaluate_model(
        y_holdout,
        predictions,
    )

    print("\n=== CATBOOST RESULTS ===")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")

    # --------------------------------------------------------
    # 8. Display feature importance
    # --------------------------------------------------------

    print("\n=== FEATURE IMPORTANCE ===")

    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print(importances.to_string(index=False))

    # --------------------------------------------------------
    # 9. Save trained model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "catboost_baseline.cbm"

    model.save_model(str(model_path))

    print(f"\nModel saved: {model_path}")

    # --------------------------------------------------------
    # 10. Save holdout predictions for error analysis
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame({
        "load_id": holdout["load_id"],
        "actual_rate": y_holdout,
        "predicted_rate": predictions,
        "absolute_error": np.abs(
            y_holdout.to_numpy() - predictions
        ),
    })

    predictions_path = (
        OUTPUT_DIR / "catboost_holdout_predictions.csv"
    )

    results.to_csv(
        predictions_path,
        index=False,
    )

    print(f"Predictions saved: {predictions_path}")
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
