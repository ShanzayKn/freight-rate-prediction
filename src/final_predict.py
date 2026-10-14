
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from features import (
    CATEGORICAL_FEATURES,
    select_features,
)


ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA / "train_test.csv"
VALIDATION_PATH = DATA / "validation.csv"
TEMPLATE_PATH = DATA / "validation_predictions_template.csv"
DECEMBER_PATH = DATA / "december_chart_inputs.csv"
DECEMBER_OUTPUT = (
    ROOT / "data" / "processed" / "december_predictions.csv"
)
OUTPUT_PREDICTIONS = ROOT / "validation_predictions.csv"

FEATURE_SET = "basic"

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


def main():
    print("=== FINAL FREIGHT RATE PREDICTION ===")

    # Load labeled historical data and future inputs
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    december = pd.read_csv(DECEMBER_PATH)

    # Check required counts
    if len(validation) != 12000:
        raise ValueError("Expected exactly 12,000 validation loads.")

    if len(december) != 31:
        raise ValueError("Expected exactly 31 December records.")

    # Never use future target prices during training.
    X_train = select_features(train, FEATURE_SET)
    y_train = train["posted_rate"]

    X_validation = select_features(validation, FEATURE_SET)
    X_december = select_features(december, FEATURE_SET)

    # Ensure identical feature schema across inputs
    assert list(X_train.columns) == list(X_validation.columns)
    assert list(X_train.columns) == list(X_december.columns)

    print(f"\nTraining records: {len(train):,}")
    print(f"Feature set: {FEATURE_SET}")

    # Train the selected configuration
    model = CatBoostRegressor(**MODEL_CONFIG)

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "catboost_final.cbm"
    model.save_model(str(model_path))

    # --------------------------------------------------------
    # Predict the 12,000 validation shipments
    # --------------------------------------------------------

    validation_predictions = model.predict(X_validation)

    if not np.isfinite(validation_predictions).all():
        raise ValueError("Non-finite validation predictions.")

    validation_predictions = np.maximum(
        validation_predictions,
        0.01,
    )

    prediction_map = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": validation_predictions,
    })

    if prediction_map["load_id"].duplicated().any():
        raise ValueError("Duplicate validation load IDs.")

    # Preserve the template's exact load_id ordering
    if len(template) != len(validation):
        raise ValueError("Template and validation row counts differ.")

    if "load_id" not in template.columns:
        raise ValueError("Template is missing load_id.")

    output = template[["load_id"]].merge(
        prediction_map,
        on="load_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if output["predicted_rate"].isna().any():
        raise ValueError("Some template IDs have no predictions.")

    if len(output) != 12000:
        raise ValueError("Unexpected prediction output length.")

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print(
        f"\nSaved 12,000 predictions: "
        f"{OUTPUT_PREDICTIONS}"
    )

    # --------------------------------------------------------
    # Predict the 31 December shipments
    # --------------------------------------------------------

    december_predictions = model.predict(X_december)

    if not np.isfinite(december_predictions).all():
        raise ValueError("Non-finite December predictions.")

    december["predicted_rate"] = np.maximum(
        december_predictions,
        0.01,
    )

    # Preserve the required seven-column format
    december = december[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ]

    DECEMBER_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    december.to_csv(
        DECEMBER_OUTPUT,
        index=False,
    )

    print("Saved 31 December predictions.")
    print(f"Final model saved: {model_path}")
    print("\nFinal prediction pipeline completed.")


if __name__ == "__main__":
    main()
