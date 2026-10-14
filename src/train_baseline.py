
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "processed" / "holdout.csv"

TARGET = "posted_rate"


def make_features(df):
    """Prepare a common feature set for both prediction scenarios."""

    X = df[
        ["pickup", "delivery", "distance",
         "equipment", "weight", "date"]
    ].copy()

    dates = pd.to_datetime(X["date"], errors="raise")

    X["month"] = dates.dt.month
    X["day_of_week"] = dates.dt.dayofweek
    X["day_of_year"] = dates.dt.dayofyear

    # Make this transformation independent of earlier cleaning steps.
    X["weight"] = X["weight"].abs()

    X = X.drop(columns=["date"])

    return X


def make_model():
    numeric_features = [
        "distance",
        "weight",
        "month",
        "day_of_week",
        "day_of_year",
    ]

    categorical_features = [
        "pickup",
        "delivery",
        "equipment",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0)),
    ])

    return model


def evaluate(name, model, X_train, y_train, X_holdout, y_holdout):
    model.fit(X_train, y_train)

    predictions = model.predict(X_holdout)

    mae = mean_absolute_error(y_holdout, predictions)
    rmse = np.sqrt(mean_squared_error(y_holdout, predictions))

    print(f"\n{name}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")

    return model


def main():
    print("=== BASELINE MODEL TRAINING ===")

    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    X_train = make_features(train)
    y_train = train[TARGET]

    X_holdout = make_features(holdout)
    y_holdout = holdout[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Holdout rows:  {len(X_holdout):,}")

    # Model 1: always predicts the training mean.
    dummy = DummyRegressor(strategy="mean")

    evaluate(
        "Dummy Mean Baseline",
        dummy,
        X_train, y_train,
        X_holdout, y_holdout,
    )

    # Model 2: regularized linear regression.
    ridge = make_model()

    evaluate(
        "Ridge Regression Baseline",
        ridge,
        X_train, y_train,
        X_holdout, y_holdout,
    )


if __name__ == "__main__":
    main()
