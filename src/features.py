
import pandas as pd


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

BASIC_FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
]

MARKET_FEATURES = BASIC_FEATURES + [
    "market_index",
]

FULL_FEATURES = MARKET_FEATURES + [
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

FEATURE_SETS = {
    "basic": BASIC_FEATURES,
    "market": MARKET_FEATURES,
    "full": FULL_FEATURES,
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic, reusable feature transformations.

    Works on training and inference data without needing
    the actual target price.
    """
    data = df.copy()

    required = {
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required input columns: {sorted(missing)}"
        )

    dates = pd.to_datetime(data["date"], errors="raise")

    data["month"] = dates.dt.month
    data["day_of_week"] = dates.dt.dayofweek
    data["day_of_year"] = dates.dt.dayofyear

    data["weight"] = data["weight"].abs()

    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
        )

    return data


def select_features(
    df: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    """
    Prepare and select the requested model features.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    prepared = prepare_features(df)
    columns = FEATURE_SETS[feature_set]

    missing = set(columns) - set(prepared.columns)

    if missing:
        raise ValueError(
            f"Missing features for '{feature_set}': "
            f"{sorted(missing)}"
        )

    return prepared[columns].copy()
