
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "train_test.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_DATE = pd.Timestamp("2025-09-01")
DEV_SPLIT_DATE = pd.Timestamp("2025-07-01")
def clean_data(df):
    """Apply deterministic, row-level data corrections."""
    df = df.copy()

    # Preserve information about sign errors
    df["weight_was_negative"] = df["weight"] < 0

    # Correct suspected sign errors
    df["weight"] = df["weight"].abs()

    # Convert dates to proper datetime values
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df


def main():
    print("=== DATA PREPARATION ===")

    df = pd.read_csv(RAW_DATA)

    # Check required columns
    required = {"load_id", "date", "weight", "posted_rate"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Apply row-level cleaning
    df = clean_data(df)

    # Sort chronologically
    df = df.sort_values(["date", "load_id"])

    # Time-based separation
    train = df[df["date"] < SPLIT_DATE].copy()
    validation = df[df["date"] >= SPLIT_DATE].copy()
    
    # Internal development split
    dev_train = train[
        train["date"] < DEV_SPLIT_DATE
    ].copy()

    dev_validation = train[
        train["date"] >= DEV_SPLIT_DATE
    ].copy()

    # Confirm chronological separation
    assert not dev_train.empty
    assert not dev_validation.empty

    assert (
        dev_train["date"].max()
        < dev_validation["date"].min()
    )

    # Safety checks
    assert not train.empty
    assert not validation.empty
    assert train["date"].max() < validation["date"].min()

    # Save prepared datasets
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DIR / "holdout.csv", index=False)
    
    
    dev_train.to_csv(
        PROCESSED_DIR / "dev_train.csv",
        index=False,
    )

    dev_validation.to_csv(
        PROCESSED_DIR / "dev_validation.csv",
        index=False,
    )

    print("\n=== DEVELOPMENT SPLIT ===")
    print(f"Development training: {len(dev_train):,}")
    print(f"Development validation: {len(dev_validation):,}")

    print(
        "Development training period:",
        dev_train["date"].min().date(),
        "to",
        dev_train["date"].max().date(),
    )

    print(
        "Development validation period:",
        dev_validation["date"].min().date(),
        "to",
        dev_validation["date"].max().date(),
    )

    print(f"Training records: {len(train):,}")
    print(f"Validation records: {len(validation):,}")

    print(
        "Training period:",
        train["date"].min().date(),
        "to",
        train["date"].max().date()
    )

    print(
        "Validation period:",
        validation["date"].min().date(),
        "to",
        validation["date"].max().date()
    )

    print(
        "Corrected negative weights:",
        int(df["weight_was_negative"].sum())
    )

    print("Prepared files saved to data/processed/")


if __name__ == "__main__":
    main()
