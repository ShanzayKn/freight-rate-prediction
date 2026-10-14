
from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the original training dataset
DATA_PATH = "data/raw/train-test.csv"


def main():
    print("=== FREIGHT RATE DATA EXPLORATION ===")

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Basic structure
    print("\n1. DATASET SHAPE")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # Preview first five records
    print("\n2. FIRST FIVE RECORDS")
    print(df.head().to_string())

    # Identify columns and types
    print("\n3. COLUMN INFORMATION")
    df.info()

    # Count missing values
    print("\n4. MISSING VALUES")
    print(df.isnull().sum().to_string())

    # Identify duplicate records
    print("\n5. DUPLICATE ROWS")
    print(df.duplicated().sum())

    # Numerical summaries
    print("\n6. NUMERICAL STATISTICS")
    print(df.describe().to_string())

    # Actual freight rate distribution
    print("\n7. FREIGHT RATE STATISTICS")
    print(df["posted_rate"].describe().to_string())

    # Investigate invalid cargo weights
    print("\n8. INVALID WEIGHT RECORDS")
    invalid_weights = df[df["weight"] <= 0]
    print(f"Invalid weights: {len(invalid_weights)}")
    print(
        invalid_weights[
            ["load_id", "distance", "weight", "posted_rate"]
        ].head(10).to_string(index=False)
    )

    # Investigate unusually expensive shipments
    print("\n9. EXPENSIVE SHIPMENTS")
    expensive_loads = df.nlargest(10, "posted_rate")
    print(
        expensive_loads[
            ["load_id", "distance", "equipment",
             "weight", "posted_rate"]
        ].to_string(index=False)
    )

    # Count equipment categories
    print("\n10. EQUIPMENT DISTRIBUTION")
    print(df["equipment"].value_counts().to_string())

    # Inspect date coverage
    print("\n11. DATE RANGE")
    dates = pd.to_datetime(df["date"], errors="coerce")
    print("Earliest:", dates.min())
    print("Latest:", dates.max())
    print("Invalid dates:", dates.isna().sum())

    # Check invalid distances and rates
    print("\n12. INVALID DISTANCES AND RATES")
    print("Non-positive distances:", (df["distance"] <= 0).sum())
    print("Non-positive rates:", (df["posted_rate"] <= 0).sum())

    print("\n13. NEGATIVE WEIGHT INVESTIGATION")

    negative = df[df["weight"] < 0].copy()
    positive = df[df["weight"] > 0].copy()

    print("\nNegative weights by equipment:")
    print(negative["equipment"].value_counts())

    print("\nPositive weight statistics:")
    print(positive["weight"].describe())

    print("\nAbsolute negative weight statistics:")
    print(negative["weight"].abs().describe())

    print("\nNegative weights by month:")
    negative["month"] = pd.to_datetime(
        negative["date"]
    ).dt.to_period("M")

    print(
        negative.groupby("month")
        .size()
        .to_string()
    )

    print("\nNegative weight rate by equipment (%):")
    negative_rate = (
        df.assign(is_negative=df["weight"] < 0)
        .groupby("equipment")["is_negative"]
        .mean()
        .mul(100)
        .round(2)
    )
    print(negative_rate)


if __name__ == "__main__":
    main()
