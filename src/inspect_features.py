
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

FILES = {
    "training": "train_test.csv",
    "validation": "validation.csv",
    "december": "december_chart_inputs.csv",
}


def main():
    datasets = {}

    for name, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        df = pd.read_csv(path)
        datasets[name] = df

        print(f"\n=== {name.upper()} ===")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print("\nMissing values:")
        print(df.isna().sum().to_string())

    print("\n=== FEATURE COMPARISON ===")

    for name, df in datasets.items():
        print(f"{name}: {len(df.columns)} columns")

    common = set.intersection(
        *(set(df.columns) for df in datasets.values())
    )

    print("\nCommon columns:")
    print(sorted(common))

    for name, df in datasets.items():
        extra = set(df.columns) - common
        print(f"\nOnly in {name} or other subsets:")
        print(sorted(extra))


if __name__ == "__main__":
    main()
