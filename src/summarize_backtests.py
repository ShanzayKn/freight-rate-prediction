
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = ROOT / "reports" / "backtest_results.csv"


def main():
    print("=== POOLED BACKTEST RESULTS ===")

    df = pd.read_csv(RESULTS_PATH)

    results = []

    for feature_set, group in df.groupby("feature_set"):
        weights = group["validation_rows"].to_numpy()

        # Pool absolute errors across all evaluated rows
        pooled_mae = np.average(
            group["mae"],
            weights=weights,
        )

        # Pool squared errors before taking the root
        pooled_rmse = np.sqrt(
            np.average(
                group["rmse"] ** 2,
                weights=weights,
            )
        )

        results.append({
            "feature_set": feature_set,
            "validation_rows": int(weights.sum()),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
        })

    summary = pd.DataFrame(results).sort_values(
        "pooled_mae"
    )

    print(summary.round(2).to_string(index=False))

    output = ROOT / "reports" / "pooled_backtest_summary.csv"

    summary.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
