
import json
import platform
import sys

from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy
import pandas
import sklearn


ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_DIR = ROOT / "reports" / "experiments"


def save_experiment(
    experiment_name,
    model_config,
    feature_set,
    metrics,
    train_rows,
    validation_rows,
    train_period,
    validation_period,
):
    """
    Save reproducible experiment metadata as JSON.
    """

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    record = {
        "experiment_name": experiment_name,
        "timestamp_utc": timestamp,
        "model": "CatBoostRegressor",
        "feature_set": feature_set,
        "model_config": model_config,
        "metrics": metrics,
        "dataset": {
            "training_rows": train_rows,
            "validation_rows": validation_rows,
            "training_period": train_period,
            "validation_period": validation_period,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "catboost": catboost.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    filename = f"{experiment_name}_{timestamp}.json"

    output_path = EXPERIMENT_DIR / filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=4,
        )

    return output_path
