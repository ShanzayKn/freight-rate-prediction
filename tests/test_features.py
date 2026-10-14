
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow tests to import the current src module.
# We will replace this with proper package installation later.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import prepare_features, select_features


@pytest.fixture
def sample_loads():
    """Create a small, controlled freight dataset."""
    return pd.DataFrame({
        "pickup": ["Dallas", "Chicago"],
        "delivery": ["Atlanta", "Boston"],
        "distance": [800.0, 1200.0],
        "equipment": ["Dry Van", "Reefer"],
        "weight": [-32000.0, np.nan],
        "date": ["2025-07-01", "2025-07-02"],
    })


def test_negative_weight_correction(sample_loads):
    result = prepare_features(sample_loads)

    assert result.loc[0, "weight"] == 32000.0


def test_missing_weight_preserved(sample_loads):
    result = prepare_features(sample_loads)

    assert pd.isna(result.loc[1, "weight"])


def test_date_features(sample_loads):
    result = prepare_features(sample_loads)

    assert result.loc[0, "month"] == 7
    assert result.loc[0, "day_of_week"] == 1
    assert result.loc[0, "day_of_year"] == 182


def test_basic_feature_selection(sample_loads):
    result = select_features(sample_loads, "basic")

    assert "distance" in result.columns
    assert "month" in result.columns
    assert "date" not in result.columns
    assert "posted_rate" not in result.columns


def test_missing_required_column(sample_loads):
    broken = sample_loads.drop(columns=["distance"])

    with pytest.raises(ValueError, match="distance"):
        prepare_features(broken)


def test_unknown_feature_set(sample_loads):
    with pytest.raises(ValueError, match="Unknown feature set"):
        select_features(sample_loads, "invalid")


def test_original_dataframe_unchanged(sample_loads):
    original = sample_loads.copy(deep=True)

    prepare_features(sample_loads)

    pd.testing.assert_frame_equal(sample_loads, original)


def test_categorical_missing_values(sample_loads):
    sample_loads.loc[0, "equipment"] = None

    result = prepare_features(sample_loads)

    assert result.loc[0, "equipment"] == "Unknown"
