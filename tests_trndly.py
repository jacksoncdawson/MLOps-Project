import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "trndly"))

from pipelines.training.feature_contract import (
    DEFAULT_MISSING_SCORE,
    FEATURE_VECTOR_COLUMNS,
    normalize_token,
    validate_trend_signals_frame,
    build_trend_lookup,
    compute_feature_scores,
    prepare_training_frame,
)


def _minimal_trend_signals() -> pd.DataFrame:
    """A small valid trend signals DataFrame for reuse across tests."""
    return pd.DataFrame(
        [
            {"feature_type": "color", "feature_value": "red", "current": 0.8},
            {"feature_type": "category", "feature_value": "tops", "current": 0.6},
            {"feature_type": "material", "feature_value": "cotton", "current": 0.4},
        ]
    )


# 1. normalize_token strips whitespace and lowercases
def test_normalize_token():
    assert normalize_token("  RED  ") == "red"
    assert normalize_token("Cotton") == "cotton"
    assert normalize_token(42) == "42"


# 2. validate_trend_signals_frame raises on an empty DataFrame - shouldn't work on an empty dataframe
def test_validate_trend_signals_frame_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_trend_signals_frame(pd.DataFrame())


# 3. validate_trend_signals_frame raises when required columns are missing
# missing the current columnn which shows the current trend score 
def test_validate_trend_signals_frame_raises_on_missing_columns():
    bad = pd.DataFrame([{"feature_type": "color", "feature_value": "red"}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_trend_signals_frame(bad)


# 5. compute_feature_scores always returns all four expected keys with values in [0, 1]
def test_compute_feature_scores_output_shape_and_range():
    lookup = build_trend_lookup(_minimal_trend_signals())
    item = {"color": "red", "category": "tops", "material": "cotton"}
    scores = compute_feature_scores(item, lookup)

    assert set(scores.keys()) == set(FEATURE_VECTOR_COLUMNS)
    for key, value in scores.items():
        assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"
