"""Pytest suite implementing 9 tests from trndly/tests/test_ideas.md.

Coverage:
    Model inference        -> tests 1, 2, 3
    Feature engineering    -> tests 4, 5
    Data collection        -> tests 6, 7
    Data cleaning          -> tests 8, 9

Tests are resilient to partially-built features: if a dependency (e.g., a live
scraper, a trained MLflow model) is not yet implemented, the test is marked
``xfail``/``skip`` with a clear reason so CI keeps green while still documenting
intent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable so ``pipelines.training...`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    DEFAULT_MISSING_SCORE,
    FEATURE_VECTOR_COLUMNS,
    TIMEFRAMES,
    TREND_SIGNAL_COLUMNS,
    build_feature_frame,
    load_trend_lookup,
    normalize_token,
    prepare_training_frame,
    validate_trend_signals_frame,
)

SYNTHETIC_DIR = PROJECT_ROOT / "pipelines" / "training" / "synthetic_data"
TREND_SIGNALS_CSV = SYNTHETIC_DIR / "trend_signals.csv"
TRAIN_CSV = SYNTHETIC_DIR / "train.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trend_lookup():
    if not TREND_SIGNALS_CSV.exists():
        pytest.skip(f"Trend signals CSV missing at {TREND_SIGNALS_CSV}")
    return load_trend_lookup(TREND_SIGNALS_CSV)


@pytest.fixture(scope="module")
def tiny_model(trend_lookup):
    """Train a small RF on the synthetic train split as a stand-in for the
    registered MLflow champion model. Keeps tests hermetic (no network)."""
    if not TRAIN_CSV.exists():
        pytest.skip(f"Training CSV missing at {TRAIN_CSV}")

    from sklearn.ensemble import RandomForestClassifier

    frame = pd.read_csv(TRAIN_CSV)
    prepared = prepare_training_frame(frame)
    x = prepared[FEATURE_VECTOR_COLUMNS]
    y = prepared["best_timeframe"]

    model = RandomForestClassifier(n_estimators=25, random_state=0)
    model.fit(x, y)
    return model


@pytest.fixture()
def sample_item():
    return {
        "item_name": "Linen Summer Dress",
        "color": "white",
        "category": "dress",
        "material": "linen",
    }


# ---------------------------------------------------------------------------
# 1. Model inference
# ---------------------------------------------------------------------------
def test_model_can_make_inference(tiny_model, trend_lookup, sample_item):
    """Test 1: the trend prediction model produces a prediction without error."""
    frame = build_feature_frame([sample_item], trend_lookup)
    predictions = tiny_model.predict(frame)

    assert len(predictions) == 1
    assert predictions[0] is not None


def test_inference_is_in_expected_range(tiny_model, trend_lookup, sample_item):
    """Test 2: predicted label is one of the documented timeframes."""
    frame = build_feature_frame([sample_item], trend_lookup)
    prediction = str(tiny_model.predict(frame)[0])

    assert prediction in TIMEFRAMES, (
        f"Model produced '{prediction}', which is outside the allowed set {TIMEFRAMES}."
    )


@pytest.mark.xfail(
    reason=(
        "Trend-monotonicity regression check is not yet implemented; the "
        "comparison utility `detect_downtrend` will live in pipelines/monitoring."
    ),
    strict=False,
)
def test_inference_trend_is_not_going_down(tiny_model, trend_lookup):
    """Test 3: when we re-score yesterday's batch against today's model, the
    average alignment score for the predicted timeframe should not drop."""
    from pipelines.monitoring import detect_downtrend  # type: ignore[import-not-found]

    items = [
        {"item_name": "Item A", "color": "black", "category": "tops", "material": "cotton"},
        {"item_name": "Item B", "color": "red",   "category": "dress", "material": "silk"},
    ]
    frame = build_feature_frame(items, trend_lookup)
    preds_today = tiny_model.predict(frame)

    # Placeholder "yesterday" scores - in real use these come from the monitoring store.
    yesterday_scores = np.full(len(items), 0.5)
    today_scores = frame[[f"avg_{tf}" for tf in TIMEFRAMES]].max(axis=1).to_numpy()

    assert not detect_downtrend(yesterday_scores, today_scores, preds_today)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def test_feature_creation_pipeline_produces_contract_columns(trend_lookup, sample_item):
    """Test 4: `build_feature_frame` emits the full feature vector contract."""
    frame = build_feature_frame([sample_item], trend_lookup)

    assert list(frame.columns) == FEATURE_VECTOR_COLUMNS
    assert len(frame) == 1
    assert frame.notna().all().all(), "Feature frame should never contain NaNs."
    assert ((frame >= 0.0) & (frame <= 1.0)).all().all(), (
        "All feature values should be clipped into [0, 1]."
    )


def test_nulls_are_handled_safely_during_training_prep():
    """Test 5: `prepare_training_frame` replaces NaN feature values with the
    documented default instead of raising or propagating nulls."""
    base = {col: 0.5 for col in FEATURE_VECTOR_COLUMNS}
    base["best_timeframe"] = "next_week"

    rows = [base.copy(), base.copy()]
    rows[0][FEATURE_VECTOR_COLUMNS[0]] = np.nan
    rows[1][FEATURE_VECTOR_COLUMNS[-1]] = None

    prepared = prepare_training_frame(pd.DataFrame(rows))

    assert prepared[FEATURE_VECTOR_COLUMNS].notna().all().all()
    assert prepared[FEATURE_VECTOR_COLUMNS[0]].iloc[0] == pytest.approx(DEFAULT_MISSING_SCORE)
    assert prepared[FEATURE_VECTOR_COLUMNS[-1]].iloc[1] == pytest.approx(DEFAULT_MISSING_SCORE)


# ---------------------------------------------------------------------------
# 3. Data collection
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason=(
        "Live scraper module (scripts/scrapers/run_scrapers.py) has not been "
        "written yet; this test will start passing once the scraper exposes "
        "`fetch_latest()` and returns a non-empty DataFrame."
    ),
    strict=False,
)
def test_scrapers_are_running_and_returning_data():
    """Test 6: scrapers run end-to-end without being blocked and return rows."""
    from scripts.scrapers.run_scrapers import fetch_latest  # type: ignore[import-not-found]

    result = fetch_latest(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert {"source", "fetched_at"}.issubset(result.columns)


def test_historical_data_is_in_right_form_for_training():
    """Test 7: the checked-in historical/synthetic training CSV still matches
    the feature contract and contains every supported timeframe label."""
    if not TRAIN_CSV.exists():
        pytest.skip(f"Historical training CSV missing at {TRAIN_CSV}")

    frame = pd.read_csv(TRAIN_CSV)

    missing = [col for col in FEATURE_VECTOR_COLUMNS if col not in frame.columns]
    assert not missing, f"Historical data missing feature columns: {missing}"
    assert "best_timeframe" in frame.columns
    assert set(frame["best_timeframe"]).issubset(set(TIMEFRAMES))
    assert len(frame) > 0


# ---------------------------------------------------------------------------
# 4. Data cleaning
# ---------------------------------------------------------------------------
def test_normalize_token_trims_and_lowercases():
    """Test 8: `normalize_token` standardizes user-supplied strings."""
    assert normalize_token("  Navy  ") == "navy"
    assert normalize_token("COTTON") == "cotton"
    assert normalize_token(None) == "none"  # stringified, then lowercased
    assert normalize_token(42) == "42"


def test_trend_signals_validation_rejects_bad_inputs():
    """Test 9: empty frames raise, missing columns raise, and extra rows with
    out-of-range scores are clipped into [0, 1]."""
    with pytest.raises(ValueError):
        validate_trend_signals_frame(pd.DataFrame())

    incomplete = pd.DataFrame([{"feature_type": "color", "feature_value": "navy"}])
    with pytest.raises(ValueError):
        validate_trend_signals_frame(incomplete)

    noisy = pd.DataFrame(
        [
            {
                "feature_type": "color",
                "feature_value": "navy",
                "current": 5.0,      # above 1.0
                "next_week": -0.3,   # below 0.0
                "next_month": 0.4,
                "three_months": 0.6,
                "six_months": 0.9,
            }
        ]
    )
    cleaned = validate_trend_signals_frame(noisy)
    assert list(cleaned.columns) == TREND_SIGNAL_COLUMNS
    assert cleaned["current"].iloc[0] == pytest.approx(1.0)
    assert cleaned["next_week"].iloc[0] == pytest.approx(0.0)
