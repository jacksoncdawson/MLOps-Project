"""
Feature contract for the trndly timing model.

This module is the single source of truth for how raw trend data and raw
item data get turned into the numeric feature vector the model consumes, 
and for the label space the model predicts over. 

Every other piece of the training and serving pipeline (data prep, training,
inference, scoring APIs) imports from here so the shape of the data stays
consistent end to end.
"""

# imports 
from __future__ import annotations
import math
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd


# The five time windows the model can predict
# labels the model chooses between when answering "when will this item peak?".
TIMEFRAMES: list[str] = [
    "current",
    "next_week",
    "next_month",
    "three_months",
    "six_months",
]

# The three attributes of an item the model cares about (inputs)
FEATURE_TYPES: list[str] = ["color", "category", "material"]

# The fields a user-supplied item is expected to have (item name and three feature types)
USER_ITEM_FIELDS: list[str] = ["item_name", *FEATURE_TYPES]

# The name of the column in the training CSV that holds the correct answer (label)
TARGET_COLUMN_DEFAULT = "best_timeframe"

# The two identifier columns in the trend-signals CSV. Every row of that CSV.
TREND_SIGNAL_ID_COLUMNS = ["feature_type", "feature_value"]

# The full set of required columns in the trend-signals CSV: the two IDs
# above plus a `current` column holding today's trend score.
TREND_SIGNAL_COLUMNS: list[str] = [*TREND_SIGNAL_ID_COLUMNS, "current"]

# Fallback score used whenever a feature value is missing from the trend table
DEFAULT_MISSING_SCORE = 0.05

# The four "current trend" features computed from the Google Trends snapshot.
# These describe what is hot RIGHT NOW.
CURRENT_SIGNAL_COLUMNS: list[str] = [
    "color_current",
    "category_current",
    "material_current",
    "avg_current",
]

# Calendar offsets (in months) at which we sample the H&M-derived seasonality
# curve. Chosen to mirror TIMEFRAMES so each label has a corresponding feature.
SEASONAL_OFFSETS: list[int] = [0, 1, 2, 3, 6]

# Features derived from the historical H&M seasonality curve. They tell the
# model "given the current calendar month, what does the historical demand
# curve look like for this combo at +0, +1, +2, +3, +6 months ahead?".
SEASONAL_FEATURE_COLUMNS: list[str] = [
    *(f"season_plus_{k}" for k in SEASONAL_OFFSETS),
    "months_since_peak",
    "months_until_peak",
    "sin_month",
    "cos_month",
]

# Full feature vector handed to the model: current Google Trends signals
# concatenated with historical seasonality features.
FEATURE_VECTOR_COLUMNS: list[str] = [
    *CURRENT_SIGNAL_COLUMNS,
    *SEASONAL_FEATURE_COLUMNS,
]

# Schema for the seasonality_table.csv produced by hmn_seasonal_processor.py.
# One row per (color, category, material) combo holding the normalized 12-month
# curve plus how many transactions back it (used to gate trust during lookup).
SEASONALITY_ID_COLUMNS: list[str] = ["color", "category", "material"]
SEASONALITY_MONTH_COLUMNS: list[str] = [f"month_{m}" for m in range(1, 13)]
SEASONALITY_TABLE_COLUMNS: list[str] = [
    *SEASONALITY_ID_COLUMNS,
    *SEASONALITY_MONTH_COLUMNS,
    "n_observations",
]

# Minimum transactions backing a seasonality curve for it to be trusted.
# When a (color, category, material) triple falls below this, the lookup
# backs off progressively to (category, material) → (category,) → global.
SEASONALITY_MIN_OBSERVATIONS: int = 50

# Per-feature numeric ranges used when validating training data. sin/cos
# legitimately live in [-1, 1] so the universal [0, 1] clip from the
# previous version of this contract would have silently destroyed them.
FEATURE_VALUE_RANGES: dict[str, tuple[float, float]] = {
    **{c: (0.0, 1.0) for c in CURRENT_SIGNAL_COLUMNS},
    **{f"season_plus_{k}": (0.0, 1.0) for k in SEASONAL_OFFSETS},
    "months_since_peak": (0.0, 11.0),
    "months_until_peak": (0.0, 11.0),
    "sin_month": (-1.0, 1.0),
    "cos_month": (-1.0, 1.0),
}

# Fallback fill values for each feature when a row arrives with NaN. Most
# features fall back to DEFAULT_MISSING_SCORE; sin/cos fall back to 0.0
# because there is no meaningful "missing month".
FEATURE_FILL_VALUES: dict[str, float] = {
    **{c: DEFAULT_MISSING_SCORE for c in CURRENT_SIGNAL_COLUMNS},
    **{f"season_plus_{k}": DEFAULT_MISSING_SCORE for k in SEASONAL_OFFSETS},
    "months_since_peak": 0.0,
    "months_until_peak": 0.0,
    "sin_month": 0.0,
    "cos_month": 0.0,
}

# {feature_type: {feature_value: current_score}}
# A type alias describing the shape of our in-memory trend lookup table.
# Example:
#   {
#     "color":    {"red": 0.82, "blue": 0.31, ...},
#     "category": {"dress": 0.67, "jacket": 0.44, ...},
#     "material": {"denim": 0.51, "silk": 0.22, ...},
#   }
TrendLookup = dict[str, dict[str, float]]


# A 12-element numpy array indexed Jan..Dec, peak-normalized so max == 1.0.
SeasonalityCurve = np.ndarray

# Multi-resolution seasonality lookup. Keys are tuples of normalized strings
# at progressively coarser specificity so a lookup can degrade gracefully:
#   ("blue", "pants", "linen")   — full triple
#   ("pants", "linen")           — backoff: drop color
#   ("pants",)                   — backoff: category only
#   ()                           — global mean (always present)
# Values are (curve, n_observations) so callers can decide whether to trust
# a level or fall through to the next one.
SeasonalityTable = dict[tuple[str, ...], tuple[SeasonalityCurve, int]]


def normalize_token(value: object) -> str:
    """
        Takes any value (string, number, None, etc.), turns it into a string,
        trims surrounding whitespace, and lowercases it. 

        Args:
            value: Any value (string, number, None, etc.)

        Returns:
            A normalized string.
    """
    return str(value).strip().lower()



def validate_trend_signals_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
        Takes a raw pandas DataFrame loaded from the trend-signals CSV and returns
        a cleaned-up version we can safely use. Raises `ValueError` if the data is
        unusable (empty, missing columns, nothing left after filtering).

        Args:
            frame: A pandas DataFrame loaded from the trend-signals CSV.

        Returns:
            A cleaned-up pandas DataFrame.
    """
    # check if the frame is empty - rasie ValueError 
    if frame.empty:
        raise ValueError("Trend signals dataset is empty.")

    # Check that frame has required cols 
    missing_columns = [column for column in TREND_SIGNAL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Trend signals dataset is missing required columns: "
            f"{missing_columns}. Required: {TREND_SIGNAL_COLUMNS}."
        )

    # Keep only the columns we care about
    validated = frame[TREND_SIGNAL_COLUMNS].copy()

    # Normalize the two identifier columns
    validated["feature_type"] = validated["feature_type"].map(normalize_token)
    validated["feature_value"] = validated["feature_value"].map(normalize_token)

    # Drop any rows whose `feature_type` isn't one of the supported feature types
    validated = validated[validated["feature_type"].isin(FEATURE_TYPES)].copy()
    if validated.empty:
        # Everything got filtered out — now empty - value error 
        raise ValueError(
            f"No rows left after filtering to supported feature types: {FEATURE_TYPES}."
        )

    # Clean up the numeric `current` column - turns the column into floats
    # replaces anything unparseable with NaN (0.05) and clips to [0, 1]
    validated["current"] = (
        pd.to_numeric(validated["current"], errors="coerce")
        .fillna(DEFAULT_MISSING_SCORE)
        .clip(lower=0.0, upper=1.0)
    )

    # keep last occurrence of same (feature_type, feature_value) pair
    validated = validated.drop_duplicates(
        subset=["feature_type", "feature_value"],
        keep="last",
    )
    return validated


def load_trend_signals_frame(csv_path: str | Path) -> pd.DataFrame:
    """
        Reads a trend-signals CSV from disk and returns a validated DataFrame.
        Accepts either a plain string path or a `Path` object.

        Args:
            csv_path: A string path or `Path` object to the trend-signals CSV.

        Returns:
            A validated pandas DataFrame.
    """
    path = Path(csv_path).expanduser().resolve()
    # Load the CSV into a DataFrame (pandas infers types per column).
    frame = pd.read_csv(path)
    return validate_trend_signals_frame(frame)



def build_trend_lookup(frame: pd.DataFrame) -> TrendLookup:
    """
        Turns a validated DataFrame into the nested-dict `TrendLookup` structure
        defined above. This is the form the rest of the code uses to look up
        trend scores by (feature_type, feature_value).

        Args:
            frame: A validated pandas DataFrame.

        Returns:
            A nested-dict `TrendLookup` structure.
    """
    # validate the frame again
    validated = validate_trend_signals_frame(frame)

    # Start with one empty inner dict per supported feature type
    lookup: TrendLookup = {feature_type: {} for feature_type in FEATURE_TYPES}

    for row in validated.itertuples(index=False):
        # `getattr(row, "x")` pulls the value of column `x` off the tuple.
        feature_type = getattr(row, "feature_type")
        feature_value = getattr(row, "feature_value")
        # Store the score keyed by (feature_type, feature_value)
        lookup[feature_type][feature_value] = float(getattr(row, "current"))

    return lookup


def load_trend_lookup(csv_path: str | Path) -> TrendLookup:
    """
        Convenience wrapper: read a CSV from disk and return a ready-to-use lookup
        table in one call. Combines `load_trend_signals_frame` + `build_trend_lookup`.

    Args:
        csv_path: A string path or `Path` object to the trend-signals CSV.

    Returns:
        A ready-to-use lookup table.
    """
    return build_trend_lookup(load_trend_signals_frame(csv_path))


# --------------------------------------------------------------------------- #
# Seasonality table: load / validate / build / lookup                          #
# --------------------------------------------------------------------------- #


def validate_seasonality_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
        Validate and clean a raw seasonality DataFrame loaded from
        seasonality_table.csv. Returns the cleaned frame, or raises ValueError
        if the data is unusable.

        The on-disk schema is SEASONALITY_TABLE_COLUMNS:
            color, category, material, month_1..month_12, n_observations

        Each month_* value is interpreted as a peak-normalized share, i.e. the
        per-combo curve has max == 1.0 across the 12 calendar months.

        Args:
            frame: A pandas DataFrame loaded from seasonality_table.csv.

        Returns:
            A cleaned pandas DataFrame.
    """
    if frame.empty:
        raise ValueError("Seasonality dataset is empty.")

    missing_columns = [c for c in SEASONALITY_TABLE_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Seasonality dataset is missing required columns: "
            f"{missing_columns}. Required: {SEASONALITY_TABLE_COLUMNS}."
        )

    validated = frame[SEASONALITY_TABLE_COLUMNS].copy()

    for id_column in SEASONALITY_ID_COLUMNS:
        validated[id_column] = validated[id_column].map(normalize_token)

    for month_column in SEASONALITY_MONTH_COLUMNS:
        validated[month_column] = (
            pd.to_numeric(validated[month_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
        )

    validated["n_observations"] = (
        pd.to_numeric(validated["n_observations"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    # Drop combos whose curve is entirely zero — they carry no signal.
    curve_sums = validated[SEASONALITY_MONTH_COLUMNS].sum(axis=1)
    validated = validated[curve_sums > 0.0].copy()
    if validated.empty:
        raise ValueError("All seasonality rows had zero curves.")

    validated = validated.drop_duplicates(subset=SEASONALITY_ID_COLUMNS, keep="last")
    return validated


def load_seasonality_frame(csv_path: str | Path) -> pd.DataFrame:
    """
        Read seasonality_table.csv from disk and return a validated DataFrame.

        Args:
            csv_path: A string path or `Path` object to the seasonality CSV.

        Returns:
            A validated pandas DataFrame.
    """
    path = Path(csv_path).expanduser().resolve()
    frame = pd.read_csv(path)
    return validate_seasonality_frame(frame)


def _normalize_curve_to_peak(curve: np.ndarray) -> np.ndarray:
    """Divide a 12-element curve by its max so peak-month == 1.0. Zero curves
    pass through unchanged."""
    peak = float(np.max(curve))
    if peak <= 0.0:
        return curve.astype(float)
    return (curve / peak).astype(float)


def _aggregate_curves(
    curves: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Compute a weighted-average curve across multiple combos, then peak-normalize
    the result. `curves` has shape (n, 12); `weights` has shape (n,).
    Returns (aggregated_curve, total_weight).
    """
    total_weight = int(weights.sum())
    if total_weight <= 0 or len(curves) == 0:
        return np.zeros(12, dtype=float), 0
    weighted = (curves * weights[:, None]).sum(axis=0) / float(total_weight)
    return _normalize_curve_to_peak(weighted), total_weight


def build_seasonality_table(frame: pd.DataFrame) -> SeasonalityTable:
    """
        Turn a validated seasonality DataFrame into the multi-resolution
        SeasonalityTable used at lookup time.

        The returned dict always contains:
          - one entry per (color, category, material) triple
          - one entry per (category, material) pair, aggregated across colors
          - one entry per (category,), aggregated across colors and materials
          - the empty-tuple key (), which holds the global mean

        Curves at every level are re-peak-normalized so max == 1.0.

        Args:
            frame: A validated pandas DataFrame.

        Returns:
            A SeasonalityTable populated at every backoff level.
    """
    validated = validate_seasonality_frame(frame)
    table: SeasonalityTable = {}

    month_matrix = validated[SEASONALITY_MONTH_COLUMNS].to_numpy(dtype=float)
    obs_array = validated["n_observations"].to_numpy(dtype=int)

    # Full-triple level: store curves as-is (already peak-normalized upstream
    # but re-normalize defensively in case the upstream writer changed).
    for i, row in enumerate(validated.itertuples(index=False)):
        key = (row.color, row.category, row.material)
        table[key] = (_normalize_curve_to_peak(month_matrix[i]), int(obs_array[i]))

    # (category, material) level — average across colors weighted by n_obs.
    for (category, material), idx in validated.groupby(
        ["category", "material"]
    ).groups.items():
        positions = [validated.index.get_loc(i) for i in idx]
        curve, total = _aggregate_curves(
            curves=month_matrix[positions],
            weights=obs_array[positions],
        )
        table[(category, material)] = (curve, total)

    # (category,) level — average across colors and materials.
    for category, idx in validated.groupby("category").groups.items():
        positions = [validated.index.get_loc(i) for i in idx]
        curve, total = _aggregate_curves(
            curves=month_matrix[positions],
            weights=obs_array[positions],
        )
        table[(category,)] = (curve, total)

    # Global mean — always present so lookup never fails.
    global_curve, global_total = _aggregate_curves(
        curves=month_matrix,
        weights=obs_array,
    )
    table[()] = (global_curve, global_total)

    return table


def load_seasonality_table(csv_path: str | Path) -> SeasonalityTable:
    """
        Convenience wrapper: read seasonality_table.csv and return the lookup
        dict in one call.

        Args:
            csv_path: A string path or `Path` object to the seasonality CSV.

        Returns:
            A ready-to-use SeasonalityTable.
    """
    return build_seasonality_table(load_seasonality_frame(csv_path))


# A flat curve to use when even the global mean is missing (e.g. tests with a
# tiny synthetic table). All months equally likely → no seasonal signal.
_FLAT_CURVE: SeasonalityCurve = np.ones(12, dtype=float)


def _lookup_seasonality_curve(
    item: Mapping[str, object],
    table: SeasonalityTable,
) -> SeasonalityCurve:
    """
        Find the most specific seasonality curve we trust for this item. Tries
        keys in order from most to least specific and returns the first whose
        backing observation count meets SEASONALITY_MIN_OBSERVATIONS.

        Lookup order:
            (color, category, material) → (category, material) → (category,) → ()

        If nothing in the table qualifies (or the table is empty), returns a
        flat curve so downstream code never sees NaN.

        Args:
            item: dict-like with keys `color`, `category`, `material`.
            table: SeasonalityTable produced by build_seasonality_table.

        Returns:
            A 12-element numpy array indexed Jan..Dec.
    """
    color = normalize_token(item.get("color", ""))
    category = normalize_token(item.get("category", ""))
    material = normalize_token(item.get("material", ""))

    candidate_keys: list[tuple[str, ...]] = [
        (color, category, material),
        (category, material),
        (category,),
        (),
    ]

    for key in candidate_keys:
        entry = table.get(key)
        if entry is None:
            continue
        curve, n_obs = entry
        if n_obs >= SEASONALITY_MIN_OBSERVATIONS:
            return curve

    # Nothing crossed the trust threshold; fall back to whatever the global
    # mean has, even if undertrained, before the truly-flat last resort.
    fallback = table.get(())
    if fallback is not None and float(np.max(fallback[0])) > 0.0:
        return fallback[0]
    return _FLAT_CURVE


def compute_seasonal_features(
    item: Mapping[str, object],
    reference_month: int,
    seasonality_table: SeasonalityTable,
) -> dict[str, float]:
    """
        Compute the SEASONAL_FEATURE_COLUMNS values for one (item, reference_month).

        Uses the historical seasonality curve to express:
          * the curve's value at +0/+1/+2/+3/+6 months ahead of `reference_month`
          * how far back the historical peak was, and how far ahead it is
          * sin/cos of the reference month so the model sees Dec ≈ Jan

        Args:
            item: dict-like with keys `color`, `category`, `material`.
            reference_month: Calendar month (1..12) treated as "today" for this row.
            seasonality_table: SeasonalityTable produced by build_seasonality_table.

        Returns:
            dict with keys exactly SEASONAL_FEATURE_COLUMNS.
    """
    if not 1 <= int(reference_month) <= 12:
        raise ValueError(
            f"reference_month must be in 1..12, got {reference_month!r}."
        )
    ref_month = int(reference_month)

    curve = _lookup_seasonality_curve(item=item, table=seasonality_table)

    features: dict[str, float] = {}
    for offset in SEASONAL_OFFSETS:
        # Curve is 0-indexed Jan..Dec; reference_month is 1-indexed.
        month_index = (ref_month - 1 + offset) % 12
        features[f"season_plus_{offset}"] = round(float(curve[month_index]), 6)

    peak_month_index = int(np.argmax(curve))
    peak_month = peak_month_index + 1  # 1..12
    features["months_until_peak"] = float((peak_month - ref_month) % 12)
    features["months_since_peak"] = float((ref_month - peak_month) % 12)

    angle = 2.0 * math.pi * (ref_month / 12.0)
    features["sin_month"] = round(math.sin(angle), 6)
    features["cos_month"] = round(math.cos(angle), 6)

    return features


def _lookup_current_score(feature_type: str,
                          feature_value: object,
                          lookup: TrendLookup) -> float:
    """
    Private helper. Given a feature type ("color"), a feature value ("red"),
    and a lookup table, returns the current trend score. Falls back to
    `DEFAULT_MISSING_SCORE` if either the type or the value isn't in the table.

    Args:
        feature_type: A string representing the feature type.
        feature_value: A string representing the feature value.
        lookup: A TrendLookup dictionary.

    Returns:
        A float representing the current trend score.
    """
    # Normalize both inputs
    normalized_type = normalize_token(feature_type)
    normalized_value = normalize_token(feature_value)

    feature_bucket = lookup.get(normalized_type, {})
    # return the score, or the default if value isn't on record.
    return float(feature_bucket.get(normalized_value, DEFAULT_MISSING_SCORE))



def compute_feature_scores(item: Mapping[str, object],
                           lookup: TrendLookup) -> dict[str, float]:
    """
        Returns the current trend score for each feature type and their average.
        Keys: color_current, category_current, material_current, avg_current.

        Args:
            item: A dictionary-like object with keys `color`, `category`, `material`.
            lookup: A TrendLookup dictionary.

        Returns:
            A dictionary with keys `color_current`, `category_current`, `material_current`, `avg_current`.
    """
    scores: dict[str, float] = {}
    total = 0.0
    # Walk through the three feature types in the fixed order
    for feature_type in FEATURE_TYPES:
        score = _lookup_current_score(
            feature_type=feature_type,
            feature_value=item.get(feature_type, ""),
            lookup=lookup,
        )
        # Store the individual score under e.g. "color_current", rounded 
        scores[f"{feature_type}_current"] = round(score, 6)
        total += score
    # The fourth feature: the arithmetic mean of the three scores above.
    scores["avg_current"] = round(total / len(FEATURE_TYPES), 6)
    return scores


def item_to_feature_row(
    item: Mapping[str, object],
    lookup: TrendLookup,
    *,
    reference_month: int,
    seasonality_table: SeasonalityTable,
) -> dict[str, float]:
    """
    Build a complete feature row for one item by combining current Google
    Trends scores with H&M-derived seasonality features.

    The returned dict has exactly the keys in FEATURE_VECTOR_COLUMNS.

    Args:
        item: dict-like with keys `color`, `category`, `material`.
        lookup: TrendLookup with today's Google Trends scores.
        reference_month: Calendar month (1..12) treated as "today" for this row.
            At training time this varies across the 12 months for each combo;
            at inference time pass the actual current month.
        seasonality_table: SeasonalityTable produced from the H&M seasonality CSV.

    Returns:
        dict with keys exactly FEATURE_VECTOR_COLUMNS.
    """
    return {
        **compute_feature_scores(item=item, lookup=lookup),
        **compute_seasonal_features(
            item=item,
            reference_month=reference_month,
            seasonality_table=seasonality_table,
        ),
    }



def build_feature_frame(
    items: Sequence[Mapping[str, object]],
    lookup: TrendLookup,
    *,
    reference_month: int,
    seasonality_table: SeasonalityTable,
) -> pd.DataFrame:
    """
        Turn a list of items into a DataFrame ready to feed the model. Each row
        is one item's feature vector; columns are exactly FEATURE_VECTOR_COLUMNS.

        Note: every item shares the same `reference_month`. If you need per-item
        reference months (e.g. multiple training rows for the same combo across
        different months), call `item_to_feature_row` directly per row.

        Args:
            items: A list of dict-like objects with keys `color`, `category`, `material`.
            lookup: TrendLookup with today's Google Trends scores.
            reference_month: Calendar month (1..12) treated as "today" for every item.
            seasonality_table: SeasonalityTable produced from the H&M seasonality CSV.
    """
    rows = [
        item_to_feature_row(
            item=item,
            lookup=lookup,
            reference_month=reference_month,
            seasonality_table=seasonality_table,
        )
        for item in items
    ]
    if not rows:
        return pd.DataFrame(columns=FEATURE_VECTOR_COLUMNS)
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=FEATURE_VECTOR_COLUMNS, fill_value=DEFAULT_MISSING_SCORE)


def prepare_training_frame(frame: pd.DataFrame,
                           target_column: str = TARGET_COLUMN_DEFAULT) -> pd.DataFrame:
    """
        Validates and cleans a training DataFrame (features + label column) before
        it's handed off to the model-training code. `target_column` defaults to
        "best_timeframe" but callers can override it if their CSV uses a different
        label column name.

        Args:
            frame: A pandas DataFrame containing the training data.
            target_column: The name of the label column in the training data.

        Returns:
            A cleaned pandas DataFrame ready for model training.
    """
    # The training CSV must contain the four feature columns AND the label col
    required_columns = [*FEATURE_VECTOR_COLUMNS, target_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{missing_columns}. Required feature columns are {FEATURE_VECTOR_COLUMNS}."
        )

    # Keep only the columns we need
    prepared = frame[required_columns].copy()
    # Clean each feature column using its declared range and fill value.
    # Different features have different valid ranges (notably sin/cos which
    # live in [-1, 1]) so a universal clip would silently corrupt them.
    for feature_name in FEATURE_VECTOR_COLUMNS:
        low, high = FEATURE_VALUE_RANGES[feature_name]
        fill = FEATURE_FILL_VALUES[feature_name]
        prepared[feature_name] = (
            pd.to_numeric(prepared[feature_name], errors="coerce")
            .fillna(fill)
            .clip(lower=low, upper=high)
        )

    # Normalize the label column
    prepared[target_column] = prepared[target_column].map(normalize_token)
    # Drop any row whose label isn't one of the five supported timeframes.
    prepared = prepared[prepared[target_column].isin(TIMEFRAMES)].copy()
    if prepared.empty:
        # Every row got filtered out — error 
        raise ValueError(
            "No valid training rows after filtering labels to supported timeframes: "
            f"{TIMEFRAMES}."
        )
    # Return the cleaned, label-filtered frame ready for model training.
    return prepared
