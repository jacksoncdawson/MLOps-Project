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
from pathlib import Path
from typing import Mapping, Sequence
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

# The model takes one current trend score per feature type plus their average.
# Inputs are only what is known right now — the model predicts future timing.
FEATURE_VECTOR_COLUMNS: list[str] = [
    "color_current",
    "category_current",
    "material_current",
    "avg_current",
]

# {feature_type: {feature_value: current_score}}
# A type alias describing the shape of our in-memory trend lookup table.
# Example:
#   {
#     "color":    {"red": 0.82, "blue": 0.31, ...},
#     "category": {"dress": 0.67, "jacket": 0.44, ...},
#     "material": {"denim": 0.51, "silk": 0.22, ...},
#   }
TrendLookup = dict[str, dict[str, float]]


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


def item_to_feature_row(item: Mapping[str, object],
                        lookup: TrendLookup) -> dict[str, float]:
    """
        Thin wrapper over `compute_feature_scores`. Exists so that if we ever want
        to add extra engineered features later, callers already go through this
        "item -> feature row" function.

        Args:
            item: A dictionary-like object with keys `color`, `category`, `material`.
            lookup: A TrendLookup dictionary.

        Returns:
            A dictionary with keys `color_current`, `category_current`, `material_current`, `avg_current`.
    """
    return compute_feature_scores(item=item, lookup=lookup)



def build_feature_frame(items: Sequence[Mapping[str, object]],
                        lookup: TrendLookup) -> pd.DataFrame:
    """
        Turns a list of items into a DataFrame ready to feed the model. Each row
        is one item's feature vector; columns are exactly `FEATURE_VECTOR_COLUMNS`.

        Args:
            items: A list of dictionary-like objects with keys `color`, `category`, `material`.
            lookup: A TrendLookup dictionary.
        """
    # compute the feature row dict for each item.
    rows = [item_to_feature_row(item=item, lookup=lookup) for item in items]
    if not rows:
        return pd.DataFrame(columns=FEATURE_VECTOR_COLUMNS)
    # Build the DataFrame from the list of dicts
    frame = pd.DataFrame(rows)
    # ensure correct format
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
    # Clean each feature column
    for feature_name in FEATURE_VECTOR_COLUMNS:
        prepared[feature_name] = (
            pd.to_numeric(prepared[feature_name], errors="coerce")
            .fillna(DEFAULT_MISSING_SCORE)
            .clip(lower=0.0, upper=1.0)
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
