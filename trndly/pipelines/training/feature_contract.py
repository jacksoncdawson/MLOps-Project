from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

TIMEFRAMES: list[str] = [
    "current",
    "next_week",
    "next_month",
    "three_months",
    "six_months",
]
FEATURE_TYPES: list[str] = ["color", "category", "material"]
USER_ITEM_FIELDS: list[str] = ["item_name", *FEATURE_TYPES]
TARGET_COLUMN_DEFAULT = "best_timeframe"
TREND_SIGNAL_ID_COLUMNS = ["feature_type", "feature_value"]
TREND_SIGNAL_COLUMNS: list[str] = [*TREND_SIGNAL_ID_COLUMNS, *TIMEFRAMES]
DEFAULT_MISSING_SCORE = 0.05

FEATURE_VECTOR_COLUMNS: list[str] = [
    *(f"{feature_type}_{timeframe}" for feature_type in FEATURE_TYPES for timeframe in TIMEFRAMES),
    *(f"avg_{timeframe}" for timeframe in TIMEFRAMES),
]

TrendLookup = dict[str, dict[str, dict[str, float]]]


def normalize_token(value: object) -> str:
    return str(value).strip().lower()


def _default_timeframe_scores() -> dict[str, float]:
    return {timeframe: DEFAULT_MISSING_SCORE for timeframe in TIMEFRAMES}


def validate_trend_signals_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Trend signals dataset is empty.")

    missing_columns = [column for column in TREND_SIGNAL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Trend signals dataset is missing required columns: "
            f"{missing_columns}. Required: {TREND_SIGNAL_COLUMNS}."
        )

    validated = frame[TREND_SIGNAL_COLUMNS].copy()
    validated["feature_type"] = validated["feature_type"].map(normalize_token)
    validated["feature_value"] = validated["feature_value"].map(normalize_token)

    validated = validated[validated["feature_type"].isin(FEATURE_TYPES)].copy()
    if validated.empty:
        raise ValueError(
            f"No rows left after filtering to supported feature types: {FEATURE_TYPES}."
        )

    for timeframe in TIMEFRAMES:
        validated[timeframe] = (
            pd.to_numeric(validated[timeframe], errors="coerce")
            .fillna(DEFAULT_MISSING_SCORE)
            .clip(lower=0.0, upper=1.0)
        )

    validated = validated.drop_duplicates(
        subset=["feature_type", "feature_value"],
        keep="last",
    )
    return validated


def load_trend_signals_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    frame = pd.read_csv(path)
    return validate_trend_signals_frame(frame)


def build_trend_lookup(frame: pd.DataFrame) -> TrendLookup:
    validated = validate_trend_signals_frame(frame)
    lookup: TrendLookup = {feature_type: {} for feature_type in FEATURE_TYPES}

    for row in validated.itertuples(index=False):
        feature_type = getattr(row, "feature_type")
        feature_value = getattr(row, "feature_value")
        lookup[feature_type][feature_value] = {
            timeframe: float(getattr(row, timeframe)) for timeframe in TIMEFRAMES
        }

    return lookup


def load_trend_lookup(csv_path: str | Path) -> TrendLookup:
    return build_trend_lookup(load_trend_signals_frame(csv_path))


def _lookup_timeframe_scores(
    feature_type: str,
    feature_value: object,
    lookup: TrendLookup,
) -> dict[str, float]:
    normalized_type = normalize_token(feature_type)
    normalized_value = normalize_token(feature_value)

    feature_bucket = lookup.get(normalized_type, {})
    timeframe_scores = feature_bucket.get(normalized_value)
    if timeframe_scores is None:
        return _default_timeframe_scores()
    return {timeframe: float(timeframe_scores.get(timeframe, DEFAULT_MISSING_SCORE)) for timeframe in TIMEFRAMES}


def compute_alignment_scores(
    item: Mapping[str, object],
    lookup: TrendLookup,
) -> dict[str, float]:
    totals = {timeframe: 0.0 for timeframe in TIMEFRAMES}

    for feature_type in FEATURE_TYPES:
        feature_scores = _lookup_timeframe_scores(
            feature_type=feature_type,
            feature_value=item.get(feature_type, ""),
            lookup=lookup,
        )
        for timeframe in TIMEFRAMES:
            totals[timeframe] += feature_scores[timeframe]

    divisor = float(len(FEATURE_TYPES))
    return {timeframe: round(totals[timeframe] / divisor, 6) for timeframe in TIMEFRAMES}


def choose_best_timeframe(alignment_scores: Mapping[str, float]) -> str:
    best_timeframe = TIMEFRAMES[0]
    best_score = float(alignment_scores.get(best_timeframe, 0.0))

    for timeframe in TIMEFRAMES[1:]:
        score = float(alignment_scores.get(timeframe, 0.0))
        if score > best_score:
            best_timeframe = timeframe
            best_score = score

    return best_timeframe


def item_to_feature_row(
    item: Mapping[str, object],
    lookup: TrendLookup,
) -> dict[str, float]:
    row: dict[str, float] = {}

    for feature_type in FEATURE_TYPES:
        feature_scores = _lookup_timeframe_scores(
            feature_type=feature_type,
            feature_value=item.get(feature_type, ""),
            lookup=lookup,
        )
        for timeframe in TIMEFRAMES:
            row[f"{feature_type}_{timeframe}"] = float(feature_scores[timeframe])

    alignment_scores = compute_alignment_scores(item=item, lookup=lookup)
    for timeframe in TIMEFRAMES:
        row[f"avg_{timeframe}"] = float(alignment_scores[timeframe])

    return row


def build_feature_frame(items: Sequence[Mapping[str, object]], lookup: TrendLookup) -> pd.DataFrame:
    rows = [item_to_feature_row(item=item, lookup=lookup) for item in items]
    if not rows:
        return pd.DataFrame(columns=FEATURE_VECTOR_COLUMNS)

    frame = pd.DataFrame(rows)
    return frame.reindex(columns=FEATURE_VECTOR_COLUMNS, fill_value=DEFAULT_MISSING_SCORE)


def prepare_training_frame(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN_DEFAULT,
) -> pd.DataFrame:
    required_columns = [*FEATURE_VECTOR_COLUMNS, target_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{missing_columns}. Required feature columns are {FEATURE_VECTOR_COLUMNS}."
        )

    prepared = frame[required_columns].copy()
    for feature_name in FEATURE_VECTOR_COLUMNS:
        prepared[feature_name] = (
            pd.to_numeric(prepared[feature_name], errors="coerce")
            .fillna(DEFAULT_MISSING_SCORE)
            .clip(lower=0.0, upper=1.0)
        )

    prepared[target_column] = prepared[target_column].map(normalize_token)
    prepared = prepared[prepared[target_column].isin(TIMEFRAMES)].copy()
    if prepared.empty:
        raise ValueError(
            "No valid training rows after filtering labels to supported timeframes: "
            f"{TIMEFRAMES}."
        )

    return prepared
