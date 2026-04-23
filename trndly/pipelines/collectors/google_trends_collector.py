"""
Google Trends collector for trndly trend signals.

Fetches the last 7 days of daily search-interest data for every color,
category, and material in the feature contract and writes the result as
trend_signals.csv with a single `current` column.

This is the only column the model needs as input. The model's job is to
predict the best listing timeframe from what is trending RIGHT NOW.

The output CSV is validated against feature_contract.validate_trend_signals_frame
before being written, so it is a drop-in replacement for the synthetic
trend_signals.csv consumed by the training pipeline and scheduleServer.

Usage:
  python google_trends_collector.py
  python google_trends_collector.py --output-path path/to/trend_signals.csv
  python google_trends_collector.py --geo US --sleep 2.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    DEFAULT_MISSING_SCORE,
    validate_trend_signals_frame,
)

# --------------------------------------------------------------------------- #
# Search query strings                                                          #
# Maps each feature_value → the Google Trends keyword used to represent it.   #
# Append a fashion context word so results stay within the apparel domain.     #
# --------------------------------------------------------------------------- #
SEARCH_QUERIES: dict[str, dict[str, str]] = {
    "color": {
        "black": "black clothing",
        "white": "white fashion",
        "blue": "blue fashion",
        "red": "red fashion",
        "green": "green fashion",
        "beige": "beige fashion",
        "pink": "pink fashion",
        "gray": "gray clothing",
        "navy": "navy clothing",
        "brown": "brown clothing",
        "purple": "purple fashion",
    },
    "category": {
        "pants": "pants fashion",
        "shorts": "shorts fashion",
        "skirt": "skirt fashion",
        "dress": "dress fashion",
        "tops": "tops fashion",
        "outerwear": "outerwear fashion",
        "shoes": "shoes fashion",
        "accessories": "fashion accessories",
    },
    "material": {
        "cotton": "cotton clothing",
        "denim": "denim clothing",
        "linen": "linen clothing",
        "silk": "silk clothing",
        "wool": "wool clothing",
        "polyester": "polyester clothing",
        "leather": "leather clothing",
        "knit": "knit clothing",
    },
}

# --------------------------------------------------------------------------- #
# Time-window definition                                                        #
# (start_days_ago, end_days_ago) — both inclusive, relative to today.          #
# Only the `current` column is produced. The model takes current signals as    #
# input and predicts future listing timing — it does not need pre-computed     #
# future scores fed in.                                                         #
# --------------------------------------------------------------------------- #
WINDOW_BOUNDS: dict[str, tuple[int, int]] = {
    "current": (0, 6),
}

PYTRENDS_TIMEFRAME = "today 3-m"  # 90 days of daily data
MAX_RETRIES = 3
RETRY_BASE_SLEEP_SECS = 60  # doubles on each retry (60 → 120 → 240)


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "synthetic_data"
        / "trend_signals.csv"
    )
    parser = argparse.ArgumentParser(
        description="Collect Google Trends data and write trend_signals.csv."
    )
    parser.add_argument(
        "--output-path",
        default=str(default_output),
        help=(
            "Destination CSV path. Defaults to the synthetic_data directory so "
            "it directly replaces the synthetic trend signals used by training."
        ),
    )
    parser.add_argument(
        "--geo",
        default="US",
        help="Two-letter country code for Google Trends geo filter (default: US).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help=(
            "Seconds to sleep between requests to avoid rate-limiting "
            "(default: 2.0). Increase to 4–5 if you hit repeated 429s."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Core fetch + window computation                                               #
# --------------------------------------------------------------------------- #

def _fetch_interest_series(
    pytrends: TrendReq,
    keyword: str,
    geo: str,
    max_retries: int = MAX_RETRIES,
) -> pd.Series:
    """
    Fetch daily interest_over_time for a single keyword.
    Returns a Series indexed by date (values 0–100), or an empty Series on failure.
    Retries with exponential backoff on rate-limit (429) errors.
    """
    for attempt in range(max_retries):
        try:
            pytrends.build_payload([keyword], timeframe=PYTRENDS_TIMEFRAME, geo=geo)
            frame = pytrends.interest_over_time()
            if frame.empty or keyword not in frame.columns:
                return pd.Series(dtype=float)
            return frame[keyword].astype(float)
        except Exception as exc:
            error_text = str(exc).lower()
            if "429" in error_text or "too many requests" in error_text:
                wait_secs = RETRY_BASE_SLEEP_SECS * (2 ** attempt)
                print(
                    f"    Rate-limited on '{keyword}'. "
                    f"Sleeping {wait_secs}s then retry {attempt + 1}/{max_retries}..."
                )
                time.sleep(wait_secs)
            else:
                print(f"    Error fetching '{keyword}': {exc}. Using default scores.")
                return pd.Series(dtype=float)

    print(f"    Exhausted retries for '{keyword}'. Using default scores.")
    return pd.Series(dtype=float)


def _series_to_window_scores(series: pd.Series) -> dict[str, float]:
    """
    Slice a daily interest series into the five timeframe windows and compute
    the mean interest for each window, normalized from 0–100 to 0–1.
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    scores: dict[str, float] = {}

    for timeframe, bounds in WINDOW_BOUNDS.items():
        if bounds is None:
            scores[timeframe] = DEFAULT_MISSING_SCORE
            continue

        if series.empty:
            scores[timeframe] = DEFAULT_MISSING_SCORE
            continue

        start_days, end_days = bounds
        window_start = today - pd.Timedelta(days=end_days)
        window_end = today - pd.Timedelta(days=start_days)

        idx = series.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        mask = (idx >= window_start) & (idx <= window_end)
        window_values = series.values[mask].astype(float)

        if len(window_values) == 0 or float(np.max(window_values)) == 0.0:
            scores[timeframe] = DEFAULT_MISSING_SCORE
        else:
            scores[timeframe] = round(float(np.mean(window_values)) / 100.0, 6)

    return scores


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #

def collect_all_signals(
    pytrends: TrendReq,
    geo: str,
    sleep_secs: float,
) -> list[dict]:
    """
    Iterate over every (feature_type, feature_value) pair, fetch its Google
    Trends series, and return a list of rows ready for trend_signals.csv.
    """
    rows: list[dict] = []
    total_keywords = sum(len(v) for v in SEARCH_QUERIES.values())
    completed = 0

    for feature_type, query_map in SEARCH_QUERIES.items():
        print(f"\n[{feature_type}] — {len(query_map)} keywords")
        for feature_value, keyword in query_map.items():
            completed += 1
            print(f"  ({completed}/{total_keywords}) '{feature_value}' → \"{keyword}\"")

            series = _fetch_interest_series(pytrends=pytrends, keyword=keyword, geo=geo)
            scores = _series_to_window_scores(series)

            rows.append(
                {
                    "feature_type": feature_type,
                    "feature_value": feature_value,
                    **scores,
                }
            )

            time.sleep(sleep_secs)

    return rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_keywords = sum(len(v) for v in SEARCH_QUERIES.values())
    est_minutes = round((total_keywords * args.sleep) / 60, 1)
    print(
        f"Google Trends collector starting\n"
        f"  geo={args.geo}  sleep={args.sleep}s  keywords={total_keywords}\n"
        f"  estimated time: ~{est_minutes} min (plus any retry delays)\n"
        f"  output: {output_path}\n"
        f"  column written: current (7-day average)"
    )

    pytrends = TrendReq(hl="en-US", tz=360)
    rows = collect_all_signals(pytrends=pytrends, geo=args.geo, sleep_secs=args.sleep)

    raw_frame = pd.DataFrame(rows)
    validated = validate_trend_signals_frame(raw_frame)
    validated.to_csv(output_path, index=False)

    print(f"\nWrote {len(validated)} rows → {output_path}")

    fallback_count = int((validated["current"] <= DEFAULT_MISSING_SCORE + 0.001).sum())
    if fallback_count:
        print(f"\n{fallback_count} feature value(s) fell back to default score ({DEFAULT_MISSING_SCORE}) — no Google Trends data returned.")
    else:
        print("All current scores populated with real Google Trends data.")


if __name__ == "__main__":
    main()
