"""
Combine per-source trend signal files into a single trend_signals.csv.

Each data source (google_trends_collector.py, hollister_scraper.py,
gap_scraper.py, ...) writes its own output file named
`trend_signals_<source>.csv` in the data/trend_signals/ subfolder. This
script reads every such file, merges the per-(feature_type, feature_value)
scores across sources, and writes the canonical `trend_signals.csv` (in
the parent data/ folder) that the training / serving pipeline actually
consumes.

MERGE STRATEGY
---------------
For each (feature_type, feature_value) pair:
  1. Collect every retailer's `current` score for that pair.
  2. Drop any score equal to DEFAULT_MISSING_SCORE — those represent
     "retailer didn't see this value" and just dilute real signal.
  3. If nothing real is left, fall back to DEFAULT_MISSING_SCORE.
  4. Otherwise take a (optionally weighted) mean of the surviving scores.

WHY AVERAGE (NOT MAX)?
----------------------
A value that only one retailer cares about shouldn't dominate the combined
score — we want something broadly popular to rise. But we also don't want
a retailer that simply didn't stock the value to push the score toward zero,
so we treat "missing" as "no information" instead of "score = 0".

Usage:
  # Combine every trend_signals_*.csv that exists in the default directory:
  python combine_trend_signals.py

  # Explicit input files, custom output:
  python combine_trend_signals.py \
      --input trend_signals_hollister.csv \
      --input trend_signals_pacsun.csv \
      --output-path trend_signals.csv

  # Weight retailers differently (same order as --input):
  python combine_trend_signals.py \
      --input trend_signals_hollister.csv --weight 1.0 \
      --input trend_signals_pacsun.csv    --weight 2.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    DEFAULT_MISSING_SCORE,
    TREND_SIGNAL_COLUMNS,
    validate_trend_signals_frame,
)
from pipelines.training.paths import (  # noqa: E402
    TREND_SIGNALS_CSV as DEFAULT_OUTPUT_PATH,
    TREND_SIGNALS_DIR as DEFAULT_SIGNALS_DIR,
    TREND_SIGNALS_GLOB as RETAILER_FILE_GLOB,
)


# --------------------------------------------------------------------------- #
# Loading                                                                       #
# --------------------------------------------------------------------------- #

def discover_retailer_files(
    signals_dir: Path,
    exclude: Path | None = None,
) -> list[Path]:
    """
    Find all per-retailer trend signal files in signals_dir.

    We exclude `exclude` (typically the canonical combined output) so we
    never feed the previous combined file back into itself.
    """
    found = sorted(signals_dir.glob(RETAILER_FILE_GLOB))
    if exclude is not None:
        exclude_resolved = exclude.resolve()
        found = [p for p in found if p.resolve() != exclude_resolved]
    return found


def load_retailer_signals(path: Path) -> pd.DataFrame:
    """
    Read a per-retailer trend signals CSV and validate it matches the
    feature contract. Adds a `source` column for debugging.
    """
    frame = pd.read_csv(path)
    validated = validate_trend_signals_frame(frame)
    validated = validated[TREND_SIGNAL_COLUMNS].copy()
    validated["source"] = path.stem.replace("trend_signals_", "") or path.stem
    return validated


# --------------------------------------------------------------------------- #
# Merging                                                                       #
# --------------------------------------------------------------------------- #

def combine_signals(
    per_retailer: list[pd.DataFrame],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Combine per-retailer frames into one canonical trend_signals frame.

    `weights[i]` corresponds to `per_retailer[i]`. When omitted, every
    retailer gets weight 1.0 (plain mean across retailers that actually
    saw the value).
    """
    if not per_retailer:
        raise ValueError("combine_signals: no input frames provided.")

    if weights is None:
        weights = [1.0] * len(per_retailer)
    if len(weights) != len(per_retailer):
        raise ValueError(
            f"combine_signals: got {len(per_retailer)} frames but "
            f"{len(weights)} weights (they must match 1:1)."
        )

    # Tag every row with the retailer's weight so we can groupby later.
    tagged = []
    for frame, weight in zip(per_retailer, weights):
        tmp = frame.copy()
        tmp["_weight"] = float(weight)
        tagged.append(tmp)
    stacked = pd.concat(tagged, ignore_index=True)

    # Drop default-score rows. A retailer that wrote DEFAULT_MISSING_SCORE
    # didn't see this value at all and shouldn't drag the combined score
    # toward zero. We use math.isclose to be safe against CSV rounding.
    def _is_real(score: float) -> bool:
        return not math.isclose(
            float(score), DEFAULT_MISSING_SCORE,
            rel_tol=0.0, abs_tol=1e-9,
        )
    real = stacked[stacked["current"].apply(_is_real)].copy()

    # Weighted mean per (feature_type, feature_value) across retailers
    # that did see the value. Pandas lets us do this as:
    #     sum(current * weight) / sum(weight)
    if not real.empty:
        real["_num"] = real["current"].astype(float) * real["_weight"]
        grouped = real.groupby(["feature_type", "feature_value"], as_index=False).agg(
            _num=("_num", "sum"),
            _den=("_weight", "sum"),
        )
        grouped["current"] = (grouped["_num"] / grouped["_den"]).round(6)
        combined_real = grouped[["feature_type", "feature_value", "current"]]
    else:
        combined_real = pd.DataFrame(columns=TREND_SIGNAL_COLUMNS)

    # Any (feature_type, feature_value) that only ever appeared as a
    # DEFAULT_MISSING_SCORE still needs a row so downstream consumers
    # see the full key space. Re-introduce those with the default score.
    all_keys = stacked[["feature_type", "feature_value"]].drop_duplicates()
    merged = all_keys.merge(
        combined_real, on=["feature_type", "feature_value"], how="left"
    )
    merged["current"] = merged["current"].fillna(DEFAULT_MISSING_SCORE).round(6)

    return merged[TREND_SIGNAL_COLUMNS].sort_values(
        ["feature_type", "feature_value"]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-retailer trend_signals_*.csv files into a single "
            "trend_signals.csv usable by the rest of the pipeline."
        )
    )
    parser.add_argument(
        "--input", action="append", default=None,
        help=(
            "Path to a per-source trend signals CSV. Repeat for multiple "
            "sources. If omitted, every trend_signals_*.csv in the default "
            "data/trend_signals/ directory is auto-discovered."
        ),
    )
    parser.add_argument(
        "--weight", action="append", type=float, default=None,
        help=(
            "Weight for the corresponding --input (same order). Defaults to "
            "1.0 for every retailer."
        ),
    )
    parser.add_argument(
        "--signals-dir", default=str(DEFAULT_SIGNALS_DIR),
        help=(
            "Directory to scan for trend_signals_*.csv when --input is not "
            f"given (default: {DEFAULT_SIGNALS_DIR}). Per-source files live "
            "in data/trend_signals/; the combined output is written one level "
            "up in data/."
        ),
    )
    parser.add_argument(
        "--output-path", default=str(DEFAULT_OUTPUT_PATH),
        help=f"Where to write the combined CSV (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_paths = [Path(p).expanduser().resolve() for p in args.input]
    else:
        signals_dir = Path(args.signals_dir).expanduser().resolve()
        input_paths = discover_retailer_files(signals_dir, exclude=output_path)
        if not input_paths:
            print(
                f"ERROR: no {RETAILER_FILE_GLOB} files found in {signals_dir}.\n"
                f"Run a retailer scraper first (e.g. hollister_scraper.py)."
            )
            sys.exit(1)

    if args.weight and len(args.weight) != len(input_paths):
        print(
            f"ERROR: got {len(input_paths)} --input files but "
            f"{len(args.weight)} --weight values. They must match 1:1."
        )
        sys.exit(1)

    print("Combining trend signal files:")
    for path in input_paths:
        print(f"  - {path}")
    print(f"Output → {output_path}\n")

    per_retailer: list[pd.DataFrame] = []
    for path in input_paths:
        if not path.exists():
            print(f"  WARNING: missing file, skipping: {path}")
            continue
        frame = load_retailer_signals(path)
        real_values = (frame["current"] != DEFAULT_MISSING_SCORE).sum()
        print(
            f"  loaded {len(frame):>3} rows "
            f"({real_values} non-default) from {path.name}"
        )
        per_retailer.append(frame.drop(columns=["source"]))

    if not per_retailer:
        print("ERROR: no valid retailer files loaded.")
        sys.exit(1)

    combined = combine_signals(per_retailer, weights=args.weight)
    validated = validate_trend_signals_frame(combined)
    validated.to_csv(output_path, index=False)

    real_values = (validated["current"] != DEFAULT_MISSING_SCORE).sum()
    print(
        f"\nWrote {len(validated)} rows "
        f"({real_values} with real signal, "
        f"{len(validated) - real_values} at default={DEFAULT_MISSING_SCORE}) "
        f"→ {output_path}"
    )


if __name__ == "__main__":
    main()
