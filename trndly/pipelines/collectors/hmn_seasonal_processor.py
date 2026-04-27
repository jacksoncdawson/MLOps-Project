"""
H&M seasonal label + feature generator for trndly training data.

Reads the H&M Kaggle transaction history to produce real, labeled training
data for the listing timeline classifier. The H&M data feeds the pipeline
in two ways:

  1. As LABELS — each (color, category, material) combo has a peak month, and
     the months_until_peak from a given reference month maps to best_timeframe.
  2. As FEATURES — the full 12-month historical purchase-share curve for each
     combo (peak-normalized) is exposed at training and inference time so the
     model sees the actual seasonality, not just the label distilled from it.

Without (2) the model would receive identical inputs for all 12 reference
months of a given combo and could not learn the seasonality at all.

HOW IT WORKS
------------
1. Join articles.csv + transactions_train.csv on article_id.
2. Map H&M attribute values to the feature_values in feature_contract.py
   (color, category, material).
3. For each unique (color, category, material) combination, compute the
   normalized 12-month purchase-share curve. Write that curve to
   seasonality_table.csv (one row per combo, peak-normalized so max == 1.0).
4. The peak month for each combo is argmax of its curve. Generate 12
   training examples per combination — one for each reference month. For
   each reference month, compute months_until_peak and map it to a
   best_timeframe label:
       0 months  → "current"
       1 month   → "next_week"
       2 months  → "next_month"
       3–4 months → "three_months"
       5+ months → "six_months"
5. Load the canonical trend_signals.csv (produced by combine_trend_signals.py
   from trend_signals_google/hollister/gap) and the freshly built
   seasonality_table to compute each row's full feature vector via
   feature_contract.item_to_feature_row.
6. Shuffle and write as train/val/test CSVs into the data/ directory,
   replacing the fully synthetic splits.

PIPELINE ORDER
--------------
1. python google_trends_collector.py    # → trend_signals_google.csv
2. python hollister_scraper.py          # → trend_signals_hollister.csv
3. python gap_scraper.py                # → trend_signals_gap.csv
4. python combine_trend_signals.py      # → canonical trend_signals.csv
5. python hmn_seasonal_processor.py     # this script — consumes the combined
                                        #   trend_signals.csv and writes
                                        #   seasonality_table.csv +
                                        #   train/val/test splits.

DATA REQUIRED
-------------
Download from: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
  - articles.csv
  - transactions_train.csv

Usage:
  python hmn_seasonal_processor.py \\
      --articles-path     /path/to/articles.csv \\
      --transactions-path /path/to/transactions_train.csv

  python hmn_seasonal_processor.py \\
      --articles-path     /path/to/articles.csv \\
      --transactions-path /path/to/transactions_train.csv \\
      --trend-signals-path path/to/trend_signals.csv \\
      --output-dir        path/to/output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    DEFAULT_MISSING_SCORE,
    FEATURE_VECTOR_COLUMNS,
    SEASONALITY_ID_COLUMNS,
    SEASONALITY_MONTH_COLUMNS,
    SEASONALITY_TABLE_COLUMNS,
    TARGET_COLUMN_DEFAULT,
    TIMEFRAMES,
    build_seasonality_table,
    build_trend_lookup,
    item_to_feature_row,
    load_trend_signals_frame,
    normalize_token,
    validate_seasonality_frame,
)
from pipelines.training.paths import (  # noqa: E402
    DATA_DIR,
    HM_ARTICLES_CSV,
    HM_TRANSACTIONS_CSV,
    TREND_SIGNALS_CSV,
)

# --------------------------------------------------------------------------- #
# Attribute mapping tables                                                      #
# Maps H&M column values → feature_values used in feature_contract.py.        #
# --------------------------------------------------------------------------- #

HMN_COLOR_MAP: dict[str, str] = {
    "black": "black",
    "white": "white",
    "off white": "white",
    "blue": "blue",
    "light blue": "blue",
    "dark blue": "navy",
    "navy blue": "navy",
    "red": "red",
    "dark red": "red",
    "green": "green",
    "dark green": "green",
    "khaki green": "green",
    "beige": "beige",
    "light beige": "beige",
    "mole": "beige",
    "sand": "beige",
    "pink": "pink",
    "light pink": "pink",
    "dusty pink": "pink",
    "grey": "gray",
    "light grey": "gray",
    "dark grey": "gray",
    "greyish beige": "gray",
    "brown": "brown",
    "dark brown": "brown",
    "bronze/copper": "brown",
    "purple": "purple",
    "lilac purple": "purple",
}

HMN_CATEGORY_MAP: dict[str, str] = {
    "trousers": "pants",
    "leggings/tights": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "dress": "dress",
    "swimwear bottom": "shorts",
    "top": "tops",
    "t-shirt": "tops",
    "blouse": "tops",
    "vest top": "tops",
    "sweater": "tops",
    "hoodie": "tops",
    "polo shirt": "tops",
    "jacket": "outerwear",
    "coat": "outerwear",
    "blazer": "outerwear",
    "cardigan": "outerwear",
    "waistcoat": "outerwear",
    "shoes": "shoes",
    "sneakers": "shoes",
    "boots": "shoes",
    "sandals": "shoes",
    "heels": "shoes",
    "flat shoes": "shoes",
    "bag": "accessories",
    "belt": "accessories",
    "hat/beanie": "accessories",
    "scarf": "accessories",
    "gloves": "accessories",
    "sunglasses": "accessories",
    "necklace": "accessories",
    "earring": "accessories",
    "bracelet": "accessories",
    "wallet": "accessories",
}

# Checked in order — first match in detail_desc wins.
HMN_MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    ("denim", "denim"),
    ("leather", "leather"),
    ("linen", "linen"),
    ("silk", "silk"),
    ("wool", "wool"),
    ("cashmere", "wool"),
    ("polyester", "polyester"),
    ("cotton", "cotton"),
    ("knit", "knit"),
    ("knitwear", "knit"),
    ("woven", "knit"),
]

# Maps months_until_peak → best_timeframe label
MONTHS_TO_TIMEFRAME: list[tuple[range, str]] = [
    (range(0, 1), "current"),
    (range(1, 2), "next_week"),
    (range(2, 3), "next_month"),
    (range(3, 5), "three_months"),
    (range(5, 13), "six_months"),
]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    default_trend_signals = TREND_SIGNALS_CSV
    default_output = DATA_DIR
    parser = argparse.ArgumentParser(
        description=(
            "Generate real-labeled train/val/test splits from H&M seasonal "
            "purchase data and current Google Trends signals."
        )
    )
    parser.add_argument(
        "--articles-path",
        default=str(HM_ARTICLES_CSV),
        help=(
            "Path to the H&M articles.csv file from the Kaggle dataset. "
            "Defaults to the path defined in paths.HM_ARTICLES_CSV "
            "(data/hm_kaggle/articles.csv)."
        ),
    )
    parser.add_argument(
        "--transactions-path",
        default=str(HM_TRANSACTIONS_CSV),
        help=(
            "Path to the H&M transactions_train.csv file from the Kaggle dataset. "
            "Defaults to the path defined in paths.HM_TRANSACTIONS_CSV "
            "(data/hm_kaggle/transactions_train.csv)."
        ),
    )
    parser.add_argument(
        "--trend-signals-path",
        default=str(default_trend_signals),
        help=(
            "Path to the canonical trend_signals.csv produced by "
            "combine_trend_signals.py (weighted mean of Google Trends + "
            "Hollister + Gap per-source files)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help="Directory to write train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test shuffle.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Attribute extraction                                                          #
# --------------------------------------------------------------------------- #

def _map_color(value: str) -> str | None:
    return HMN_COLOR_MAP.get(str(value).strip().lower())


def _map_category(value: str) -> str | None:
    return HMN_CATEGORY_MAP.get(str(value).strip().lower())


def _map_material(detail_desc: str) -> str | None:
    lowered = str(detail_desc).lower()
    for keyword, material in HMN_MATERIAL_KEYWORDS:
        if keyword in lowered:
            return material
    return None


def extract_article_attributes(articles: pd.DataFrame) -> pd.DataFrame:
    attrs = pd.DataFrame({"article_id": articles["article_id"]})
    attrs["color"] = articles["colour_group_name"].map(_map_color)
    attrs["category"] = articles["product_type_name"].map(_map_category)
    attrs["material"] = articles["detail_desc"].map(_map_material)
    return attrs


# --------------------------------------------------------------------------- #
# Seasonality curve computation                                                 #
# --------------------------------------------------------------------------- #

def compute_seasonality_curves(
    transactions: pd.DataFrame,
    attrs: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each unique (color, category, material) combination, compute the
    full normalized 12-month purchase-share curve plus the total number of
    transactions backing it.

    Approach:
      1. Compute each combo's monthly purchase share (combo_count / total_in_year_month)
         then average that share across years to get a per-month value.
      2. Pivot to one row per combo with columns month_1..month_12. Missing
         months become 0.
      3. Peak-normalize each row so the largest month becomes 1.0. Combos with
         no transactions never make it this far (they're dropped at step 1).
      4. Attach n_observations = total raw transaction count for the combo.

    Returns:
        DataFrame with the SEASONALITY_TABLE_COLUMNS schema:
          color, category, material, month_1..month_12, n_observations.
    """
    merged = transactions[["t_dat", "article_id"]].merge(attrs, on="article_id", how="left")
    merged = merged.dropna(subset=["color", "category", "material"])
    merged["month"] = pd.to_datetime(merged["t_dat"]).dt.month
    merged["year"] = pd.to_datetime(merged["t_dat"]).dt.year

    total_by_year_month = (
        merged.groupby(["year", "month"])["article_id"]
        .count()
        .rename("total")
        .reset_index()
    )

    combo_counts = (
        merged.groupby(["year", "month", "color", "category", "material"])["article_id"]
        .count()
        .rename("count")
        .reset_index()
    )
    combo_counts = combo_counts.merge(total_by_year_month, on=["year", "month"])
    combo_counts["share"] = combo_counts["count"] / combo_counts["total"]

    # Average each combo's monthly share across years so seasonality dominates
    # over year-specific noise.
    monthly_avg = (
        combo_counts.groupby(["color", "category", "material", "month"])["share"]
        .mean()
        .reset_index()
    )

    # Pivot to wide form: one row per combo, 12 columns for months.
    wide = monthly_avg.pivot_table(
        index=["color", "category", "material"],
        columns="month",
        values="share",
        fill_value=0.0,
    )
    wide = wide.reindex(columns=range(1, 13), fill_value=0.0)
    wide.columns = [f"month_{m}" for m in range(1, 13)]
    wide = wide.reset_index()

    # Peak-normalize each row so max == 1.0. Rows whose curve is entirely zero
    # are filtered out (they carry no signal).
    month_matrix = wide[SEASONALITY_MONTH_COLUMNS].to_numpy(dtype=float)
    peaks = month_matrix.max(axis=1)
    keep_mask = peaks > 0.0
    wide = wide.loc[keep_mask].reset_index(drop=True)
    month_matrix = month_matrix[keep_mask]
    peaks = peaks[keep_mask]
    wide[SEASONALITY_MONTH_COLUMNS] = month_matrix / peaks[:, None]

    # n_observations = total raw transactions for the combo across all years/months.
    n_obs = (
        merged.groupby(["color", "category", "material"])["article_id"]
        .count()
        .rename("n_observations")
        .reset_index()
    )
    wide = wide.merge(n_obs, on=SEASONALITY_ID_COLUMNS, how="left")
    wide["n_observations"] = wide["n_observations"].fillna(0).astype(int)

    return wide[SEASONALITY_TABLE_COLUMNS].copy()


def derive_peak_months(seasonality_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Pull the peak month (1..12) out of each combo's seasonality curve.

    Returns:
        DataFrame with columns: color, category, material, peak_month.
    """
    month_matrix = seasonality_frame[SEASONALITY_MONTH_COLUMNS].to_numpy(dtype=float)
    peak_indices = np.argmax(month_matrix, axis=1)
    peaks = seasonality_frame[SEASONALITY_ID_COLUMNS].copy()
    peaks["peak_month"] = peak_indices + 1  # convert 0-indexed → 1..12
    return peaks.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Label mapping                                                                 #
# --------------------------------------------------------------------------- #

def months_until_peak(peak_month: int, reference_month: int) -> int:
    """Calendar months from reference_month until peak_month (0–11)."""
    return (peak_month - reference_month) % 12


def timeframe_from_months(months: int) -> str:
    for month_range, label in MONTHS_TO_TIMEFRAME:
        if months in month_range:
            return label
    return "six_months"


# --------------------------------------------------------------------------- #
# Training data assembly                                                        #
# --------------------------------------------------------------------------- #

def build_training_rows(
    peak_months: pd.DataFrame,
    lookup: dict[str, dict[str, float]],
    seasonality_table: dict,
) -> list[dict[str, Any]]:
    """
    For each (color, category, material) combination × 12 reference months,
    produce one training row whose features include both today's Google Trends
    snapshot AND the historical seasonality curve evaluated at that ref_month.

    Crucially, each ref_month yields a DIFFERENT feature row because the
    seasonality features depend on ref_month — this is what lets the model
    learn the seasonal pattern instead of collapsing to one prediction per combo.
    """
    rows: list[dict[str, Any]] = []
    item_index = 0

    for _, peak_row in peak_months.iterrows():
        color = normalize_token(peak_row["color"])
        category = normalize_token(peak_row["category"])
        material = normalize_token(peak_row["material"])
        peak_month = int(peak_row["peak_month"])

        item = {"color": color, "category": category, "material": material}

        for ref_month in range(1, 13):
            feature_row = item_to_feature_row(
                item=item,
                lookup=lookup,
                reference_month=ref_month,
                seasonality_table=seasonality_table,
            )
            months = months_until_peak(peak_month=peak_month, reference_month=ref_month)
            label = timeframe_from_months(months)
            item_name = f"{material.title()} {category.title()} #{item_index:05d}"
            rows.append(
                {
                    "item_name": item_name,
                    "color": color,
                    "category": category,
                    "material": material,
                    "reference_month": ref_month,
                    **feature_row,
                    TARGET_COLUMN_DEFAULT: label,
                }
            )
            item_index += 1

    return rows


def split_rows(
    rows: list[dict],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split rows into train/val/test with **combo-level grouping**: every one of
    a combo's 12 reference-month rows lands in the same split. This prevents
    leakage where a combo's Jan row is in train and its Feb row is in test —
    which would let the model trivially memorize per-combo seasonality and
    inflate val/test metrics far above what it can actually generalize to a
    brand-new combo at inference time.
    """
    all_rows = pd.DataFrame(rows)

    combo_keys = (
        all_rows[["color", "category", "material"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    combo_order = rng.permutation(len(combo_keys))
    shuffled = combo_keys.iloc[combo_order].reset_index(drop=True)

    n_train = int(len(shuffled) * TRAIN_FRAC)
    n_val = int(len(shuffled) * VAL_FRAC)

    train_combos = shuffled.iloc[:n_train]
    val_combos = shuffled.iloc[n_train : n_train + n_val]
    test_combos = shuffled.iloc[n_train + n_val :]

    def _filter(combos: pd.DataFrame) -> pd.DataFrame:
        return all_rows.merge(combos, on=["color", "category", "material"], how="inner").reset_index(drop=True)

    return (_filter(train_combos), _filter(val_combos), _filter(test_combos))


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    articles_path = Path(args.articles_path).expanduser().resolve()
    transactions_path = Path(args.transactions_path).expanduser().resolve()
    trend_signals_path = Path(args.trend_signals_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (articles_path, "articles.csv"),
        (transactions_path, "transactions_train.csv"),
        (trend_signals_path, "trend_signals.csv"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    print(
        f"H&M seasonal label generator\n"
        f"  articles:      {articles_path}\n"
        f"  transactions:  {transactions_path}\n"
        f"  trend signals: {trend_signals_path}\n"
        f"  output:        {output_dir}"
    )

    print("\nLoading articles.csv...")
    articles = pd.read_csv(articles_path, dtype=str)
    print(f"  {len(articles):,} articles")

    print("Loading transactions_train.csv...")
    transactions = pd.read_csv(
        transactions_path,
        usecols=["t_dat", "article_id"],
        dtype={"article_id": str},
    )
    print(f"  {len(transactions):,} transactions")

    print("Extracting article attributes...")
    attrs = extract_article_attributes(articles)
    for ft in ("color", "category", "material"):
        print(f"  {ft}: {attrs[ft].notna().sum():,} / {len(attrs):,} articles mapped")

    print("Computing seasonality curves (~30 seconds)...")
    seasonality_frame = compute_seasonality_curves(transactions, attrs)
    print(f"  {len(seasonality_frame):,} unique (color, category, material) combinations")

    seasonality_path = output_dir / "seasonality_table.csv"
    validated_seasonality = validate_seasonality_frame(seasonality_frame)
    validated_seasonality.to_csv(seasonality_path, index=False)
    print(f"  Wrote seasonality_table → {seasonality_path}")

    print("Building seasonality lookup (with backoff levels)...")
    seasonality_table = build_seasonality_table(validated_seasonality)

    print("Deriving peak months from curves...")
    peak_months = derive_peak_months(validated_seasonality)

    print("Loading trend signals for current feature scores...")
    trend_frame = load_trend_signals_frame(trend_signals_path)
    lookup = build_trend_lookup(trend_frame)

    print("Building training rows (12 reference months × each combination)...")
    rows = build_training_rows(
        peak_months=peak_months,
        lookup=lookup,
        seasonality_table=seasonality_table,
    )
    print(f"  {len(rows):,} total training examples")

    label_dist = pd.Series([r[TARGET_COLUMN_DEFAULT] for r in rows]).value_counts()
    print("  Label distribution:")
    for label in TIMEFRAMES:
        count = label_dist.get(label, 0)
        pct = 100 * count / len(rows)
        print(f"    {label:<15} {count:>6,}  ({pct:.1f}%)")

    train_frame, val_frame, test_frame = split_rows(rows=rows, seed=args.seed)

    for frame, name in [
        (train_frame, "train"),
        (val_frame, "val"),
        (test_frame, "test"),
    ]:
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"\nWrote {len(frame):,} rows → {path}")

    print(f"\nColumns: {list(train_frame.columns)}")
    print(f"Features: {FEATURE_VECTOR_COLUMNS}")
    print("\nSample rows (first 3):")
    print(train_frame[["color", "category", "material", *FEATURE_VECTOR_COLUMNS, TARGET_COLUMN_DEFAULT]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
