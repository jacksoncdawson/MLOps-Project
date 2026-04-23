"""
H&M seasonal label generator for trndly training data.

Reads the H&M Kaggle transaction history to produce real, labeled training
data for the listing timeline classifier. This replaces the fully synthetic
train/val/test CSVs with examples where the best_timeframe label comes from
actual historical purchase seasonality.

HOW IT WORKS
------------
1. Join articles.csv + transactions_train.csv on article_id.
2. Map H&M attribute values to the feature_values in feature_contract.py
   (color, category, material).
3. For each unique (color, category, material) combination, find the calendar
   month when that combination historically sold fastest (peak_month).
4. Generate 12 training examples per combination — one for each reference
   month. For each reference month, compute months_until_peak and map it
   to a best_timeframe label:
       0 months  → "current"
       1 month   → "next_week"
       2 months  → "next_month"
       3–4 months → "three_months"
       5+ months → "six_months"
5. Load the current trend_signals.csv (written by google_trends_collector.py)
   to attach real current trend scores as input features.
6. Shuffle and write as train/val/test CSVs into the synthetic_data directory,
   replacing the fully synthetic splits.

PIPELINE ORDER
--------------
Run google_trends_collector.py first (to produce a real trend_signals.csv),
then run this script to produce real-labeled train/val/test splits.

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
    TARGET_COLUMN_DEFAULT,
    TIMEFRAMES,
    build_trend_lookup,
    item_to_feature_row,
    load_trend_signals_frame,
    normalize_token,
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
    default_trend_signals = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "synthetic_data"
        / "trend_signals.csv"
    )
    default_output = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "synthetic_data"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Generate real-labeled train/val/test splits from H&M seasonal "
            "purchase data and current Google Trends signals."
        )
    )
    parser.add_argument(
        "--articles-path",
        required=True,
        help="Path to the H&M articles.csv file from the Kaggle dataset.",
    )
    parser.add_argument(
        "--transactions-path",
        required=True,
        help="Path to the H&M transactions_train.csv file from the Kaggle dataset.",
    )
    parser.add_argument(
        "--trend-signals-path",
        default=str(default_trend_signals),
        help="Path to the trend_signals.csv produced by google_trends_collector.py.",
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
# Peak month computation                                                        #
# --------------------------------------------------------------------------- #

def compute_peak_months(
    transactions: pd.DataFrame,
    attrs: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each unique (color, category, material) combination, find the
    calendar month with the highest average purchase share.

    Returns a DataFrame with columns: color, category, material, peak_month.
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

    monthly_avg = (
        combo_counts.groupby(["color", "category", "material", "month"])["share"]
        .mean()
        .reset_index()
    )

    peak_months = (
        monthly_avg.loc[
            monthly_avg.groupby(["color", "category", "material"])["share"].idxmax()
        ][["color", "category", "material", "month"]]
        .rename(columns={"month": "peak_month"})
        .reset_index(drop=True)
    )

    return peak_months


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
) -> list[dict[str, Any]]:
    """
    For each (color, category, material) combination × 12 reference months,
    produce one training row with real current features and an H&M-derived label.
    """
    rows: list[dict[str, Any]] = []
    item_index = 0

    for _, peak_row in peak_months.iterrows():
        color = normalize_token(peak_row["color"])
        category = normalize_token(peak_row["category"])
        material = normalize_token(peak_row["material"])
        peak_month = int(peak_row["peak_month"])

        item = {"color": color, "category": category, "material": material}
        feature_row = item_to_feature_row(item=item, lookup=lookup)

        for ref_month in range(1, 13):
            months = months_until_peak(peak_month=peak_month, reference_month=ref_month)
            label = timeframe_from_months(months)
            item_name = f"{material.title()} {category.title()} #{item_index:05d}"
            rows.append(
                {
                    "item_name": item_name,
                    "color": color,
                    "category": category,
                    "material": material,
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
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    n_train = int(len(rows) * TRAIN_FRAC)
    n_val = int(len(rows) * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    all_rows = pd.DataFrame(rows)
    return (
        all_rows.iloc[train_idx].reset_index(drop=True),
        all_rows.iloc[val_idx].reset_index(drop=True),
        all_rows.iloc[test_idx].reset_index(drop=True),
    )


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

    print("Computing peak purchase months (~30 seconds)...")
    peak_months = compute_peak_months(transactions, attrs)
    print(f"  {len(peak_months):,} unique (color, category, material) combinations")

    print("Loading trend signals for current feature scores...")
    trend_frame = load_trend_signals_frame(trend_signals_path)
    lookup = build_trend_lookup(trend_frame)

    print("Building training rows (12 reference months × each combination)...")
    rows = build_training_rows(peak_months=peak_months, lookup=lookup)
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
