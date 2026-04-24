from __future__ import annotations

import argparse
import json
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
    FEATURE_TYPES,
    FEATURE_VECTOR_COLUMNS,
    TARGET_COLUMN_DEFAULT,
    TIMEFRAMES,
    build_trend_lookup,
    item_to_feature_row,
)
from pipelines.training.paths import DATA_DIR  # noqa: E402

FEATURE_VALUES = {
    "color": [
        "black",
        "white",
        "blue",
        "red",
        "green",
        "beige",
        "pink",
        "gray",
        "navy",
        "brown",
        "purple",
    ],
    "category": [
        "pants",
        "shorts",
        "skirt",
        "dress",
        "tops",
        "outerwear",
        "shoes",
        "accessories",
    ],
    "material": [
        "cotton",
        "denim",
        "linen",
        "silk",
        "wool",
        "polyester",
        "leather",
        "knit",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic trend signals, user upload payloads, "
            "and model-ready train/val/test datasets."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Directory where synthetic artifacts are written.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=700,
        help="Number of synthetic training rows.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=200,
        help="Number of synthetic validation rows.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of synthetic test rows.",
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=25,
        help="Number of synthetic user upload examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.04,
        help="Probability of replacing sampled label with a random timeframe.",
    )
    return parser.parse_args()


def generate_trend_signals(seed: int) -> pd.DataFrame:
    """
    Generate a synthetic trend_signals.csv with a single `current` score
    per (feature_type, feature_value). Scores are uniformly distributed
    across 0–1 so the full range of the feature space is covered.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for feature_type in FEATURE_TYPES:
        for feature_value in FEATURE_VALUES[feature_type]:
            rows.append(
                {
                    "feature_type": feature_type,
                    "feature_value": feature_value,
                    "current": round(float(rng.uniform(0.05, 0.95)), 6),
                }
            )

    return pd.DataFrame(rows)


def _sample_item(rng: np.random.Generator, index: int) -> dict[str, str]:
    color = str(rng.choice(FEATURE_VALUES["color"]))
    category = str(rng.choice(FEATURE_VALUES["category"]))
    material = str(rng.choice(FEATURE_VALUES["material"]))
    item_name = f"{material.title()} {category.title()} #{index:04d}"
    return {
        "item_name": item_name,
        "color": color,
        "category": category,
        "material": material,
    }


def _sample_label(
    avg_current: float,
    rng: np.random.Generator,
    label_noise: float,
) -> str:
    """
    Sample a best_timeframe label from avg_current trend score.

    High current score → item is trending now → label skews toward "current".
    Low current score  → item will trend later → label skews toward future timeframes.

    This creates a learnable signal: the model can discover that high
    avg_current predicts "current" and low avg_current predicts a later horizon.
    """
    if avg_current >= 0.65:
        base_label = "current"
    elif avg_current >= 0.50:
        base_label = "next_week"
    elif avg_current >= 0.35:
        base_label = "next_month"
    elif avg_current >= 0.20:
        base_label = "three_months"
    else:
        base_label = "six_months"

    if rng.random() < label_noise:
        return str(rng.choice(TIMEFRAMES))
    return base_label


def build_supervised_split(
    size: int,
    split_name: str,
    lookup: dict[str, dict[str, float]],
    rng: np.random.Generator,
    label_noise: float,
    index_offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row_index in range(size):
        item = _sample_item(rng=rng, index=index_offset + row_index)
        feature_row = item_to_feature_row(item=item, lookup=lookup)
        avg_current = feature_row.get("avg_current", DEFAULT_MISSING_SCORE)
        best_timeframe = _sample_label(
            avg_current=avg_current,
            rng=rng,
            label_noise=label_noise,
        )
        rows.append(
            {
                "split": split_name,
                **item,
                **feature_row,
                TARGET_COLUMN_DEFAULT: best_timeframe,
            }
        )

    frame = pd.DataFrame(rows)
    feature_subset = frame[FEATURE_VECTOR_COLUMNS]
    frame[FEATURE_VECTOR_COLUMNS] = feature_subset.clip(lower=0.0, upper=1.0)
    return frame


def build_inference_payloads(
    size: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    return [_sample_item(rng=rng, index=10_000 + i) for i in range(size)]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trend_signals = generate_trend_signals(seed=args.seed)
    lookup = build_trend_lookup(trend_signals)

    rng = np.random.default_rng(args.seed + 17)
    train_frame = build_supervised_split(
        size=args.train_size,
        split_name="train",
        lookup=lookup,
        rng=rng,
        label_noise=args.label_noise,
        index_offset=0,
    )
    val_frame = build_supervised_split(
        size=args.val_size,
        split_name="val",
        lookup=lookup,
        rng=rng,
        label_noise=args.label_noise,
        index_offset=args.train_size,
    )
    test_frame = build_supervised_split(
        size=args.test_size,
        split_name="test",
        lookup=lookup,
        rng=rng,
        label_noise=args.label_noise,
        index_offset=args.train_size + args.val_size,
    )

    user_payloads = build_inference_payloads(size=args.inference_size, rng=rng)

    trend_path = output_dir / "trend_signals.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    payloads_path = output_dir / "user_upload_items.json"

    trend_signals.to_csv(trend_path, index=False)
    train_frame.to_csv(train_path, index=False)
    val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)
    payloads_path.write_text(json.dumps(user_payloads, indent=2), encoding="utf-8")

    print("Synthetic data generated:")
    print(f"- Trend signals: {trend_path}")
    print(f"- Train split: {train_path} ({len(train_frame)} rows)")
    print(f"- Validation split: {val_path} ({len(val_frame)} rows)")
    print(f"- Test split: {test_path} ({len(test_frame)} rows)")
    print(f"- User upload payloads: {payloads_path} ({len(user_payloads)} items)")


if __name__ == "__main__":
    main()
