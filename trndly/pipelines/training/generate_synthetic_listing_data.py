from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    FEATURE_TYPES,
    FEATURE_VECTOR_COLUMNS,
    TARGET_COLUMN_DEFAULT,
    TIMEFRAMES,
    build_trend_lookup,
    compute_alignment_scores,
    item_to_feature_row,
)

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
        default=str(Path(__file__).resolve().parent / "synthetic_data"),
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
        "--label-temperature",
        type=float,
        default=9.0,
        help=(
            "Higher values make sampled labels more likely to match the "
            "strongest timeframe score."
        ),
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.04,
        help="Probability of replacing sampled label with a random timeframe.",
    )
    return parser.parse_args()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    total = exponentiated.sum()
    if math.isclose(float(total), 0.0):
        return np.full_like(exponentiated, fill_value=1.0 / len(exponentiated))
    return exponentiated / total


def _feature_profile(
    rng: np.random.Generator,
    preferred_index: int,
) -> np.ndarray:
    distance = np.abs(np.arange(len(TIMEFRAMES), dtype=float) - float(preferred_index))
    spread = rng.uniform(0.8, 1.9)
    curve = np.exp(-(distance**2) / (2 * spread * spread))
    baseline = rng.uniform(0.08, 0.22)
    scaled = baseline + (0.88 - baseline) * curve
    noise = rng.normal(loc=0.0, scale=0.03, size=len(TIMEFRAMES))
    return np.clip(scaled + noise, a_min=0.0, a_max=1.0)


def generate_trend_signals(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for feature_type in FEATURE_TYPES:
        possible_values = FEATURE_VALUES[feature_type]
        for feature_value in possible_values:
            preferred_index = int(rng.integers(low=0, high=len(TIMEFRAMES)))
            profile = _feature_profile(rng=rng, preferred_index=preferred_index)
            row: dict[str, Any] = {
                "feature_type": feature_type,
                "feature_value": feature_value,
            }
            for timeframe_index, timeframe in enumerate(TIMEFRAMES):
                row[timeframe] = round(float(profile[timeframe_index]), 6)
            rows.append(row)

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
    alignment_scores: dict[str, float],
    rng: np.random.Generator,
    temperature: float,
    label_noise: float,
) -> str:
    logits = np.array([alignment_scores[timeframe] for timeframe in TIMEFRAMES], dtype=float)
    probabilities = _softmax(logits * temperature)
    label = str(rng.choice(TIMEFRAMES, p=probabilities))

    if rng.random() < label_noise:
        label = str(rng.choice(TIMEFRAMES))
    return label


def build_supervised_split(
    size: int,
    split_name: str,
    lookup: dict[str, dict[str, dict[str, float]]],
    rng: np.random.Generator,
    label_temperature: float,
    label_noise: float,
    index_offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row_index in range(size):
        item = _sample_item(rng=rng, index=index_offset + row_index)
        alignment_scores = compute_alignment_scores(item=item, lookup=lookup)
        feature_row = item_to_feature_row(item=item, lookup=lookup)
        best_timeframe = _sample_label(
            alignment_scores=alignment_scores,
            rng=rng,
            temperature=label_temperature,
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
    lookup: dict[str, dict[str, dict[str, float]]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    payloads: list[dict[str, str]] = []
    payloads_with_reference: list[dict[str, Any]] = []

    for item_index in range(size):
        item = _sample_item(rng=rng, index=10_000 + item_index)
        alignment_scores = compute_alignment_scores(item=item, lookup=lookup)
        expected_best = max(TIMEFRAMES, key=lambda timeframe: alignment_scores[timeframe])

        payloads.append(item)
        payloads_with_reference.append(
            {
                **item,
                "expected_best_timeframe": expected_best,
                "alignment_scores": alignment_scores,
            }
        )

    return payloads, payloads_with_reference


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
        label_temperature=args.label_temperature,
        label_noise=args.label_noise,
        index_offset=0,
    )
    val_frame = build_supervised_split(
        size=args.val_size,
        split_name="val",
        lookup=lookup,
        rng=rng,
        label_temperature=args.label_temperature,
        label_noise=args.label_noise,
        index_offset=args.train_size,
    )
    test_frame = build_supervised_split(
        size=args.test_size,
        split_name="test",
        lookup=lookup,
        rng=rng,
        label_temperature=args.label_temperature,
        label_noise=args.label_noise,
        index_offset=args.train_size + args.val_size,
    )

    user_payloads, payloads_with_reference = build_inference_payloads(
        size=args.inference_size,
        lookup=lookup,
        rng=rng,
    )

    trend_path = output_dir / "trend_signals.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    payloads_path = output_dir / "user_upload_items.json"
    payloads_with_reference_path = output_dir / "user_upload_items_with_reference.json"

    trend_signals.to_csv(trend_path, index=False)
    train_frame.to_csv(train_path, index=False)
    val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)

    payloads_path.write_text(json.dumps(user_payloads, indent=2), encoding="utf-8")
    payloads_with_reference_path.write_text(
        json.dumps(payloads_with_reference, indent=2),
        encoding="utf-8",
    )

    print("Synthetic data generated:")
    print(f"- Trend signals: {trend_path}")
    print(f"- Train split: {train_path} ({len(train_frame)} rows)")
    print(f"- Validation split: {val_path} ({len(val_frame)} rows)")
    print(f"- Test split: {test_path} ({len(test_frame)} rows)")
    print(f"- User upload payloads: {payloads_path} ({len(user_payloads)} items)")
    print(f"- Payloads with reference labels: {payloads_with_reference_path}")


if __name__ == "__main__":
    main()
