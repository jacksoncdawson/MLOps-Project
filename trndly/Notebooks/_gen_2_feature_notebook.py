"""One-off generator for 2_feature_processing.ipynb — run from trndly/: python Notebooks/_gen_2_feature_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path


def md(s: str) -> dict:
    lines = s.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(s: str) -> dict:
    lines = s.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "code",
        "metadata": {"execution_count": None},
        "outputs": [],
        "source": lines,
    }


def main() -> None:
    nb_path = Path(__file__).resolve().parent / "2_feature_processing.ipynb"
    cells: list[dict] = []

    cells.append(
        md(
            r"""# Feature processing — univariate + fingerprint training tables

Build **calendar-strict** training rows from the processed monthly cubes on disk (e.g. written or renamed after `1_aggregate_historical.ipynb`):

| Part | Input | Output |
|------|--------|--------|
| **A — Univariate** | `processed_univariate.parquet` | `univariate_training.parquet` |
| **B — Fingerprint** | `processed_fingerprint.parquet` | `fingerprint_training.parquet` |

**Eligibility:** for anchor month `t`, require cube rows on every calendar month in **`t-3` … `t+6`** (10 months: three lags, anchor, six horizons). No reindex / zero-fill.

**Features:** `month_of_year`, `share_t`, `share_lag1` … `share_lag3` (`t-1` … `t-3`). **Labels:** `y_h1` … `y_h6` (`share_articles` at `t+1` … `t+6`).

**Splits:** `split_group` (`train` / `val` / `holdout`) from tail ranks on **each** table's `anchor_month` (defaults `K=2`, `J=2`). **Weights:** `sample_weight = sqrt(n_articles at t)` capped at **`SAMPLE_WEIGHT_MAX`** (metadata for training, not in `FEATURE_COLS`).
"""
        )
    )

    cells.append(
        code(
            r"""import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

DATA_DIR = "../data/processed"
IN_UNIVARIATE = f"{DATA_DIR}/processed_univariate.parquet"
IN_FINGERPRINT = f"{DATA_DIR}/processed_fingerprint.parquet"
OUT_UNIVARIATE = f"{DATA_DIR}/univariate_training.parquet"
OUT_FINGERPRINT = f"{DATA_DIR}/fingerprint_training.parquet"
OUT_META = f"{DATA_DIR}/feature_training_run.json"

HORIZONS = list(range(1, 7))
# Past context: share_lag1 = t-1, share_lag2 = t-2, share_lag3 = t-3
LAG_PAST_MONTHS = 3

SPLIT_K_HOLDOUT = 2
SPLIT_J_VAL = 2
SAMPLE_WEIGHT_MAX = 100.0

FINGERPRINT_COLS = [
    "product_type_id",
    "gender_id",
    "color_master_id",
    "graphical_appearance_id",
    "material_id",
]


def month_shift(m: pd.Timestamp, k: int) -> pd.Timestamp:
    return m + DateOffset(months=k)


def assign_split_group(df: pd.DataFrame, anchor_col: str = "anchor_month") -> pd.DataFrame:
    out = df.copy()
    months = sorted(out[anchor_col].unique())
    n = len(months)
    K, J = SPLIT_K_HOLDOUT, SPLIT_J_VAL
    while K + J >= n and (K > 0 or J > 0):
        if J > 0:
            J -= 1
        elif K > 0:
            K -= 1
    holdout = set(months[-K:]) if K else set()
    val = set(months[-(K + J) : -K]) if J and (K + J) <= n else set()

    def _sg(m):
        if m in holdout:
            return "holdout"
        if m in val:
            return "val"
        return "train"

    out["split_group"] = out[anchor_col].map(_sg).astype("category")
    return out


def build_calendar_strict_rows(
    cube: pd.DataFrame,
    key_cols: list[str],
    *,
    share_col: str = "share_articles",
    n_col: str = "n_articles",
    month_col: str = "month",
    extra_at_t: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    stats = {"n_groups": 0, "n_candidates": 0, "n_rows": 0}
    cube = cube.copy()
    cube[month_col] = pd.to_datetime(cube[month_col]).dt.as_unit("ns")
    rows: list[dict] = []
    extra_at_t = extra_at_t or {}

    for keys, grp in cube.groupby(key_cols, observed=True, sort=False):
        stats["n_groups"] += 1
        grp = grp.sort_values(month_col)
        idx = grp.set_index(month_col)
        if idx.index.has_duplicates:
            raise ValueError(f"Duplicate {month_col} in group {keys}")
        months_sorted = list(idx.index.sort_values())
        share = idx[share_col]
        n_art = idx[n_col]
        moy = idx["month_of_year"]
        source = idx["source"].iloc[0]

        for t in months_sorted:
            stats["n_candidates"] += 1
            need = [month_shift(t, k) for k in range(-LAG_PAST_MONTHS, 7)]
            if not all(m in share.index for m in need):
                continue
            rec: dict = {}
            if len(key_cols) == 1:
                rec[key_cols[0]] = keys
            else:
                for c, v in zip(key_cols, keys):
                    rec[c] = v
            rec["anchor_month"] = t
            rec["source"] = source
            rec["month_of_year"] = int(moy.loc[t])
            rec["share_t"] = float(share.loc[t])
            for i in range(1, LAG_PAST_MONTHS + 1):
                rec[f"share_lag{i}"] = float(share.loc[month_shift(t, -i)])
            for h in HORIZONS:
                rec[f"y_h{h}"] = float(share.loc[month_shift(t, h)])
            rec[n_col] = int(n_art.loc[t])
            for out_c, cube_c in extra_at_t.items():
                rec[out_c] = float(idx[cube_c].loc[t])
            rows.append(rec)

    stats["n_rows"] = len(rows)
    if not rows:
        return pd.DataFrame(), stats
    return pd.DataFrame(rows), stats
"""
        )
    )

    cells.append(md("## Part A — Load univariate cube"))
    cells.append(
        code(
            r"""uv = pd.read_parquet(IN_UNIVARIATE)
print("univariate:", uv.shape)
print(uv.dtypes)
print()
print("rows per dimension:")
print(uv.groupby("dimension", observed=True).size())
print()
print("month range:", uv["month"].min(), "->", uv["month"].max())
uv.head(3)
"""
        )
    )

    cells.append(md("## Part A — Build `univariate_training` rows"))
    cells.append(
        code(
            r"""uni_raw, uni_stats = build_calendar_strict_rows(
    uv,
    key_cols=["dimension", "level_id"],
    extra_at_t=None,
)
print("Part A calendar-strict stats:", uni_stats)

univariate = uni_raw.copy()
univariate["sample_weight"] = np.sqrt(np.maximum(univariate["n_articles"].astype(float), 0.0)).clip(
    upper=SAMPLE_WEIGHT_MAX
)
univariate = assign_split_group(univariate, "anchor_month")

UNIVARIATE_META = ["anchor_month", "dimension", "level_id", "source", "split_group", "sample_weight", "n_articles"]
UNIVARIATE_FEATURE_COLS = ["month_of_year", "share_t"] + [f"share_lag{i}" for i in range(1, LAG_PAST_MONTHS + 1)]
UNIVARIATE_TARGET_COLS = [f"y_h{h}" for h in HORIZONS]
univariate = univariate[UNIVARIATE_META + UNIVARIATE_FEATURE_COLS + UNIVARIATE_TARGET_COLS]

print("univariate_training shape:", univariate.shape)
print(univariate.head(2))
"""
        )
    )

    cells.append(md("## Part B — Load fingerprint cube"))
    cells.append(
        code(
            r"""fp = pd.read_parquet(IN_FINGERPRINT)
print("fingerprint:", fp.shape)
print(fp.dtypes)
print("month range:", fp["month"].min(), "->", fp["month"].max())
fp.head(3)
"""
        )
    )

    cells.append(md("## Part B — Build `fingerprint_training` rows"))
    cells.append(
        code(
            r"""fp_raw, fp_stats = build_calendar_strict_rows(
    fp,
    key_cols=FINGERPRINT_COLS,
    extra_at_t={"avg_price_t": "avg_price"},
)
print("Part B calendar-strict stats:", fp_stats)

fp_train = fp_raw.copy()
fp_train["sample_weight"] = np.sqrt(np.maximum(fp_train["n_articles"].astype(float), 0.0)).clip(
    upper=SAMPLE_WEIGHT_MAX
)
fp_train = assign_split_group(fp_train, "anchor_month")

FINGERPRINT_META = ["anchor_month", *FINGERPRINT_COLS, "source", "split_group", "sample_weight", "n_articles"]
FINGERPRINT_FEATURE_COLS = ["month_of_year", "share_t", "avg_price_t"] + [
    f"share_lag{i}" for i in range(1, LAG_PAST_MONTHS + 1)
]
FINGERPRINT_TARGET_COLS = [f"y_h{h}" for h in HORIZONS]
fp_train = fp_train[FINGERPRINT_META + FINGERPRINT_FEATURE_COLS + FINGERPRINT_TARGET_COLS]

print("fingerprint_training shape:", fp_train.shape)
print(fp_train.head(2))
"""
        )
    )

    cells.append(md("## Persist parquet + run manifest"))
    cells.append(
        code(
            r"""os.makedirs(DATA_DIR, exist_ok=True)

univariate.to_parquet(OUT_UNIVARIATE, index=False)
fp_train.to_parquet(OUT_FINGERPRINT, index=False)

meta = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "calendar_strict": {
        "past_months": LAG_PAST_MONTHS,
        "future_months": 6,
        "description": "rows require cube rows t-3..t+6 (lags at t-1,t-2,t-3)",
    },
    "split_defaults": {
        "K_holdout_tail": SPLIT_K_HOLDOUT,
        "J_val_before_holdout": SPLIT_J_VAL,
        "note": "per-table tail ranks on anchor_month",
    },
    "sample_weight": {"formula": "min(sqrt(n_articles_at_anchor), cap)", "cap": SAMPLE_WEIGHT_MAX},
    "inputs": {"univariate": IN_UNIVARIATE, "fingerprint": IN_FINGERPRINT},
    "outputs": {
        "univariate_training": {"path": OUT_UNIVARIATE, "rows": int(len(univariate)), "cols": list(univariate.columns)},
        "fingerprint_training": {
            "path": OUT_FINGERPRINT,
            "rows": int(len(fp_train)),
            "cols": list(fp_train.columns),
        },
    },
    "part_a_stats": uni_stats,
    "part_b_stats": fp_stats,
    "univariate_feature_cols": UNIVARIATE_FEATURE_COLS,
    "univariate_target_cols": UNIVARIATE_TARGET_COLS,
    "fingerprint_feature_cols": FINGERPRINT_FEATURE_COLS,
    "fingerprint_target_cols": FINGERPRINT_TARGET_COLS,
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

for p in (OUT_UNIVARIATE, OUT_FINGERPRINT, OUT_META):
    print("wrote", p, os.path.getsize(p))
"""
        )
    )

    cells.append(md("## Validation and dataset QA\n\nStructural checks, numeric bounds, parquet round-trip, `describe` summaries, and sample rows.\n"))
    cells.append(
        code(
            r"""# --- Structural asserts (univariate) ---
assert not univariate.empty, "univariate training table is empty"
dup = univariate.duplicated(subset=["dimension", "level_id", "anchor_month"]).sum()
assert dup == 0, f"univariate duplicate keys: {dup}"

for h in HORIZONS:
    assert univariate[f"y_h{h}"].notna().all(), f"univariate y_h{h} has nulls"
assert set(univariate["split_group"].cat.categories) <= {"train", "val", "holdout"}
for sg in ["train", "val", "holdout"]:
    assert (univariate["split_group"] == sg).any(), f"univariate missing split {sg}"

for h in HORIZONS:
    delta = (
        pd.to_datetime(univariate["anchor_month"]) + DateOffset(months=h) - pd.to_datetime(univariate["anchor_month"])
    ).dt.days
    assert (delta > 0).all()

# --- Structural asserts (fingerprint) ---
assert not fp_train.empty, "fingerprint training table is empty"
dup_fp = fp_train.duplicated(subset=[*FINGERPRINT_COLS, "anchor_month"]).sum()
assert dup_fp == 0, f"fingerprint duplicate keys: {dup_fp}"

for h in HORIZONS:
    assert fp_train[f"y_h{h}"].notna().all(), f"fp y_h{h} has nulls"
assert fp_train["avg_price_t"].notna().all(), "avg_price_t has nulls on kept rows"
assert np.isfinite(fp_train["avg_price_t"]).all(), "avg_price_t non-finite"
for sg in ["train", "val", "holdout"]:
    assert (fp_train["split_group"] == sg).any(), f"fingerprint missing split {sg}"

# Shares are catalog proportions; allow tiny float slack above 1.0
_share_cols_t = ["share_t"] + [f"share_lag{i}" for i in range(1, LAG_PAST_MONTHS + 1)] + [f"y_h{h}" for h in HORIZONS]
for c in _share_cols_t:
    lo, hi = float(univariate[c].min()), float(univariate[c].max())
    assert -1e-3 <= lo <= hi <= 1.0 + 1e-3, f"univariate {c} out of [0,1] range: [{lo}, {hi}]"

_share_cols_f = ["share_t"] + [f"share_lag{i}" for i in range(1, LAG_PAST_MONTHS + 1)] + [f"y_h{h}" for h in HORIZONS]
for c in _share_cols_f:
    lo, hi = float(fp_train[c].min()), float(fp_train[c].max())
    assert -1e-3 <= lo <= hi <= 1.0 + 1e-3, f"fingerprint {c} out of [0,1] range: [{lo}, {hi}]"

# Parquet round-trip (final artifacts on disk)
univariate_disk = pd.read_parquet(OUT_UNIVARIATE)
fp_disk = pd.read_parquet(OUT_FINGERPRINT)
assert len(univariate_disk) == len(univariate), "univariate parquet row count mismatch"
assert len(fp_disk) == len(fp_train), "fingerprint parquet row count mismatch"
assert list(univariate_disk.columns) == list(univariate.columns), "univariate parquet column order mismatch"
assert list(fp_disk.columns) == list(fp_train.columns), "fingerprint parquet column order mismatch"

print("all validation asserts PASSED (structure + share bounds + parquet round-trip)")

# --- Dataset QA: univariate ---
print("\n" + "=" * 72)
print("UNIVARIATE — `univariate_training` (in-memory, matches written parquet)")
print("=" * 72)
print(f"rows: {len(univariate):,}  |  columns: {len(univariate.columns)}  |  memory ~{univariate.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"anchor_month: {univariate['anchor_month'].min()} .. {univariate['anchor_month'].max()}  |  unique anchors: {univariate['anchor_month'].nunique()}")
print("\nrows per split_group:")
print(univariate["split_group"].value_counts().sort_index())
print("\nrows per dimension:")
print(univariate.groupby("dimension", observed=True).size().sort_values(ascending=False))
print("\nnumeric summary (features + targets + weight):")
_num_t = UNIVARIATE_FEATURE_COLS + UNIVARIATE_TARGET_COLS + ["sample_weight", "n_articles"]
print(univariate[_num_t].describe(percentiles=[0.05, 0.5, 0.95]).T.round(6))

print("\n--- sample: first 3 rows (key columns) ---")
_disp_t = ["anchor_month", "dimension", "level_id", "split_group", "share_t", "y_h1", "y_h6"]
print(univariate[_disp_t].head(3).to_string())
print("\n--- sample: random 3 rows (seed=42) ---")
print(univariate.sample(3, random_state=42)[_disp_t].sort_values("anchor_month").to_string())

# --- Dataset QA: fingerprint ---
print("\n" + "=" * 72)
print("FINGERPRINT — `fingerprint_training`")
print("=" * 72)
print(f"rows: {len(fp_train):,}  |  columns: {len(fp_train.columns)}  |  memory ~{fp_train.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"anchor_month: {fp_train['anchor_month'].min()} .. {fp_train['anchor_month'].max()}  |  unique anchors: {fp_train['anchor_month'].nunique()}")
print("\nrows per split_group:")
print(fp_train["split_group"].value_counts().sort_index())
print("\nnumeric summary (features + targets + weight + avg_price_t):")
_num_f = FINGERPRINT_FEATURE_COLS + FINGERPRINT_TARGET_COLS + ["sample_weight", "n_articles"]
print(fp_train[_num_f].describe(percentiles=[0.05, 0.5, 0.95]).T.round(6))

_disp_f = ["anchor_month", "product_type_id", "gender_id", "color_master_id", "split_group", "share_t", "avg_price_t", "y_h1", "y_h6"]
print("\n--- sample: first 3 rows ---")
print(fp_train[_disp_f].head(3).to_string())
print("\n--- sample: random 3 rows (seed=42) ---")
print(fp_train.sample(3, random_state=42)[_disp_f].sort_values("anchor_month").to_string())

print("\n--- calendar-strict yield (from build step) ---")
print("uni_stats:", uni_stats)
print("fp_stats:", fp_stats)
"""
        )
    )

    cells.append(md("## Visual spot checks"))
    cells.append(
        code(
            r"""import matplotlib.pyplot as plt

pick = (
    univariate.sort_values(["dimension", "level_id", "anchor_month"])
    .groupby(["dimension", "level_id"], observed=True)
    .size()
    .sort_values(ascending=False)
    .head(1)
)
(d0, lid0) = pick.index[0]
sub = univariate[(univariate["dimension"] == d0) & (univariate["level_id"] == lid0)].sort_values("anchor_month")
if len(sub) >= 1:
    row = sub.iloc[len(sub) // 2]
    t0 = row["anchor_month"]
    hist_m = [month_shift(t0, k) for k in range(-LAG_PAST_MONTHS, 1)]
    fut_m = [month_shift(t0, h) for h in HORIZONS]
    uv_ser = uv.set_index(["dimension", "level_id", "month"]).sort_index()

    def sh(d, lid, m):
        return float(uv_ser.loc[(d, lid, m), "share_articles"])

    xs = hist_m + fut_m
    ys = [sh(d0, lid0, m) for m in xs]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(len(xs)), ys, "o-")
    ax.axvline(LAG_PAST_MONTHS, color="gray", ls="--", label="anchor")
    ax.set_xticks(range(len(xs)), [m.strftime("%Y-%m") for m in xs], rotation=45, ha="right")
    ax.set_title(f"Part A spot: {d0} level_id={lid0}")
    ax.set_ylabel("share_articles")
    ax.legend()
    plt.tight_layout()
    plt.show()

fp_counts = fp_train.groupby(FINGERPRINT_COLS, observed=True).size().sort_values(ascending=False)
if len(fp_counts):
    key = fp_counts.index[0]
    mask = np.logical_and.reduce([fp_train[c] == key[i] for i, c in enumerate(FINGERPRINT_COLS)])
    q = fp_train.loc[mask].sort_values("anchor_month")
    if len(q) >= 1:
        row = q.iloc[len(q) // 2]
        t0 = row["anchor_month"]
        hist_m = [month_shift(t0, k) for k in range(-LAG_PAST_MONTHS, 1)]
        fut_m = [month_shift(t0, h) for h in HORIZONS]
        fp_idx = fp.set_index([*FINGERPRINT_COLS, "month"]).sort_index()

        def sh_fp(k, m):
            return float(fp_idx.loc[(*k, m), "share_articles"])

        xs = hist_m + fut_m
        ys = [sh_fp(key, m) for m in xs]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(len(xs)), ys, "o-")
        ax.axvline(LAG_PAST_MONTHS, color="gray", ls="--")
        ax.set_xticks(range(len(xs)), [m.strftime("%Y-%m") for m in xs], rotation=45, ha="right")
        ax.set_title("Part B spot: fingerprint with most training rows")
        ax.set_ylabel("share_articles")
        plt.tight_layout()
        plt.show()
"""
        )
    )

    tfeat = "`, `".join(["month_of_year", "share_t"] + [f"share_lag{i}" for i in range(1, 4)])
    ttarg = "`, `".join([f"y_h{h}" for h in range(1, 7)])
    ffeat = "`, `".join(["month_of_year", "share_t", "avg_price_t"] + [f"share_lag{i}" for i in range(1, 4)])
    ftarg = ttarg

    cells.append(
        md(
            f"""## Downstream `3_*` handoff

Train with **FEATURE_COLS** only in `X`; pass **sample_weight** separately if the estimator supports it.

**Univariate — `univariate_training.parquet`**

| | Columns |
|---|---------|
| **FEATURE_COLS** | `{tfeat}` |
| **TARGET_COLS** | `{ttarg}` |
| **Metadata** | `anchor_month`, `dimension`, `level_id`, `source`, `split_group`, `sample_weight`, `n_articles` |

**Fingerprint — `fingerprint_training.parquet`**

| | Columns |
|---|---------|
| **FEATURE_COLS** | `{ffeat}` |
| **TARGET_COLS** | `{ftarg}` |
| **Metadata** | `anchor_month`, five fingerprint id columns, `source`, `split_group`, `sample_weight`, `n_articles` |
"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    nb_path.write_text(json.dumps(nb, indent=1))
    print("Wrote", nb_path, "n_cells=", len(cells))


if __name__ == "__main__":
    main()
