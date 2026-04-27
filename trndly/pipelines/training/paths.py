"""
Central path registry for the trndly training + serving pipeline.

Every script should import its filesystem paths from here instead of
recomputing ``Path(__file__).resolve().parents[...]`` locally. This keeps
the data layout in one place so moves/renames are a one-line change.

DIRECTORY LAYOUT
----------------
<PROJECT_ROOT>/                                ← trndly/
  pipelines/
    collectors/                                ← COLLECTORS_DIR
    training/                                  ← TRAINING_DIR
      paths.py                                 ← this file
      feature_contract.py
      data/                                    ← DATA_DIR
        trend_signals/                         ← TREND_SIGNALS_DIR
          trend_signals_google.csv             ← TREND_SIGNALS_GOOGLE_CSV
          trend_signals_hollister.csv          ← TREND_SIGNALS_HOLLISTER_CSV
          trend_signals_gap.csv                ← TREND_SIGNALS_GAP_CSV
        hm_kaggle/                             ← HM_KAGGLE_DIR  (raw Kaggle dump)
          articles.csv                         ← HM_ARTICLES_CSV
          transactions_train.csv               ← HM_TRANSACTIONS_CSV
        trend_signals.csv                      ← TREND_SIGNALS_CSV  (combined)
        seasonality_table.csv                  ← SEASONALITY_TABLE_CSV
        train.csv / val.csv / test.csv         ← TRAIN_CSV / VAL_CSV / TEST_CSV
        user_upload_items.json                 ← USER_UPLOAD_ITEMS_JSON
        user_upload_items_with_reference.json  ← USER_UPLOAD_ITEMS_WITH_REFERENCE_JSON
  backend/
  frontend/                                    ← FRONTEND_DIR
    index.html                                 ← static UI served at /ui
  tests/
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Top-level anchors                                                             #
# --------------------------------------------------------------------------- #

# This file sits at <trndly>/pipelines/training/paths.py, so parents[2] == trndly/.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

PIPELINES_DIR: Path = PROJECT_ROOT / "pipelines"
COLLECTORS_DIR: Path = PIPELINES_DIR / "collectors"
TRAINING_DIR: Path = PIPELINES_DIR / "training"

# Static demo UI, served by the FastAPI app at /ui (see scheduleServer.py).
# Lives as a sibling of backend/ since it's a project-level concern, not a
# backend-internal one.
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"

# --------------------------------------------------------------------------- #
# Training data artifacts                                                       #
# --------------------------------------------------------------------------- #

# All generated training artifacts live in DATA_DIR. Scripts should create it
# via ensure_data_dirs() rather than mkdir-ing ad hoc.
DATA_DIR: Path = TRAINING_DIR / "data"

# Per-source trend signal CSVs (one file per collector/scraper). The canonical
# combined output lives one level up in DATA_DIR so it never gets re-ingested
# by the auto-discovery glob.
TREND_SIGNALS_DIR: Path = DATA_DIR / "trend_signals"

# Canonical combined signals — written by combine_trend_signals.py, consumed
# by hmn_seasonal_processor.py and the backend inference path.
TREND_SIGNALS_CSV: Path = DATA_DIR / "trend_signals.csv"

# H&M-derived historical seasonality curves. Written by
# hmn_seasonal_processor.py, consumed everywhere compute_seasonal_features is
# used (training row assembly and inference).
SEASONALITY_TABLE_CSV: Path = DATA_DIR / "seasonality_table.csv"

# Model training splits produced by hmn_seasonal_processor.py (or the legacy
# synthetic generator).
TRAIN_CSV: Path = DATA_DIR / "train.csv"
VAL_CSV: Path = DATA_DIR / "val.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"

# Synthetic upload payloads used for smoke-testing the API surface.
USER_UPLOAD_ITEMS_JSON: Path = DATA_DIR / "user_upload_items.json"
USER_UPLOAD_ITEMS_WITH_REFERENCE_JSON: Path = DATA_DIR / "user_upload_items_with_reference.json"

# --------------------------------------------------------------------------- #
# H&M Kaggle raw dump                                                           #
# --------------------------------------------------------------------------- #

# Raw H&M Kaggle files live under DATA_DIR so they (a) stay alongside the other
# pipeline artifacts and (b) are captured by the `**/data/` .gitignore rule,
# preventing the 3+ GB transactions file from being committed. Download once
# with the Kaggle CLI; hmn_seasonal_processor.py reads from here by default.
HM_KAGGLE_DIR: Path = DATA_DIR / "hm_kaggle"
HM_ARTICLES_CSV: Path = HM_KAGGLE_DIR / "articles.csv"
HM_TRANSACTIONS_CSV: Path = HM_KAGGLE_DIR / "transactions_train.csv"

# --------------------------------------------------------------------------- #
# Per-source trend signal CSVs                                                  #
# --------------------------------------------------------------------------- #

TREND_SIGNALS_GOOGLE_CSV: Path = TREND_SIGNALS_DIR / "trend_signals_google.csv"
TREND_SIGNALS_HOLLISTER_CSV: Path = TREND_SIGNALS_DIR / "trend_signals_hollister.csv"
TREND_SIGNALS_GAP_CSV: Path = TREND_SIGNALS_DIR / "trend_signals_gap.csv"

# Glob used by combine_trend_signals.py to auto-discover every per-source file
# in TREND_SIGNALS_DIR without having to know them by name.
TREND_SIGNALS_GLOB: str = "trend_signals_*.csv"


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def per_source_trend_signals_csv(source: str) -> Path:
    """
    Path to the per-source trend signals CSV for a given source name.

    Examples:
        per_source_trend_signals_csv("google")    → TREND_SIGNALS_GOOGLE_CSV
        per_source_trend_signals_csv("hollister") → TREND_SIGNALS_HOLLISTER_CSV
        per_source_trend_signals_csv("pacsun")    → <...>/trend_signals_pacsun.csv

    Useful for new sources that don't yet have a dedicated module-level
    constant — the filename is derived from the source name so adding a new
    scraper doesn't require editing this file.
    """
    return TREND_SIGNALS_DIR / f"trend_signals_{source}.csv"


def ensure_data_dirs() -> None:
    """
    Create DATA_DIR, TREND_SIGNALS_DIR, and HM_KAGGLE_DIR if they don't exist.
    Idempotent — safe to call at the top of any script that is about to write
    artifacts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TREND_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    HM_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
