# Monthly tick runbook

The monthly tick produces a fresh predictions parquet and publishes it
as the static JSON the SPA serves. This document is the operator
runbook: prerequisites,
how to invoke each stage, what to expect, and how to debug a failure.

---

## Prerequisites

1. **Python environment.** `trndly/.venv/bin/python` with
   `requirements.txt` installed.
   ```bash
   cd trndly
   python3.11 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. **Historical cubes.** `data/processed/historical_*.parquet` must
   exist (one-time bootstrap from notebooks 0 → 1 against the H&M Kaggle
   dump in `data/raw/kaggle/`). These are immutable; the tick only reads
   them.

3. **MLflow tracking URI** *(optional — not used by the monthly tick).*
   The tick's champion management is fully local: promotion is decided by
   comparing this tick's candidate `model_training_run.json` against
   `data/models/champion.json` (see the `evaluate` stage). The private
   MLflow (Cloud Run + Cloud SQL + GCS, see `serving-redesign.md` Phase 3)
   is live but not yet wired into any stage below — that's Phase 4
   ("MVP vs. target" at the bottom).

4. **Outbound network** for the scrape stage. Each retailer has its own
   bot-protection quirks (see `pipelines/collectors/README.md`).

---

## Running the tick

### Full chain

```bash
.venv/bin/python -m pipelines.monthly run
```

Stage order: `scrape → build_cube → aggregate → features → train → evaluate → predict → publish`.
Wall-clock: ~15 minutes, dominated by the scrape stage (~9–10 min with
PDP enrichment).

**Per-tick checkpoints + idempotency (plan §12).** A `run` writes an immutable
checkpoint under `data/ticks/<YYYY-MM>/` (merged → training → model candidate →
predictions → published), keyed by the **tick month** (current calendar month by
default, or `--month YYYY-MM`). `run` is a **no-op when the tick's `_SUCCESS`
marker already exists** — pass `--force` to re-run and overwrite that month. The
manifest is written and `_SUCCESS` is touched **last**, so a crash never leaves a
tick marked complete. `historical_*`/`live_*` stay in `data/processed/` as shared
inputs; the cross-tick champion lives in `data/models/`.

The `run` subcommand takes `--month`, `--force`, `--skip-scrape`, and
`--skip-build-cube`. It does **not** forward scraper flags — to control the
scrapers (subset of retailers, PDP enrichment) call the standalone scrape module
directly (see the `scrape` stage section).

### Skip stages / re-run

If `data/raw/items/items_*.csv` are already on disk, skip the scrape:

```bash
.venv/bin/python -m pipelines.monthly run --skip-scrape
```

If the `data/processed/live_*_<YYYY-MM>.parquet` cubes are also current, skip
the rebuild too:

```bash
.venv/bin/python -m pipelines.monthly run --skip-scrape --skip-build-cube
```

To re-run a month that already has a `_SUCCESS` marker:

```bash
.venv/bin/python -m pipelines.monthly run --force
.venv/bin/python -m pipelines.monthly run --month 2026-06 --force   # an explicit month
```

The skip path typically takes ~1–2 minutes (train usually dominates).

### Individual stages

```bash
.venv/bin/python -m pipelines.monthly scrape
.venv/bin/python -m pipelines.monthly build_cube
.venv/bin/python -m pipelines.monthly aggregate
.venv/bin/python -m pipelines.monthly features
.venv/bin/python -m pipelines.monthly train
.venv/bin/python -m pipelines.monthly evaluate
.venv/bin/python -m pipelines.monthly predict
.venv/bin/python -m pipelines.monthly publish
```

Individual stage subcommands run for the **current** tick month with no
idempotency guard (handy for debugging a single stage in place).

Each of these monthly-CLI subcommands takes **no options** other than
`-h`. They call the corresponding `run_<stage>()` with default
arguments. (To pass scrape flags, use the standalone scrape module —
see below.) Each stage prints structured `INFO`-level logs and the CLI
exits non-zero on any stage failure.

---

## What each stage does

### `scrape` — `pipelines.monthly.scrape`

Subprocesses each retail scraper sequentially, writing the immutable
per-month raw landing zone (within-month re-runs overwrite that month):

1. `gap_scraper.py`     → `data/raw/items/items_gap_<YYYY-MM>.csv`
2. `uniqlo_scraper.py`  → `data/raw/items/items_uniqlo_<YYYY-MM>.csv`
3. `american_eagle_scraper.py` → `data/raw/items/items_american_eagle_<YYYY-MM>.csv`
4. `hollister_scraper.py` → `data/raw/items/items_hollister_<YYYY-MM>.csv`

Building the live cubes is the **separate `build_cube` stage** below — it used
to run inside `scrape`.

**Flags (standalone module only).** The monthly-CLI `scrape` subcommand
above takes no options. To control the scrapers, invoke the module
directly:

```bash
.venv/bin/python -m pipelines.monthly.scrape [flags]
```

- `--retailers gap,uniqlo` — run a subset (default: all four)
- `--no-enrich-pdp` — skip PDP material enrichment for ~3× speedup at
  the cost of ~14% material unknown (default is `--enrich-pdp`)

> Note the dotted path: the flags live on `pipelines.monthly.scrape`
> (the module), **not** on `pipelines.monthly scrape` (the monthly-CLI
> subcommand). The latter ignores everything but `-h`.

### `build_cube` — `pipelines.collectors.build_live_cube`

Unions the discovered `items_*.csv` (immutable monthly files preferred over
legacy via `discover_items_files`), collapses unisex SKUs, and writes one
snapshot per month: `data/processed/live_{fingerprint,univariate}_<YYYY-MM>.parquet`.
Within-month re-runs overwrite that month's cube.

To override inputs/outputs, invoke the module directly with
`--input` / `--signals-dir` / `--output-dir`.

### `aggregate` — `pipelines.monthly.aggregate`

Reads `historical_{fingerprint,univariate}.parquet` + globs every
`live_*_<YYYY-MM>.parquet` + unions the persistent synthetic-prior cube
`backfill_*.parquet` when present (ADR 0002); concatenates with dedup on
`(month, *FINGERPRINT_COLS, source)` (fingerprint) or
`(month, dimension, level_id, source)` (univariate) keeping `last`;
writes the tick's `data/ticks/<YYYY-MM>/merged_{fingerprint,univariate}.parquet`.

Always rebuilds — `historical_*` and `backfill_*` are inputs, never
overwritten. If no live parquets are found, the merged cube is
historical-only.

### `features` — `pipelines.monthly.features`

Builds calendar-strict training rows from the merged cubes. For each
anchor month `t`, requires cube rows on every month in `t-3..t+6` (10
months: 3 lags + anchor + 6 horizons). Rows that don't qualify are
silently dropped. Outputs (into the tick checkpoint):
- `data/ticks/<YYYY-MM>/training_univariate.parquet`
- `data/ticks/<YYYY-MM>/training_fingerprint.parquet`
- `data/ticks/<YYYY-MM>/training_run.json` (feature/target column manifest +
  split/sample-weight contract)

Sample weights = `min(sqrt(n_articles_at_anchor), 100.0)`; split groups
(train/val/holdout) assigned by tail rank on each table's distinct
`anchor_month` values (defaults `K=2` holdout, `J=2` val; the split
adapts downward if there are too few distinct months).

### `train` — `pipelines.monthly.train`

Fits two multi-output `RandomForestRegressor` models (`n_estimators=200`,
`min_samples_leaf=2`, `max_depth=None`, `random_state=42`) — one per
training table. Each predicts the 6-vector `[y_h1, ..., y_h6]`. A
persistence baseline (`ŷ_h = share_t`) is computed as a sanity floor;
a model that doesn't beat baseline on weighted holdout MAE gets a
`WARNING` (it does not block promotion).

Outputs this tick's **candidate** model (never the canonical champion):
- `data/ticks/<YYYY-MM>/model/fingerprint_model.joblib`
- `data/ticks/<YYYY-MM>/model/univariate_model.joblib`
- `data/ticks/<YYYY-MM>/model/model_training_run.json` (metrics + manifest)

### `evaluate` — `pipelines.monthly.evaluate`

Compares this tick's candidate manifest
(`data/ticks/<YYYY-MM>/model/model_training_run.json`) against
`data/models/champion.json` (the cross-tick champion pointer). The
comparison is **per model** (univariate, fingerprint):
- No incumbent recorded → promote
- `candidate.holdout_wmae <= incumbent.holdout_wmae` → promote
- Else → keep that model's reigning champion

**Promote-copy (plan §12).** Because `train` writes the candidate into the
tick (never the canonical `data/models/` joblib), there is **no clobber and no
revert**: on a per-model **win**, evaluate copies the tick's candidate joblib
over `data/models/<role>_model.joblib` (the weights `predict` loads) and
repoints `champion.json[role]` at this month; on a **loss**, the canonical
joblib is left untouched and the reigning champion stays. Decisions are
independent per model. A losing candidate can therefore never reach serving —
`predict` always loads the canonical champion. (Superseded later by Phase 4's
MLflow `champion` alias.)

### `predict` — `pipelines.monthly.predict`

Loads the **canonical champion** joblibs from `data/models/`, iterates the
universe, scores everything that has lag coverage, classifies state via
`pipelines.monthly.state`, decodes IDs to names via `lookup.csv`, validates via
`pipelines/contracts.py`, writes into the tick checkpoint:
- `data/ticks/<YYYY-MM>/predictions_univariate.parquet` (low hundreds of rows)
- `data/ticks/<YYYY-MM>/predictions_fingerprint.parquet` (a few thousand rows)

Exact counts depend on the eligible anchor (the sparse live 2026-05 anchor
yields ~120 univariate / ~3,830 fingerprint rows).

Anchor month is the latest cube month with 3 contiguous prior months,
picked separately per cube (`_find_eligible_anchor`). The persistent
backfill cube (ADR 0002) normally keeps the latest live month eligible;
if it's still isolated, predict logs a `WARNING`, tells you to run
`scripts/backfill_anchor_lags.py` to regenerate the synthetic priors,
and falls back to the nearest eligible (typically historical) anchor.
Individual fingerprints / levels that lack lag coverage at the chosen
anchor are silently skipped.

#### State classification (used by `predict`)

`pipelines.monthly.state.classify_state` is a **forward-first hybrid**
(first match wins), with all thresholds defined as module-level
constants in `state.py`:

1. **peak** — the in-band high (over `{lag1, share_t, h1, h2}`) is a
   real high (strictly above anchor, or tied with anchor while `lag1`
   is below it), drops to `y_h6` by at least `PEAK_MIN_DROP` (0.08)
   relative to the high, and the forward forecast declines
   (`y_h6 < share_t`). → `"at peak"`
2. **rising** — `y_h6 > RISING_RATIO (1.08) × share_t` → `"+N% next 6mo"`
3. **falling** — `y_h6 < FALLING_RATIO (0.92) × share_t` → `"−N% next 6mo"`
4. **flat** — otherwise → `"stable"`

Direction (rising/falling) is decided off the **forward window only**
(`share_t → y_h6`); past lags feed only the peak band. Any non-finite
value, or a zero anchor denominator, classifies as flat/stable.

### `publish` — `pipelines.monthly.publish`

The final stage. Reads the tick's `predictions_*` + `merged_*` + `lookup.csv`
through the shared `pipelines/serving` module (the same code the dev API uses)
and emits browser-ready JSON:
- `data/ticks/<YYYY-MM>/published/{trends,options,health,fingerprint}.json`
  (the immutable checkpoint copy)
- and refreshes `frontend/data/{trends,options,health,fingerprint}.json` — the
  canonical files the static SPA / CDN fetch.

`fingerprint.json` is the single 5-D-keyed bundle the client looks up. The
lag-join (`share_lag*`/`share_t`) is attached here and gated by the golden-file
test (`tests/serving/test_publish.py`).

---

## After the tick

The serving path is **static**: `publish` already wrote the canonical JSON into
`frontend/data/`, so the static SPA (Firebase Hosting / a local static server)
serves the new tick with no process to restart.

The local `scheduleServer` FastAPI app is a dev convenience over the same shared
module — it reads the **latest successful tick** at startup, so restart it to
pick up a new tick:

```bash
bash scripts/run_api.sh   # uvicorn backend.services.scheduleServer:app on :8000
```

It is read-only and makes no live model calls in the request path. Verify via
`/health`:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# Expect: predictions_loaded=true, predictions_anchor_month=<expected month>
# (also reports lags_synthetic=true if the anchor used backfilled priors)
```

---

## Common failure modes

### `RuntimeError: predict: no univariate predictions produced — check cube lag coverage`

**Cause:** the latest cube month has no prior 3 months in the cube
(e.g., a single live month with no historical neighbors).

**Fix:** wait for additional live data to accumulate, OR make sure
`historical_*.parquet` is present so the predict stage can use a
historical anchor month. (If you want the newest live month as the
anchor, run `python scripts/backfill_anchor_lags.py` first.)

### `train` warns "model does NOT beat persistence baseline on holdout"

**Cause:** the model is performing worse than just predicting
`y_h = share_t` for every horizon. With small training data this is
not unusual. (This warning is emitted by the `train` stage, not
`evaluate`.)

**Fix:** investigate via `data/ticks/<YYYY-MM>/model/model_training_run.json`. The
candidate is still promoted by `evaluate` if it beats the *prior*
champion; the baseline warning is informational only.

### `aggregate` complains `missing fingerprint historical at .../historical_fingerprint.parquet — run notebook 1 first.`

**Cause:** notebook 1 hasn't been run on the H&M Kaggle data.

**Fix:** run `notebooks/0_clean_historical.ipynb` then
`notebooks/1_aggregate_historical.ipynb` from the trndly/ directory.
These produce the immutable historical cubes; only needs to happen
once.

### Scraper fails with 403 / 401 / Akamai mention

See `pipelines/collectors/README.md` — the "Known limitations",
"Operational notes", and per-retailer (American Eagle / Hollister)
sections. Most common fixes:
- Re-run American Eagle's Playwright bootstrap to refresh the JWT
- Wait ~30 minutes (Akamai cooldown)
- Run with `--retailers` (on the standalone
  `pipelines.monthly.scrape` module) to skip the offending retailer for
  now

---

## What to do if WMAE regresses

If `evaluate` keeps a model this month (the candidate lost), the candidate is
worse than the reigning champion. Investigate by comparing this tick's candidate
manifest `data/ticks/<YYYY-MM>/model/model_training_run.json` against
`data/models/champion.json` (which records the champion's month + holdout WMAE):

- Did the new live month introduce noisy outliers?
- Did `feature_importances_` shift drastically?
- Is the holdout split still representative of recent months?

There is nothing to recover: the promote-copy flow means a losing candidate
never overwrites the canonical `data/models/<role>_model.joblib`, so the reigning
champion keeps serving. The losing candidate's weights remain archived in
`data/ticks/<YYYY-MM>/model/` for inspection. Decisions are per model, so one
model can keep while its sibling promotes. Document notable regressions in the
repo-root `TODO.md` (`../TODO.md` relative to `trndly/`).

---

## MVP vs. target (current limitations)

The monthly tick is the **local-MVP**. The following are deliberately
deferred to cloud deployment and are NOT how the tick works today:

- **Champion management.** *Now:* per-tick model isolation + promote-copy
  (`data/models/champion.json`; the winning candidate joblib is copied over the
  canonical `data/models/<role>_model.joblib` on a win, no clobber/revert since
  `train` writes only the tick candidate). *Target:* MLflow model-registry alias
  (`MlflowClient.set_registered_model_alias(name=..., alias='champion',
  version=...)`) against the rebuilt private MLflow registry (Cloud Run +
  Cloud SQL + GCS, see `serving-redesign.md`).
- **Anchor lag backfill.** *Now:* the persistent backfill cube
  (`data/processed/backfill_*.parquet`, generated once by
  `scripts/backfill_anchor_lags.py` and unioned in by `aggregate` —
  ADR 0002). Synthetic data by design; *target:* delete it once live
  scrapes reach ≥4 contiguous months (~2026-08).
- **Scraper parallelism.** *Now:* sequential subprocess calls
  (gap → uniqlo → american_eagle → hollister), ~9–10 min with PDP
  enrichment. *Target:* parallel dispatch (blocked today by each
  scraper owning its own `asyncio.run`).
- **Storage / cadence.** *Now:* local parquet, manual CLI invocation
  from a dev machine (serving itself is already static + cloud: Firebase
  Hosting, with Terraform IaC in `infra/`). *Target:* all data in GCS +
  a scheduled, idempotent Cloud Run Job for the tick — Phase 6
  (PROPOSED, gated on ADR 0001).
