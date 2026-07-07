# TODO

Forward-looking work list for the trndly forecaster pipeline.
Last updated: 2026-07-07 (audited 2026-07-07).

For the shipped state, read [README.md](README.md) and
[trndly/docs/architecture.md](trndly/docs/architecture.md). For
recent landmark commits, run `git log --oneline -20`.

---

## Active

*Empty — pinned items below are next-up candidates when you reprioritize.*

---

## Pinned (out of scope until reprioritized)

### Cloud deployment (remaining target work)

Serving is live and static (Firebase Hosting + CDN — `trndly.web.app`),
MLflow runs private on Cloud Run + Cloud SQL + GCS, and all infra is
Terraform (`trndly/infra/`): Phases 0–3 of the build plan
(`trndly/docs/serving-redesign.md`) shipped 2026-06-24. The tick itself
is still laptop-driven. What remains:

- **MLflow registry wiring (Phase 4).** `evaluate.py` still uses the
  local `champion.json` promote-copy; target is
  `MlflowClient.set_registered_model_alias` against the live private
  MLflow, with `train` logging model versions.
- **Auth + persistent inventory (Phase 5).** Firebase Auth + Firestore;
  `frontend/auth.js` is a demo stub, inventory is session state.
- **Cloud-native tick (Phase 6 — PROPOSED, review first).** Scheduled,
  idempotent tick on a Cloud Run Job; all data in GCS (`paths.py` is the
  single chokepoint — pure local `Path` logic today, no `fsspec`/`gs://`
  resolver; `gcsfs` already in `requirements.txt`). Requires accepting
  ADR 0001 (`trndly/docs/decisions/0001-cloud-tick-cdn-refresh.md`).

### Univariate `dimension` feature

Univariate model is currently dimension-blind (features:
`[month_of_year, share_t, share_lag1..3]`, confirmed in
`features.py::UNIVARIATE_FEATURE_COLS`). Adding `dimension` as a
pandas Categorical lets the model specialize per dim (color seasonality
≠ material seasonality) without splitting into N models. Touchpoints:
`pipelines/monthly/features.py` (add column to
`UNIVARIATE_FEATURE_COLS` + `training_run.json`),
`pipelines/cube_slicing.py::build_univariate_inference_row` (emit the
column), `pipelines/monthly/predict.py` (passes through).

### State-classifier threshold tuning

`pipelines/monthly/state.py` was rewritten in 2026-05 to a forward-first
hybrid rule (peak band considers past lags + anchor + first 2 forward
horizons; rising/falling decided on the forward ratio `y_h6 / share_t`).
Current constants (verified module-level in `state.py`):

- `RISING_RATIO = 1.08` — forward must beat anchor by >8% to fire rising
- `FALLING_RATIO = 0.92` — forward must trail anchor by >8% to fire falling
- `PEAK_MIN_DROP = 0.08` — peak must drop ≥8% to its forward end to fire

The remaining work: validate against real-distribution histograms now
that we have a stable 2026-05 anchor; consider seasonality-aware variants
("rising for time of year"). The numeric thresholds may need re-tuning
once enough live months accumulate to evaluate against held-out data.

### Frontend fingerprint synthesis quality

`frontend/api.js::synthesizeFingerprintSeries` produces a joint forecast
by multiplying per-dimension relative motions when the fingerprint lookup
misses (a `fingerprint.json` miss in static mode; a 404 on the dev API).
This is a multiplicative-independence approximation — fine for many
cases but doesn't capture cross-dimension correlations (some
materials/types co-occur more than independence predicts).

Possible follow-ups:
- Share-weight the factors so tiny-share dimensions (e.g. Blazer at
  0.0002) contribute less than dominant ones (Women at 0.54).
- Expand `pipelines/monthly/predict.py` to compute predictions for the
  full Cartesian product, not just observed combinations. ~3.77M rows
  if done naively — needs filtering down to plausible combos.
- Trail real fingerprint forecasts vs. synthesized for combos where both
  exist, to quantify error.

Not blocking; the chart legend labels synthesized series clearly
("We've never seen this item before!").

### Auto-rebootstrap AE on 401

American Eagle's Akamai JWT has ~30-min TTL.
[american_eagle_scraper.py](trndly/pipelines/collectors/american_eagle_scraper.py)
should detect a 401 mid-run and re-invoke `_bootstrap_session` instead
of failing the whole scrape. Today there is no 401 detection in the
fetch loop, so the user has to re-run from scratch.

### Local venv Python version

`pytest-asyncio` is in `requirements.txt` and CI (Python 3.11) is
green. The on-disk `.venv` is Python 3.14, though — the supported
version is 3.11 (`scripts/setup_venv.sh` and CI both pin it). Rebuild
the venv via `scripts/setup_venv.sh` if local runs misbehave.

---

## Brittle areas (carry from previous handoffs — still apply)

In rough order of "most likely to break first":

### AE Akamai fingerprint check (HIGH)

[american_eagle_scraper.py](trndly/pipelines/collectors/american_eagle_scraper.py)
requires a one-time Playwright bootstrap that captures the **full set**
of browser headers (`sec-ch-ua-*`, `sec-fetch-*`, `aesite`, `aelang`,
`channeltype`, `Authorization: Bearer <JWT>`). Captured headers pin to
whatever Chrome version Playwright is running — if Akamai later
validates against a *current* Chrome, you'll get silent 403s on httpx.

**Detection:** `Phase 1` log shows `[http] api ... got 403, retry ...` spam.
**Fix:** re-run Playwright bootstrap with a Chrome devtools network
capture, diff request headers, update `STATIC_API_HEADERS`.

### Hollister TLS/HTTP fingerprint (MEDIUM)

[hollister_scraper.py](trndly/pipelines/collectors/hollister_scraper.py)
only works because plain `httpx` over HTTP/1.1 happens to satisfy
Akamai's edge fingerprint. If `httpx` changes its default TLS handshake
or someone "improves" the client to use HTTP/2, the scraper silently
dies.

**Detection:** `productTotalCount=0 totalPages=0` and a 149-byte response
body. Caught by `pytest -m live`'s Hollister structural sanity check.

### Hollister Apollo-state parsing

Catalog data lives inside
`window['APOLLO_STATE__catalog-mfe-web-service-CategoryPageFrontEnd-config'] = {...}`
in the SSR HTML. If Hollister renames the variable or wraps it
differently, parsing returns `None` and Hollister's items file becomes
empty. Constant: `APOLLO_STATE_PREFIX` at the top of
[hollister_scraper.py](trndly/pipelines/collectors/hollister_scraper.py)
(consumed by `_parse_apollo_state`).

### PDP fabric regex per retailer

Each retailer's PDP fabric extraction depends on a regex matching
specific JSON-string structure:

| Retailer  | Pattern | Lives in |
| --------- | ------- | -------- |
| Gap       | `\\"label\\":\\"Fabric \\u0026 care\\".*?\\"bullets\\":\[(.*?)\]` | `gap_scraper.py:FABRIC_BULLETS_RE` |
| Uniqlo    | `"composition"\s*:\s*"((?:.\|[^"])*)"` | `uniqlo_scraper.py` |
| AE        | JSON path `data["data"]["attributes"]["copySections"]["material"]["bullets"]` | `american_eagle_scraper.py:_fetch_pdp_fabric` |
| Hollister | `"fabricDetails":"((?:[^"\\]\|\\.)*)"` | `hollister_scraper.py:PDP_FABRIC_RE` |

If any retailer changes their PDP serialization, enrichment silently
returns empty strings → `material_raw` unknown rate jumps from ~2% to
~14%.

### `feature_lookups.py` ID drift

The validator catches drift between the hand-typed `*_TO_ID` dicts and
`data/reference/lookup.csv` at module import. **If you edit either, the
import will raise — fix the diff before the scrapers can run.**
Negative test in
[tests/test_trndly.py::test_lookup_consistency_validator_detects_drift](trndly/tests/test_trndly.py).

### Sparse cube — synthetic anchor priors (persistent backfill cube)

`pipelines/monthly/predict.py` requires 4 contiguous months (anchor +
3 lags), but the data has a ~5-year gap: historical (2018-10 → 2020-08)
vs live (2026-05 →). Per
[ADR 0002](trndly/docs/decisions/0002-persistent-backfill-cube.md),
[scripts/backfill_anchor_lags.py](trndly/scripts/backfill_anchor_lags.py)
generated synthetic 2026-02/03/04 rows **once** into
`data/processed/backfill_*.parquet` (seasonal ratios from 2019+2020,
rescaled to the live anchor), and `aggregate` unions them into every
tick's merged cube — no per-tick re-run, nothing to clobber. Synthetic
rows stay traceable (`source='backfill'`; serving exposes
`lags_synthetic: true` and the chart legend footnotes it).

**Remove when** live scrapes reach ≥4 contiguous months (~2026-08 at
monthly cadence): delete `backfill_*.parquet` and the script; predict
then anchors on real lag history.

---

## Useful pointers

### Conventions

- **Cwd matters.** All Python invocations expect `trndly/` as the
  working directory (the inner one). The monthly CLI's `python -m
  pipelines.monthly` resolves imports off cwd; running from the project
  root will fail with `ModuleNotFoundError: pipelines`.
- **Python interpreter.** `trndly/.venv/bin/python` is the supported
  env, built from `trndly/requirements.txt` via `scripts/setup_venv.sh`.
  The supported version is **Python 3.11** (matches CI); the venv
  currently on disk is 3.14 — see "Local venv Python version" above.

### Smoke commands

```bash
cd /Users/jackcdawson/Desktop/trndly/trndly

# Full monthly tick (scrape → build_cube → aggregate → features → train →
# evaluate → predict → publish). ~15 min including scrape. Writes data/ticks/<YYYY-MM>/.
.venv/bin/python -m pipelines.monthly run

# Skip scrape stage (use existing items_*.csv). ~1 min.
.venv/bin/python -m pipelines.monthly run --skip-scrape

# Single retailer
.venv/bin/python pipelines/collectors/gap_scraper.py

# All 4 scrapers + build_live_cube (replaces old run_all.sh)
.venv/bin/python -m pipelines.monthly scrape

# Just the merge stage (rebuild merged_*.parquet from cubes on disk)
.venv/bin/python -m pipelines.monthly aggregate

# Test integrity (296 collected; 3 live network tests deselected)
.venv/bin/python -m pytest tests/ -q

# Serve the API
.venv/bin/python -m uvicorn backend.services.scheduleServer:app --port 8000
```

### Where to look when X fails

| Symptom | Most likely cause | Where to look |
| ------- | ----------------- | ------------- |
| `ModuleNotFoundError: No module named 'pipelines'` | Wrong cwd | `cd trndly/` first |
| `ValueError: feature_lookups.py drift vs ...` at import | Hand-edited `*_TO_ID` dict diverged from `data/reference/lookup.csv` | The diff in the error message |
| Hollister `productTotalCount=0` | TLS fingerprint changed (or HTML rewritten) | `hollister_scraper.py:_parse_apollo_state` |
| AE 100% 403s | Akamai tightened or Playwright Chromium too old | `american_eagle_scraper.py:_bootstrap_session` |
| Material unknowns spike to ~14% | PDP fabric regex broke (retailer changed PDP HTML) | The `*_RE` constants in each scraper |
| Live cube share-sums fail invariant | `build_live_cube.py` upstream got NaN IDs | `validate_live_*_frame` in `pipelines/contracts.py` raises with details |
| `/options` returns empty arrays | `data/reference/lookup.csv` missing or wrong category | Check `lookup.csv` `category` column values |
| `/trends` returns `[]` | No predictions parquet, or anchor month has no rows | Run `python -m pipelines.monthly predict`; restart API |
| API returns 503 with "predictions bundle not loaded" | No predictions parquet found at startup | `ls data/ticks/*/predictions_*.parquet`; run the monthly tick |
| `predict` exits "no univariate predictions produced" | Latest cube month has no 3 contiguous prior months | See "Sparse cube → empty predictions" above |
| `async def functions are not natively supported` | venv missing `pytest-asyncio` or on the wrong Python | Rebuild the venv on 3.11 via `scripts/setup_venv.sh` |

### Documentation

- [README.md](README.md) — entry point, repo layout, quick start
- [trndly/docs/architecture.md](trndly/docs/architecture.md) — full
  architecture (shipped + future)
- [trndly/docs/api.md](trndly/docs/api.md) — endpoint reference
- [trndly/docs/monthly_tick.md](trndly/docs/monthly_tick.md) — operator
  runbook
- [trndly/docs/rationale.md](trndly/docs/rationale.md) — design
  decisions
- [trndly/pipelines/collectors/README.md](trndly/pipelines/collectors/README.md)
  — scrapers, items.csv schema, brittle areas
- [trndly/data/reference/SCHEMA.md](trndly/data/reference/SCHEMA.md) —
  per-dimension reachability audit (lookup vs. historical vs. live vs.
  merged) plus the deliberately-unreachable allow-list rationale
- [trndly/data/reference/lookup.csv](trndly/data/reference/lookup.csv)
  — canonical feature universe; `*_TO_ID` dicts validated against it
  at import
