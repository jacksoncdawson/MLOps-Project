# trndly — Architecture

trndly is a fashion trend forecasting platform that helps secondhand
apparel resellers list and source inventory at the right time. Two
RandomForest regressors produce 6-horizon catalog-share forecasts:

- **Univariate model** — one row per `(dimension, level_id)`. Trains on
per-feature time series (every color, every material, every product
type, etc.) at the cube grain. Used for trend exploration ("which
colors will be rising in 6 months?").
- **Fingerprint model** — one row per 5-D fingerprint
`(product_type_id, gender_id, color_master_id, graphical_appearance_id, material_id)`. Used for per-item recommendations ("when should I list
this specific blazer?").

This document describes the **shipped** architecture. A `Future` section
at the bottom captures the GCP target we're working toward.

---

## Data flow (shipped)

```
                ┌──────────────────────────────────────────────┐
                │  pipelines/collectors/                       │
                │    {gap,uniqlo,american_eagle,hollister}_    │
                │  scraper.py                                │
                │      └─► data/raw/items/items_<retailer>_<YYYY-MM>.csv
                │  build_live_cube.py                          │
                │      └─► data/processed/live_*_<YYYY-MM>.parquet
                └──────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────────────┐
                │  pipelines/monthly/  (the monthly tick)      │
                │    scrape    — runs collectors + build_live  │
                │    aggregate — historical + live → merged_*  │
                │    features  — calendar-strict windows       │
                │    train     — RF fit, write joblibs         │
                │    evaluate  — promote-copy winner → models/ │
                │    predict   — score universe → ticks/<M>/   │
                │    publish   — emit JSON → frontend/data/    │
                │    cli       — `python -m pipelines.monthly run`
                └──────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────────────┐
                │  data/ticks/<YYYY-MM>/ (immutable checkpoint)│
                │    predictions_{univariate,fingerprint}.parquet
                │    published/*.json  → frontend/data/*.json  │
                └──────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────────────┐
                │  Firebase Hosting (CDN)  — trndly.web.app    │
                │    SPA + frontend/data/*.json, same-origin   │
                │    (deployed via `firebase deploy` / CI)     │
                └──────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────────────┐
                │  frontend/  (React + JSX-via-Babel)          │
                │    Trends screen      ← trends.json          │
                │    Add Item screen    ← options.json         │
                │    Item Detail screen ← fingerprint.json     │
                │                         (client 5-D lookup)  │
                │    Inventory          (session state)        │
                └──────────────────────────────────────────────┘
```

Inference is **precomputed monthly** and serving is **static**: the
`publish` stage emits browser-ready JSON, and there is no server — and no
live `model.predict()` call — in the request path. For local development,
`backend/services/scheduleServer.py` (FastAPI) serves the same data over
HTTP endpoints and mounts the SPA at `/ui/` (see
[api.md](api.md)); it imports the same shared `pipelines/serving/`
module as `publish`, so the two paths can't drift apart silently.

---

## Repo layout (the bits worth knowing)

```
trndly/
├── pipelines/
│   ├── paths.py            — central path registry (the chokepoint)
│   ├── contracts.py        — schema validators (live cubes + predictions)
│   ├── cube_slicing.py     — shared cube → feature-row helpers
│   ├── collectors/         — 4 retail scrapers + build_live_cube.py
│   ├── serving/            — shared lag-join + schemas (publish + dev API import this)
│   ├── monthly/            — the monthly tick (scrape→build_cube→...→predict→publish)
│   │   ├── scrape.py
│   │   ├── aggregate.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── state.py        — trend-state classifier
│   │   ├── predict.py
│   │   ├── publish.py      — emits browser-ready JSON
│   │   └── cli.py          — `python -m pipelines.monthly`
│   └── ...
├── backend/services/
│   └── scheduleServer.py   — slimmed dev API over pipelines/serving (no .env)
├── frontend/               — React SPA, no build step (fetches data/*.json)
├── notebooks/              — 0 (Kaggle clean), 1 (historical agg), 4 (HP sweep)
├── tests/
└── data/
    ├── raw/{items,kaggle}/ — items_<retailer>_<YYYY-MM>.csv
    ├── reference/          — lookup.csv, SCHEMA.md
    ├── processed/          — historical_* / live_*_<YYYY-MM> (shared cube inputs)
    ├── models/             — fingerprint_model.joblib, univariate_model.joblib,
    │                         champion.json   (cross-tick CHAMPION)
    └── ticks/<YYYY-MM>/    — immutable per-tick checkpoint (merged/training/model/
                              predictions/published + manifest.json + _SUCCESS)
```

---

## The monthly tick (`python -m pipelines.monthly run`)

Stages in order (`pipelines/monthly/cli.py`, `FULL_ORDER`):


Every per-tick artifact lands in an immutable checkpoint `data/ticks/<YYYY-MM>/`
(plan §12); `historical_*`/`live_*` stay in `data/processed/` as shared inputs,
the cross-tick champion in `data/models/`.

| #   | Stage        | Inputs                                            | Outputs                                                                                       |
| --- | ------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | `scrape`     | retailer APIs                                     | `data/raw/items/items_<retailer>_<YYYY-MM>.csv`                                                |
| 2   | `build_cube` | `items_*.csv`                                     | `data/processed/live_*_<YYYY-MM>.parquet`                                                      |
| 3   | `aggregate`  | historical + live cubes                           | `data/ticks/<YYYY-MM>/merged_*.parquet`                                                        |
| 4   | `features`   | tick merged cubes                                 | `data/ticks/<YYYY-MM>/training_*.parquet`, `training_run.json`                                 |
| 5   | `train`      | tick training tables                              | `data/ticks/<YYYY-MM>/model/*.joblib` + `model_training_run.json` (this tick's **candidate**)  |
| 6   | `evaluate`   | candidate manifest + `data/models/champion.json`  | per-model promote-copy: winner's joblib → `data/models/`, repoint `champion.json`              |
| 7   | `predict`    | canonical champion + tick merged + `lookup.csv`   | `data/ticks/<YYYY-MM>/predictions_*.parquet` (state classification baked in)                   |
| 8   | `publish`    | tick predictions + merged + `lookup.csv`          | `data/ticks/<YYYY-MM>/published/*.json` + refreshed `frontend/data/*.json`                     |


Stages can be invoked individually:

```bash
python -m pipelines.monthly aggregate
python -m pipelines.monthly run --skip-scrape   # use existing items_*.csv
```

**Cadence:** manual for now. The CLI is the one place that drives the
chain. Cloud Scheduler / Vertex AI wiring is in the `Future` section.

**Model:** each model is a multi-output `RandomForestRegressor`
(`n_estimators=200`, `min_samples_leaf=2`, `max_depth=None`,
`random_state=42`) predicting `y_h1..y_h6`. A persistence baseline
(ŷ_h = `share_t`) is computed as a sanity floor. `train.py` writes the
candidate joblibs into `data/ticks/<YYYY-MM>/model/` — it never touches the
canonical `data/models/` champion. `predict.py` loads the **canonical champion**
joblibs from `data/models/`.

**Promotion rule** (`evaluate.py`, the *local-MVP* champion): for each model
independently, if candidate `holdout_wmae <= incumbent.holdout_wmae`, promote.
With no incumbent recorded, the candidate is promoted; on a tie the candidate
wins. **Promote-copy (plan §12):** on a win, evaluate copies the tick's candidate
joblib over `data/models/<role>_model.joblib` and repoints `data/models/champion.json[role]`
at this month; on a loss it does nothing (the reigning champion keeps serving).
Because `train` never clobbers the canonical joblib, there is **no revert** — a
losing candidate can never reach serving. Target state: an MLflow registry
`champion` alias against the rebuilt private MLflow.

**Trend state classification** (`pipelines/monthly/state.py`) — a
forward-first hybrid mapping a `(past3 lags + anchor + 6 forward)`
trajectory to `rising | peak | flat | falling`. The rising/falling
verdict is decided off the **forward window only** (`share_t → y_h6`);
past lags feed only the peak detector. Rules are checked in this order
(first match wins):

- `peak` (checked first): the in-band high — over the band
  `{share_lag1, share_t, y_h1, y_h2}` (`share_lag1` only when past lags
  are supplied) — is a *real* high (strictly above `share_t`, or tied
  with `share_lag1` below `share_t`), **and** the drop from that high to
  `y_h6` is at least `PEAK_MIN_DROP` of the high, **and** the forecast
  declines from anchor (`y_h6 < share_t`).
- `rising`: `y_h6 > RISING_RATIO × share_t`
- `falling`: `y_h6 < FALLING_RATIO × share_t`
- `flat`: otherwise. Any non-finite value or zero denominator → `flat`.

Module-level constants:

```python
RISING_RATIO  = 1.08
FALLING_RATIO = 0.92
PEAK_MIN_DROP = 0.08
```

These thresholds are a deliberate design choice (the previous iteration
used an end-to-end lag3→h6 ratio, which conflated past growth with
forecast direction), not unreviewed placeholders. They can still be
re-tuned as more live months accrue.

---

## Predictions parquet (binding contract)

Two parquets per monthly tick. Schema validators in `pipelines/contracts.py`
(`validate_predictions_univariate_frame`, `validate_predictions_fingerprint_frame`).
The validators enforce column presence + order, no nulls in any column,
`state ∈ {rising, peak, flat, falling}`, and non-empty `stat` strings.

**`predictions_univariate_<YYYY-MM>.parquet`** — 13 columns:

```
anchor_month, model_version,
dimension, level_id, level_name,
y_h1, y_h2, y_h3, y_h4, y_h5, y_h6,
state, stat
```

**`predictions_fingerprint_<YYYY-MM>.parquet`** — 20 columns:

```
anchor_month, model_version,
product_type_id, gender_id, color_master_id, graphical_appearance_id, material_id,
product_type_name, gender_name, color_master_name, graphical_appearance_name, material_name,
y_h1, ..., y_h6,
state, stat
```

`predict.py` scores the entire universe at the latest *eligible* anchor
(`_find_eligible_anchor`: the latest month with 3 contiguous prior
months in the merged cube). Rows where the cube lacks t-3..t-1 lag
coverage at that anchor are silently skipped — the predictions parquet
only contains rows for which a forecast was producible. When the latest
live month is isolated (no real priors), `scripts/backfill_anchor_lags.py`
is a manual stopgap that synthesizes seasonal priors so the live month
becomes eligible.

---

## Local dev API surface

`backend/services/scheduleServer.py` — the local development server (not
in the production request path; see [api.md](api.md) for full endpoint
docs). All routes are GET; no request triggers a model call.


| Route                       | Behavior                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `GET /`                     | redirect → `/ui/`                                                                                                  |
| `GET /health`               | bundle status: `predictions_loaded`, `predictions_anchor_month`, row counts, `lags_synthetic`                      |
| `GET /options`              | dropdown vocabularies (read live from `lookup.csv`), `[{name, id}]` per category (`colors`, `categories`, `materials`, `appearances`, `genders`) |
| `GET /trends`               | every univariate prediction row. Optional filters `?dimension=&state=`                                             |
| `GET /forecast/fingerprint` | one fingerprint forecast. Query: 5 `*_id` ints. 404 if no precomputed match                                        |
| `/ui/*`                     | static React app (`trndly/frontend/`)                                                                              |


Swagger UI at `/docs`. OpenAPI JSON at `/openapi.json`.

`/options` is read directly from `lookup.csv` on each request (so the UI
stays in sync with the canonical ID universe). The `BUNDLE` global — the
two predictions frames plus the joined lag shares — is loaded at startup
via the lifespan hook. At load time the service also joins observed
shares at the anchor and its three prior months (`share_lag3`,
`share_lag2`, `share_lag1`, `share_t`) from `merged_*.parquet`, and sets
`BUNDLE.lags_synthetic` if any of those lag months came from the backfill
stopgap (surfaced via `/health`). Restart the container (or re-run
`pipelines.monthly predict` then restart) to refresh.

---

## Frontend

Stack: **React 18** + **JSX-via-Babel** (no build step). React, ReactDOM,
and `@babel/standalone` load from unpkg via `<script>` tags in
`index.html`; application files load as `<script type="text/babel">`.
Data fetching is a tiny in-house `useFetch` hook inside `dataProvider.js`
— keyed module-level cache, optional poll, optional refocus revalidation.
SWR was the original choice (see [rationale.md](rationale.md)), but SWR
2.x ships ESM only with no UMD bundle, which doesn't fit the no-build
setup; the in-house hook covers what we actually use in ~50 lines.

`frontend/dataProvider.js` provides a single `useData()` hook exposing:

- `inventory`, `signals`, `addItem` — session-scoped local state (start empty;
  no seeded mocks)
- `trends` — `trends.json` (dev API: `/trends`). `undefined` while loading/errored; screens
  render explicit loading/error states instead of silently substituting
  fixtures. Each row carries 10 numeric points (`share_lag3..t` joined
  from the merged cube at service startup, then `y_h1..y_h6`) so charts
  draw 3 months of history + 6 months of forecast. The `api.js` reshape
  drops `Unknown` rows (every dimension reserves `id=0 = Unknown` for
  unclassified items — surfacing those in the UI isn't actionable).
- `options` / `lookupIds` — `options.json` (dev API: `/options`), falling back to the
  `LOOKUP_OPTIONS` seed in `data.js` only for `colorSpectrum` /
  `productGroup` (the two vocabularies the endpoint doesn't expose yet).
- `health` — `health.json` (dev API: `/health`; 15s poll, revalidate-on-focus); drives the
  sidebar API status pill + the chart-legend "synthetic past" footnote
  when `lags_synthetic` is set.

`api.js` fetches the published static JSON by default (relative
`./data/*.json` paths, same-origin on Firebase Hosting or the dev
server's `/ui/` mount) and reshapes it (`mapTrendsToTrendData`,
`mapOptionsToLookupOptions`, `indexOptionsById`) into the shapes the
screens consume. Setting `window.API_BASE` repoints the fetches at a
live `scheduleServer` — the response shapes are identical, so screens
don't care which mode they're in.

### Item recommendation pipeline

Per-item recommendations (the pill on Item Detail) are derived directly
from a single 10-point series — the same series the Overall Popularity
chart shows. Source priority:

1. `GET /forecast/fingerprint` — gold standard for 5-D combinations that
   are in the precomputed cube.
2. `synthesizeFingerprintSeries(tags, trends)` in `api.js` — multiplicative
   joint of per-dimension univariate motions. Fires when the fingerprint
   endpoint 404s (combination not precomputed). UI labels the chart
   "We've never seen this item before! Predicting based on this item's
   distinct characteristics."
3. `null` — chart hidden; pill reads "More data needed".

`deriveRecommendationFromSeries(series)` in `data.js` then returns one of
five outcomes (`list now / hold 1mo / hold 2mo / hold 3+ / more data needed`)
based on the argmax in the forward window vs. anchor, with a 2.5% minimum
upside threshold (`UPSIDE_THRESHOLD = 0.025`). Per-feature signal cards on
Item Detail are labels only — they do not aggregate into the
recommendation, but they ARE the data source the synthesis step consumes.

### Trend label vocabulary vs. recommendation vocabulary

Two separate vocabularies, two purposes:

- **Trend label** (`rising / peak / flat / falling`): per-feature direction
  classified by [pipelines/monthly/state.py](../pipelines/monthly/state.py)
  on the univariate forward window. Shown on Trends cards + Item Detail
  signal cards.
- **Recommendation outcome** (`list now / hold 1mo / hold 2mo / hold 3+ /
  more data needed`): item-level action derived from
  `deriveRecommendationFromSeries` against the fingerprint/synthesized
  series. Shown on Item Detail pill + Inventory grouping.

The two never coerce into each other.

---

## Testing

The suite is **296 collected tests** (293 run by default; 3 `live`
network tests are deselected) across `tests/scrapers/`,
`tests/monthly/`, `tests/serving/` (the publish golden-file + dev-server
parity tests), and the root (`test_paths.py` + `test_trndly.py`).
Collected count exceeds function count because of parametrization
(`test_feature_lookups.py` and `test_state.py` dominate). `pytest.ini`
sets `addopts = -m "not live"` (live retailer tests opt-in via
`pytest -m live`) and `asyncio_mode = strict`.

**CI:** `.github/workflows/tests.yml` (at the **repo root**, one level
above the `trndly/` package) runs `pytest tests -v --junitxml=pytest-report.xml`
on every push and PR to `main`, plus `workflow_dispatch`, on Python 3.11
with `working-directory: trndly`, and uploads the junit report as an
artifact. Tests are gated — a non-zero exit blocks the run.


| Layer                 | Where                    | Notes                                                    |
| --------------------- | ------------------------ | -------------------------------------------------------- |
| Schema validators     | `pipelines/contracts.py` | Called inside producers + tests                          |
| Path-existence        | `tests/test_paths.py`    | Parametric over every `Path` constant                    |
| Lookup consistency    | `tests/test_trndly.py`   | feature_lookups dicts vs lookup.csv                      |
| Items CSV ID validity | `tests/test_trndly.py`   | scraper outputs vs lookup.csv                            |
| Live cube validators  | `tests/test_trndly.py`   | concat-compatibility with historical                     |
| Tick unit tests       | `tests/monthly/`         | state classifier, evaluate logic, predict E2E            |
| Scrapers              | `tests/scrapers/`        | Mock-based; some require `pytest-asyncio`/`pytest-httpx` |
| Publish + dev API     | `tests/serving/`         | Golden-file diff on published JSON (the lag-join gate) + server-parity tests |


---

## Future (target architecture, not yet shipped)

Items below are deliberately out of scope for the current MVP. The shipped
architecture is designed to swap each piece in without restructuring.

### Storage migration: local parquet → GCS / BigQuery

`paths.py` is the single chokepoint. Migration adds a backend abstraction
(e.g., `fsspec`-resolved paths or a `gs://`-aware helper) without touching
consumer code. `gcsfs` is already a transitive dependency. Illustrative target bucket
layout (buckets are provisioned via Terraform per the build plan,
[serving-redesign.md](serving-redesign.md)):

- `gs://<data-bucket>/ticks/<YYYY-MM>/` — immutable per-tick checkpoint (predictions + `published/`), per plan §12.5
- `gs://<data-bucket>/processed/` — shared cube inputs (`historical_*` / `live_*`)

### Cloud cadence: manual CLI → scheduled cloud tick (Phase 6, PROPOSED)

Replace `python -m pipelines.monthly run` with an idempotent Cloud Run
Job fired by Cloud Scheduler, with all data in GCS — see
[phase6-cloud-native-tick.md](phase6-cloud-native-tick.md); the serving
refresh is gated on [ADR 0001](decisions/0001-cloud-tick-cdn-refresh.md).
The CLI's stage order doesn't change.

### MLflow registry: local champion file → managed champion alias (Phase 4)

The **private MLflow is live** (Cloud Run + Cloud SQL + GCS artifacts,
IAM-gated — built in Phase 3 of [serving-redesign.md](serving-redesign.md);
the compromised dev-era server it replaces was retired and destroyed).
What is **not** yet wired up:

- The monthly tick's champion management. `evaluate.py` is explicitly the
  local-MVP version — the champion is the local `data/models/champion.json`
  pointer + canonical joblibs (per-tick model isolation + promote-copy, no
  revert; plan §12), not a registry alias. The target is
  `MlflowClient.set_registered_model_alias(name=..., alias='champion',
  version=...)` against the rebuilt private MLflow registry. Plumbing
  notes live in `pipelines/monthly/evaluate.py`.
- The serving path. The old `backend/services/.env` (leftover `MLFLOW_*`
  vars + a since-revoked key, from an older registry-backed serving design)
  has been **deleted** as part of the §2.5 incident remediation; it was never
  referenced by `scheduleServer.py`. Serving reads precomputed parquet/JSON
  only — the static publisher's output, with no compute behind it.

### Auth: none → Firebase Auth + Firestore (Phase 5)

Firebase Auth replaces the demo stub (`frontend/auth.js` — `login()`
always succeeds and returns a hardcoded demo user), and inventory
becomes per-user in Firestore (instead of session state), with security
rules scoping every read/write to the signed-in `uid`.

### Containers: the broken serving Dockerfile is gone

The old `trndly/Dockerfile` shipped the FastAPI server (and a `COPY` of a
non-existent `pipelines/training` dir). With serving now static (above), it
served no purpose and was **deleted** (with its orphaned `.dockerignore`) in
Phase 2. The only container the build defines is the **private MLflow image**
(`infra/mlflow/Dockerfile.mlflow`, Phase 3 — built via Cloud Build → Artifact
Registry, run on Cloud Run). If the monthly tick is ever automated unattended,
it would run as a **Cloud Run Job** from its own image (a clean follow-on, plan
§11), but the tick has no container today — it runs as a local CLI.