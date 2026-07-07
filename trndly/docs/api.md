# trndly API reference (local dev server)

**Production serving is static** — the tick's `publish` stage emits
browser-ready JSON that ships to Firebase Hosting; there is no server in
the request path (see [architecture.md](architecture.md)). The FastAPI
service documented here (`backend/services/scheduleServer.py`) is a
**local development convenience**: it serves the same data over HTTP
endpoints, mounts the SPA at `/ui/`, and doubles as the schema
reference. It imports the shared `pipelines/serving/` module — the same
code `publish` uses — so its responses match the published JSON.

All data routes are `GET`; no request triggers a model call (all
inference is precomputed monthly by `pipelines.monthly.predict`; the
service loads the predictions parquets once at startup and only reads
them).

Default base URL: `http://127.0.0.1:8000` (or the port passed to
uvicorn). The UI at `/ui/` is same-origin, so no CORS setup is required.
`GET /` issues a 307 redirect to `/ui/`. OpenAPI JSON: `GET
/openapi.json` — Swagger UI: `GET /docs`.

**When predictions aren't loaded.** If the startup load fails (e.g. no
`predictions_*.parquet` on disk), `/health` reports
`status: "degraded"` and the data routes (`/trends`,
`/forecast/fingerprint`) return `503 Service Unavailable` with the load
error in `detail`. `/options` independently returns `503` if
`data/reference/lookup.csv` is missing.

---

## `GET /health`

Liveness + bundle status.

```bash
curl -s http://localhost:8000/health
```

Healthy response:

```json
{
  "status": "healthy",
  "predictions_loaded": true,
  "predictions_anchor_month": "2026-05",
  "predictions_univariate_rows": 119,
  "predictions_fingerprint_rows": 3830,
  "lags_synthetic": true,
  "error": null
}
```

Degraded response (no predictions found at startup):

```json
{
  "status": "degraded",
  "predictions_loaded": false,
  "predictions_anchor_month": null,
  "predictions_univariate_rows": null,
  "predictions_fingerprint_rows": null,
  "lags_synthetic": false,
  "error": "no predictions_univariate_*.parquet found; run `python -m pipelines.monthly run`"
}
```

When `predictions_loaded` is `false`, `status` flips to `"degraded"` and
`error` carries the load failure reason. The loader reports the first
missing parquet, so the message is one of `no predictions_univariate_*.parquet
found; run \`python -m pipelines.monthly run\`` or the matching
`predictions_fingerprint_*` variant.

`lags_synthetic` is `true` when one or more of the anchor's 3 prior lag
months in the merged cube came from
[`scripts/backfill_anchor_lags.py`](../scripts/backfill_anchor_lags.py) —
the synthetic-history stopgap used until enough live months have been
scraped to provide real lag context. UI surfaces a footnote on the chart
legend when this flag is set.

---

## `GET /options`

Vocabularies for UI dropdowns. Each category returns `[{name, id}]` so
the frontend can assemble the fingerprint query string by mapping back
from the user's name selection. Rows are sorted by `name`.

```bash
curl -s http://localhost:8000/options
```

```json
{
  "colors":      [{"name": "Beige", "id": 4}, {"name": "Black", "id": 1}, ...],
  "categories":  [{"name": "Alice band", "id": 49}, {"name": "Bag", "id": 50}, ...],
  "materials":   [{"name": "acrylic", "id": 32}, {"name": "canvas", "id": 14}, ...],
  "appearances": [{"name": "All over pattern", "id": 5}, ...],
  "genders":     [{"name": "Men", "id": 3}, {"name": "Unisex", "id": 2}, {"name": "Women", "id": 1}]
}
```

Source: `data/reference/lookup.csv`. Filters by the `category` column to
the five dimensions the UI exposes — `color_master` → `colors`,
`product_type` → `categories`, `material` → `materials`,
`graphical_appearance` → `appearances`, `gender` → `genders`. The
`color_spectrum` and `product_group` categories exist in `lookup.csv`
but are not returned here (the frontend seeds those from its local
`data.js` fallback). Returns `503` if `lookup.csv` is missing.

---

## `GET /trends`

Every univariate prediction row, optionally filtered. One row per
`(dimension, level_id)` from the latest predictions parquet.

**Query params:**
| Param | Type | Notes |
|---|---|---|
| `dimension` | str | optional. e.g. `color_master`, `material`, `product_type`, `graphical_appearance`, `gender` (the dimensions emitted by `predict`) |
| `state` | str | optional. one of `rising`, `peak`, `flat`, `falling` |

Both filters are exact-match against the parquet columns; unknown values
simply yield an empty list.

```bash
curl -s 'http://localhost:8000/trends?dimension=color_master&state=rising'
```

```json
[
  {
    "dimension": "color_master",
    "level_id": 4,
    "level_name": "Beige",
    "share_lag3": 0.082, "share_lag2": 0.085, "share_lag1": 0.094, "share_t": 0.104,
    "y_h1": 0.108, "y_h2": 0.111, "y_h3": 0.115, "y_h4": 0.118, "y_h5": 0.121, "y_h6": 0.124,
    "state": "rising",
    "stat": "+22% next 6mo"
  },
  ...
]
```

`share_lag3` / `share_lag2` / `share_lag1` / `share_t` are the observed
catalog shares at anchor − 3, − 2, − 1, and the anchor month itself. They
are joined onto the predictions cube at service startup from the latest tick's
`merged_univariate.parquet` (`data/ticks/<YYYY-MM>/`) so the chart has 3 months of
real context to draw before the forecasted 6. May be `null` if the
underlying cube row is missing — in practice the predictions cube only
emits rows where the lag history is complete, so this is rare.

**"Unknown" rows.** Every dimension reserves `level_id = 0` for items the
scraper couldn't categorize (e.g. `color_master:0 = Unknown`,
`material:0 = Unknown`, etc.). The API returns these rows uniformly. The
React frontend drops them at the `api.js` reshape boundary
(`mapTrendsToTrendData` skips any row whose `level_name` is `"Unknown"`,
case-insensitive) — they're real data but not actionable for a reseller.
API clients that DO care about unclassified buckets can read them
directly.

---

## `GET /forecast/fingerprint`

One fingerprint forecast. All five IDs are required as query params.
Returns `404` if no precomputed row matches.

**Query params:** all `int`, all required.
- `product_type_id`
- `gender_id`
- `color_master_id`
- `graphical_appearance_id`
- `material_id`

```bash
curl -s 'http://localhost:8000/forecast/fingerprint?product_type_id=1&gender_id=1&color_master_id=0&graphical_appearance_id=0&material_id=5'
```

```json
{
  "product_type_id": 1, "gender_id": 1, "color_master_id": 0,
  "graphical_appearance_id": 0, "material_id": 5,
  "product_type_name": "Trousers", "gender_name": "Women",
  "color_master_name": "Unknown", "graphical_appearance_name": "Unknown",
  "material_name": "viscose",
  "share_lag3": 4.1e-05, "share_lag2": 3.7e-05, "share_lag1": 3.9e-05, "share_t": 4.0e-05,
  "y_h1": 4.589e-05, "y_h2": 4.962e-05, "y_h3": 5.226e-05,
  "y_h4": 5.267e-05, "y_h5": 5.282e-05, "y_h6": 5.302e-05,
  "state": "rising",
  "stat": "+32% next 6mo"
}
```

The `share_lag3` / `share_lag2` / `share_lag1` / `share_t` fields are the
observed shares for this 5-D fingerprint at anchor − 3..0 months,
joined onto the predictions cube at startup from the latest tick's
`merged_fingerprint.parquet` (`data/ticks/<YYYY-MM>/`).

```bash
# Missing fingerprint → 404
curl -i 'http://localhost:8000/forecast/fingerprint?product_type_id=999999&gender_id=1&color_master_id=1&graphical_appearance_id=1&material_id=1'
# HTTP/1.1 404 Not Found
# {"detail":"no precomputed forecast for fingerprint product_type_id=999999 ..."}
```

**Why some fingerprints are missing:** the predictions parquet only
contains rows where the cube has 4 contiguous months of history at the
anchor month (t-3..t). 5-D combinations that didn't appear in the cube
at the latest anchor — or that lack lag coverage — are silently
skipped during `pipelines.monthly.predict` and therefore 404 here.

**Frontend handling of misses.** In static mode the UI looks the 5-D key
up in `fingerprint.json` client-side; this endpoint is only called when
`window.API_BASE` points at a dev server. Either way, on a miss (null
lookup / 404) the frontend falls back to
`synthesizeFingerprintSeries(tags, trends)` in `frontend/api.js`
— a multiplicative joint built from the per-dimension univariate
forecasts already in memory (no extra fetch). The chart legend
labels the result: "We've never seen this item before! Predicting based
on this item's distinct characteristics." See
[architecture.md § Item recommendation pipeline](architecture.md#item-recommendation-pipeline)
for the full source-priority chain.

---

## State + stat semantics

`state ∈ {rising, peak, flat, falling}` is computed once during
`pipelines.monthly.predict` (in `pipelines/monthly/state.py`) and stored
in the parquet. The API doesn't recompute on read.

`stat` reports the **forward** percentage change (`y_h6 / share_t − 1`,
rounded to int) and is keyed off `state`:
- `rising`: `"+{int}% next 6mo"`
- `falling`: `"−{int}% next 6mo"` (U+2212 minus sign)
- `peak`: `"at peak"`
- `flat`: `"stable"`

Current thresholds (in `pipelines/monthly/state.py`, see file docstring for
the full rule):

- `RISING_RATIO = 1.08` — forward must beat anchor by >8% to fire rising
- `FALLING_RATIO = 0.92` — forward must trail anchor by >8% to fire falling
- `PEAK_MIN_DROP = 0.08` — peak must drop ≥8% to its forward end to fire

All three are flagged for tuning against the real prediction
distributions (see [TODO.md](../../TODO.md) "State-classifier threshold
tuning").

---

## Reloading predictions

The `BUNDLE` global is loaded once at startup via the FastAPI lifespan
hook. To pick up a new monthly tick's output, **restart the service**.
There's no hot-reload endpoint — predictions change at most monthly. (The
production static path has no equivalent step: `publish` refreshes
`frontend/data/*.json` and `firebase deploy` ships it.)
