# trndly — Frontend

Desktop web app for trndly: a tool for resellers (Depop, Poshmark, vintage
shops) that pairs trend prediction with per-item listing recommendations.

The screens consume the published static JSON in `frontend/data/`
(emitted by the tick's `publish` stage) — the same files Firebase Hosting
serves in production. A local FastAPI server is available as a dev
alternative (see [../docs/api.md](../docs/api.md)). Auth is still a demo
no-op — one of a few remaining placeholders (Settings is unwired and
inventory isn't persisted; see below).

---

## Quick start

No build step. The app uses Babel-in-browser to transpile JSX on load,
and fetches its data from relative `./data/*.json` paths — so any static
server pointed at `frontend/` works:

```sh
# from trndly/frontend/
python3 -m http.server 8080
# → http://localhost:8080/
```

Alternatively, run the FastAPI dev server, which mounts the SPA at `/ui/`
and serves the same data through live endpoints (`/trends`, `/options`,
`/forecast/fingerprint`, `/health`):

```sh
# from trndly/ (the inner package dir)
.venv/bin/python -m uvicorn backend.services.scheduleServer:app --port 8000
# → http://localhost:8000/ui/
# (or: bash scripts/run_api.sh)
```

To point a statically-served frontend at a running API instead of the
JSON files, set `window.API_BASE = 'http://localhost:8000'`.

If the data fetch fails (no `data/*.json` published yet, or the API isn't
running), every screen that depends on it (Highlights, Trends) shows an
explicit error card with a backend-agnostic message. The sidebar carries
a live status pill (green / amber / red dot) driven by the health data.

---

## File map

```
frontend/
├── index.html              ← entry point. Loads CSS, React, Babel, then JS in order.
├── App.jsx                 ← root: <AuthProvider> + <DataProvider> wrap the app.
├── data.js                 ← helpers + STATE_META + LOOKUP_OPTIONS seed + deriveRecommendationFromSeries.
├── api.js                  ← /trends + /options + /forecast/fingerprint fetchers + reshape adapters.
├── dataProvider.js         ← useData() hook (in-house useFetch + session-scoped inventory).
├── auth.js                 ← DEMO AUTH — context + always-succeeds login (placeholder).
├── tokens.css              ← design tokens (CSS custom properties).
├── colors_and_type.css     ← semantic color + typography aliases.
├── README.md               ← this file.
│
├── assets/                 ← brand SVGs.
│
├── components/             ← reusable UI primitives.
│   ├── Sidebar.jsx         ← left nav rail + API status pill + user pill + TopBar.
│   ├── Button.jsx          ← multi-variant button.
│   ├── Tag.jsx             ← pill badges.
│   ├── TrendCard.jsx       ← TrendCard + TrendChip + ChartSparkline.
│   ├── Chart.jsx           ← HighlightSparkline, ItemPopularityChart, ChartLegend, SectionLabel.
│   └── ItemGraphic.jsx     ← garment placeholder SVGs (visual fallback only).
│
└── screens/                ← full-page views.
    ├── ScreenLogin.jsx     ← demo sign-in screen.
    ├── ScreenHighlights.jsx← landing: 4 callouts derived from /trends (biggest mover / at peak / sleeping low / sharpest drop).
    ├── ScreenTrends.jsx    ← all features. ≥1500px viewport: 5 per-category columns with state-filter chips. Narrower: chip+grid layout.
    ├── ScreenInventory.jsx ← user inventory grouped by listing window (5 groups; starts empty).
    ├── ScreenItem.jsx      ← per-item detail (chart + signal cards). Series source: /forecast/fingerprint → synthesized joint → "more data needed".
    ├── ScreenAdd.jsx       ← add new item form (with image dropzone). Computes recommendation at submit time from synthesized series.
    └── ScreenSettings.jsx  ← placeholder.
```

---

## How rendering works

There is no bundler. `index.html` loads everything in a deliberate order:

1. **Tokens / typography CSS** — design vars must exist before any component renders.
2. **React + ReactDOM (UMD)** — exposes `React`, `ReactDOM` globals.
3. **Babel Standalone** — transpiles every `<script type="text/babel">` block.
4. **`data.js`** — plain JS (no `text/babel`), attaches helpers + LOOKUP_OPTIONS to `window`.
5. **`api.js`** — plain JS, fetchers + reshape helpers on `window`.
6. **`auth.js`** — defines `AuthProvider` + `useAuth` on `window`.
7. **`dataProvider.js`** — defines `DataProvider` + `useData` (with the
   in-house `useFetch` hook for `/trends`, `/options`, `/health`).
8. **Components** (leaves before composites: `Tag`, `Button`, `Chart`,
   `ItemGraphic`, `TrendCard`, `Sidebar`).
9. **Screens** (`ScreenLogin` first, then the authenticated screens).
10. **`App.jsx`** — final render call.

Every module attaches its public surface to `window` (`Object.assign(window,
{...})`) so siblings can pick them up by global lookup. This is the cost of
not having a bundler. When migrating to Vite, swap each global write for a
real ESM `export`.

> Note: `data.js` and `api.js` load as plain `<script>` tags (no JSX, so no
> Babel pass) — they run before the `text/babel` modules. Everything from
> `auth.js` onward is `type="text/babel"`.

**Cache-bust query (`?v=…`).** Babel-standalone transpiles JSX in the browser
and caches transforms by URL — when you edit a `.jsx` file, bump its `?v=`
query string in `index.html` or the browser will keep serving the previous
build.

---

## Data flow

```
frontend/data/*.json   (default: published by the tick, served same-origin)
   trends.json         univariate predictions        → TrendRow[]
   options.json        dropdown vocabularies         → categorized {name,id}[]
   fingerprint.json    5-D-keyed bundle              → client-side lookup
   health.json         bundle status + anchor month
      — OR, when window.API_BASE is set —
FastAPI dev server     /trends /options /forecast/fingerprint /health
   ▼
api.js          apiFetcher + reshape adapters (no caching)
   ▼
dataProvider.js useData() — useFetch-cached + session state for inventory
   ▼
screens         render trends, signals, status; show loading/error/empty
                states explicitly when the trends or health fetch fails
```

Trends, options, and health are fetched (and cached) by
`dataProvider.js`'s `useFetch`; the fingerprint series is resolved
per-item by `ScreenItem.jsx` (a key lookup into the fetched
`fingerprint.json` bundle, or a `/forecast/fingerprint` query in API
mode).

Inventory and per-item signals are **session-scoped** — `useState([])` /
`useState({})` reset on every page reload. To persist, swap the empty
initial state for an API fetch and route `addItem` through a new POST
endpoint (not built).

The static files and the dev API share one source of truth — both are
produced/served through the shared `pipelines/serving/` module — and one
response shape, so the frontend doesn't know or care which mode it's in.

---

## API contracts the screens consume

After `api.js` reshapes, screens see:

- **Trends array** — `{ name, category, state, stat, series }` per row.
  - `category`: `'color' | 'material' | 'appearance' | 'product type' | 'gender'`
  - `state`: `'rising' | 'peak' | 'flat' | 'falling'`
    — produced by [pipelines/monthly/state.py](../pipelines/monthly/state.py).
    These are the **trend labels** (per-feature direction), distinct from
    the **recommendation outcomes** further down.
  - `stat`: free-text forecast string (e.g. `'+38% next 6mo'`, `'at peak'`,
    `'−18% next 6mo'`).
  - `series`: `{ past: [s_lag3, s_lag2, s_lag1, s_t], future: [y_h1, …, y_h6] }`
    — the 10 numeric points that drive the sparkline. `null` if the API
    didn't carry the lag values (`seriesFromRow` returns null if any past or
    future point is null).
  - **"Unknown" rows are filtered out at the api.js layer** (every dimension
    reserves `id=0` for unclassified items; surfacing them isn't actionable
    for a reseller). They're still present in the API response — the filter
    is purely UI hygiene.
- **Options** — `{ color, productType, material, appearance, gender,
  colorSpectrum, productGroup }`. First five come from `/options`; last two
  seeded from `LOOKUP_OPTIONS` in `data.js` because the endpoint doesn't
  expose them yet. (`/options` returns `colors`, `categories`, `materials`,
  `appearances`, `genders` — see `OptionsResponse` in `scheduleServer.py`.)
- **`lookupIds`** — `{ category: { name: id } }` maps for posting IDs back
  to the API (e.g. for `/forecast/fingerprint` queries). Built by
  `indexOptionsById` in `api.js` from the `/options` `{name, id}` pairs.
- **`health`** — `{ status, predictions_loaded, predictions_anchor_month,
  predictions_univariate_rows, predictions_fingerprint_rows, lags_synthetic,
  error }`. Powers the sidebar status pill and the "synthetic past" footnote
  on the Item Detail chart legend. (`status` is `'healthy'` or `'degraded'`.)
- **Inventory item** — `{ name, color, type, cost, added, state, image?,
  signals?, tags? }`. Built by `ScreenAdd` and held in `useData().inventory`.
  - `state` is the **recommendation outcome** (see below), not the trend
    `state` vocabulary above.
  - `tags` is the raw 5-tag map (`{ productType, color, material,
    appearance, gender }`) the user picked; `ScreenItem` uses it to resolve
    fingerprint IDs and to synthesize a fallback series.
- **Signal card** — `{ label, value, state, category }`. Built by
  `buildSignalsFromTags()` in `data.js`, which calls
  `lookupTrendState(value, category)` — **category-aware** to disambiguate
  names that exist in multiple dimensions (e.g. `denim` is both a material
  and a graphical_appearance). Signal cards are **informational labels
  only**; they do not drive the recommendation. They DO serve as the data
  source for the synthesized fingerprint when the precomputed combination
  is unavailable.

### Recommendation pipeline

The item-level recommendation pill is derived directly from a single
10-point series — the same series the Overall Popularity chart shows.
Source priority:

1. The precomputed fingerprint — a 5-D key lookup into `fingerprint.json`
   (or `GET /forecast/fingerprint` in API mode). Gold standard.
2. `synthesizeFingerprintSeries(tags, trends)` in `api.js` — when the
   fingerprint lookup misses. Computes a joint forecast by
   multiplying each populated tag dimension's relative motion across the
   forward window (multiplicative independence). The chart renders a
   user-friendly note: **"We've never seen this item before! Predicting
   based on this item's distinct characteristics."**
3. `null` — when no tags resolve and no fingerprint is available. Chart
   hidden; pill reads "More data needed".

`deriveRecommendationFromSeries(series)` in `data.js` returns one of five
outcomes, which map to inventory groups + pill labels:

| internal state    | pill label         | inventory group        | meaning                              |
|-------------------|--------------------|------------------------|--------------------------------------|
| `list now`        | **List now**       | List now               | argmax in forward window is at anchor, OR upside < 2.5% |
| `hold 1mo`        | **1 month**        | List in 1 month        | forecast peaks at h1                 |
| `hold 2mo`        | **2 months**       | List in 2 months       | forecast peaks at h2                 |
| `hold 3+`         | **Hold 3+ months** | Hold 3+ months         | forecast peaks at h3–h6              |
| `more data needed`| **More data needed** | More data needed     | no series available                  |

Threshold is `UPSIDE_THRESHOLD = 0.025` in `data.js` (the peak must beat
anchor by at least 2.5% for any "hold" recommendation; otherwise storage
cost isn't worth it).

> The `List now` inventory group also catches a legacy `falling` state, so
> items added before the recommendation rework still bucket correctly
> (see `TIMELINE_GROUPS` in `ScreenInventory.jsx`).

---

## Failure modes (and how the UI surfaces them)

Each screen consumes data through `useData()` and renders one of four
states (loading / error / empty / data):

| Surface         | API call          | Loading                         | Error                                      | Empty                          |
|-----------------|-------------------|---------------------------------|--------------------------------------------|--------------------------------|
| Highlights      | `/trends`         | 4 row skeletons                 | error card with `/health` advice           | "no trends to highlight"       |
| Trends          | `/trends`         | 6 card skeletons                | error card (same copy)                     | "no features match this filter" |
| Inventory       | (session-local)   | —                               | —                                          | "no items yet" + Add CTA       |
| Item detail     | `/forecast/fingerprint` (per-item) + session-local | — | (falls back to synthesized series on 404)  | "no item selected" / "no feature signals" |
| Sidebar pill    | `/health` (15s poll) | grey "API · connecting"      | red "API · offline" (tooltip = error msg)  | amber "API · degraded" if bundle missing |

Error copy is backend-agnostic — it surfaces whatever the fetch
returned, so the same messages work in static mode and API mode alike.

---

## Demo seam — auth

`auth.js` is demo-mode (and one of the remaining placeholders, alongside
Settings and inventory persistence):

- `useAuth()` returns `{ user, login, logout }`. `user` is `null` when
  signed out, `{ name, email }` when signed in.
- `login()` always succeeds and sets the user to `{ name: 'Demo User',
  email: 'demo@trndly.com' }` regardless of inputs.
- To swap in real auth, keep the same `useAuth()` shape; replace the
  in-memory `useState(null)` with a session check (cookie, JWT, OAuth
  callback) and `login()` with a real round-trip. The consumers (login
  screen + sidebar) go through `useAuth()` and won't need changes.

`ScreenAdd.jsx`'s image dropzone is also non-network on purpose — it
reads the file into a `FileReader` data URL for visual preview. Wire it
to real storage when you wire inventory persistence.

`ScreenSettings.jsx` is a placeholder too: four section cards (Account,
Notifications, Data sources, Appearance), all flagged "coming soon" and
unwired.

---

## Conventions

- **Tokens are the contract.** Production code should reference CSS vars
  from `tokens.css` / `colors_and_type.css` rather than hard-coded hex
  values. The kit's JSX inlines hexes only because Babel-in-browser
  doesn't read CSS modules.
- **No mobile.** Desktop only.
- **No icon library committed.** Sidebar icons are inlined as SVG strings
  in `Sidebar.jsx`. Adopt Lucide (MIT, stroke-based) when expanding.
- **No real photography.** Item thumbnails use placeholder garment SVGs
  from `ItemGraphic.jsx`. Wire to a real CDN when ready.
- **Charts are hand-rolled SVG.** Suitable for the prototype; swap to
  Recharts / Visx for production while keeping the same color + stroke
  conventions.
- **Each screen carries `data-screen-label`** on its root element for
  tooling / screenshot diffing.

---

## Migration checklist

1. Replace `auth.js` with a real session client. `useAuth()` shape is the contract.
2. Build inventory persistence (`POST /api/inventory`, `GET /api/inventory`)
   and swap the empty `useState([])` in `dataProvider.js` for an API fetch.
3. Extend `/options` to include `colorSpectrum` and `productGroup`, then
   delete the `LOOKUP_OPTIONS` fallback in `data.js`.
4. Wire `ScreenAdd.jsx` image upload + form submit to real endpoints.
5. Wire up `ScreenSettings.jsx` (currently a placeholder).
6. Move the no-build setup to Vite. Convert each
   `Object.assign(window, {…})` to a real ESM `export`. Drop the
   `babel-standalone` script tag from `index.html`.
7. Adopt a chart library (Recharts/Visx). Replace the SVG primitives in
   `components/Chart.jsx` while preserving the past/now/predicted color
   conventions documented in the design system.