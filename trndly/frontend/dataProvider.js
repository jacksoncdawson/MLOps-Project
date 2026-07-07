// dataProvider.js — session-scoped store for inventory + signals,
// with fetched trends + options + health (published static JSON by
// default; the dev API when window.API_BASE is set — see api.js).
//
// ─────────────────────────────────────────────────────────────────
// Public surface (the contract screens consume):
//   useData() → {
//     inventory, signals, recentlyAddedName,    // session state (start empty)
//     addItem(item),                             // mutator
//     trends,                                    // API-backed; undefined while loading/errored
//     options, lookupIds,                        // API-backed (with one seed exception)
//     trendsLoading,  trendsError,
//     optionsLoading, optionsError,
//     health, healthLoading, healthError,        // for the sidebar API-status pill
//   }
//   <DataProvider>
//
// trends / options / health come from those fetches. When they fail or are
// still loading, this provider exposes `undefined`/error rather than
// substituting fixtures — the screens render explicit loading and error
// states so the failure mode is visible.
//
// One intentional exception: `options` falls back to `LOOKUP_OPTIONS` from
// data.js. The /options endpoint does not yet expose `colorSpectrum` or
// `productGroup`, and the Add Item form depends on them. Remove the
// fallback once the endpoint is extended.
//
// Inventory starts empty and persists for the browser session only —
// refreshing the page resets it. To make it durable, swap the empty
// initial state for an API fetch and route addItem through POST /api/inventory.
//
// ─── Fetcher hook ────────────────────────────────────────────────────
// We use a tiny in-house useFetch instead of SWR. SWR 2.x ships ESM only,
// so the UMD CDN tag we used to rely on returned 404; rather than wrestle
// with bundling for a no-build setup, this hook covers what we actually
// need: keyed cache, optional polling, optional refocus revalidation.
// ─────────────────────────────────────────────────────────────────────

const DataContext = React.createContext(null);

// Module-level cache shared across all useFetch consumers of the same key.
const _fetchCache = new Map();        // key → { data?, error? }
const _fetchSubscribers = new Map();  // key → Set<setState>

function _notify(key, payload) {
  const subs = _fetchSubscribers.get(key);
  if (!subs) return;
  for (const set of subs) set({ ...payload, isLoading: false });
}

function useFetch(key, fetcher, options) {
  const opts = options || {};
  const { refreshInterval = 0, revalidateOnFocus = false } = opts;

  const cached = _fetchCache.get(key);
  const [state, setState] = React.useState(() => ({
    data:      cached ? cached.data  : undefined,
    error:     cached ? cached.error : undefined,
    isLoading: !cached,
  }));

  React.useEffect(() => {
    if (!key) return undefined;

    let alive = true;
    let subs = _fetchSubscribers.get(key);
    if (!subs) { subs = new Set(); _fetchSubscribers.set(key, subs); }
    subs.add(setState);

    const run = () => {
      Promise.resolve()
        .then(() => fetcher(key))
        .then(data => {
          if (!alive) return;
          const next = { data, error: undefined };
          _fetchCache.set(key, next);
          _notify(key, next);
        })
        .catch(error => {
          if (!alive) return;
          const next = { data: undefined, error };
          _fetchCache.set(key, next);
          _notify(key, next);
        });
    };

    // Always fire an initial fetch on mount — gives the user a fresh value
    // even when re-mounting against a stale cache entry (e.g. after fix-and-reload).
    run();

    let intervalId;
    if (refreshInterval > 0) intervalId = setInterval(run, refreshInterval);

    let focusHandler;
    if (revalidateOnFocus) {
      focusHandler = () => run();
      window.addEventListener('focus', focusHandler);
    }

    return () => {
      alive = false;
      subs.delete(setState);
      if (intervalId) clearInterval(intervalId);
      if (focusHandler) window.removeEventListener('focus', focusHandler);
    };
  }, [key, refreshInterval, revalidateOnFocus]);

  return state;
}

function DataProvider({ children }) {
  const [inventory, setInventory] = React.useState([]);
  const [signals,   setSignals]   = React.useState({});
  // Name of the most recently added item — used to animate its tile on the
  // inventory grid. Auto-clears after a few seconds so it doesn't re-animate
  // on subsequent visits within the same session.
  const [recentlyAddedName, setRecentlyAddedName] = React.useState(null);

  // ─── API-backed trends + options + health ─────────────────────────────
  const trendsRes  = useFetch('/trends',  apiFetcher, { revalidateOnFocus: false });
  const optionsRes = useFetch('/options', apiFetcher, { revalidateOnFocus: false });
  const healthRes  = useFetch('/health',  apiFetcher, { refreshInterval: 15000, revalidateOnFocus: true });

  // `undefined` while loading or errored — screens distinguish via trendsLoading/trendsError.
  const trends = React.useMemo(() => {
    if (trendsRes.data) return mapTrendsToTrendData(trendsRes.data);
    return undefined;
  }, [trendsRes.data]);

  // Options falls back to LOOKUP_OPTIONS so the Add form keeps working with
  // colorSpectrum/productGroup (not yet exposed by /options). Drop this
  // fallback once those vocabs are in the endpoint response.
  const optionsState = React.useMemo(() => {
    if (optionsRes.data) {
      return {
        options:    mapOptionsToLookupOptions(optionsRes.data),
        lookupIds:  indexOptionsById(optionsRes.data),
      };
    }
    return { options: LOOKUP_OPTIONS, lookupIds: {} };
  }, [optionsRes.data]);

  // Keep the legacy window globals in sync with whatever's currently loaded
  // — `lookupTrendState` in data.js reads window.TREND_DATA lazily, and
  // refactoring the consumers is a follow-up.
  React.useEffect(() => {
    if (trendsRes.data) window.TREND_DATA = trends;
  }, [trends]);
  React.useEffect(() => {
    if (optionsRes.data) window.LOOKUP_OPTIONS = optionsState.options;
  }, [optionsState.options]);

  const addItem = React.useCallback((item) => {
    setInventory(prev => [item, ...prev]);
    if (item.signals) {
      setSignals(prev => ({ ...prev, [item.name]: item.signals }));
    }
    setRecentlyAddedName(item.name);
    setTimeout(() => setRecentlyAddedName(curr => curr === item.name ? null : curr), 4000);
  }, []);

  const value = {
    inventory,
    signals,
    recentlyAddedName,
    addItem,
    trends,
    options:        optionsState.options,
    lookupIds:      optionsState.lookupIds,
    trendsLoading:  trendsRes.isLoading,
    optionsLoading: optionsRes.isLoading,
    trendsError:    trendsRes.error,
    optionsError:   optionsRes.error,
    health:         healthRes.data,
    healthLoading:  healthRes.isLoading,
    healthError:    healthRes.error,
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
}

function useData() {
  const ctx = React.useContext(DataContext);
  if (!ctx) throw new Error('useData must be used inside <DataProvider>');
  return ctx;
}

Object.assign(window, { DataProvider, useData, useFetch });
