# holdout dashboard

A local-first, read-only SPA over a holdout run store, served by
`src/holdout/dashboard/server.py`. Vite + React + TypeScript +
framer-motion, with hand-rolled SVG charts and a macOS-Tahoe
"Liquid Glass" design system. No chart library, no UI kit, no fonts
fetched — Apple system fonts only, zero bytes leave the machine.

## Dev workflow

1. Start the Python API server (it owns the data; the SPA is a pure client):

   ```sh
   cd ..
   .venv/bin/holdout --store .holdout-demo dashboard --port 4517 --no-open
   ```

2. Run the Vite dev server (proxies `/api` → `127.0.0.1:4517`):

   ```sh
   npm install
   npm run dev
   ```

## Build

```sh
npm run build
```

Emits into `../src/holdout/dashboard_dist/` (committed, shipped inside the
Python wheel). `vite.config.ts` sets `outDir`, `emptyOutDir`, and no source
maps. After building, the Python server serves the SPA at `/` with an
index.html fallback for client routes, so deep links like `/compare` and
`/runs/<id>` survive refresh.

## Screenshots

With the Python server running on port 4517 and assets built:

```sh
npm run screenshot   # writes docs/assets/dashboard-{dark,light,compare-dark}.png
```

## Architecture

```
src/
  main.tsx          entry: StrictMode + MotionConfig(reducedMotion) + Router
  App.tsx           shell: ambient background, sidebar, toolbar, routed views
  styles.css        design system: glass tiers, tokens, light/dark themes
  api.ts            typed fetch hook + API payload types (source of truth)
  data.tsx          shared store context: /api/runs, /api/ledger, /api/meta
  theme.tsx         auto/light/dark preference, resolved to <html data-theme>
  title.tsx         page-title plumbing for the condensing toolbar
  format.ts         percent/CI/p-value/latency/date formatting, axis ticks
  hooks.ts          ResizeObserver-based useMeasure for responsive SVG
  components/
    Glass.tsx        glass cards with cursor-following specular highlight
    Segmented.tsx    Apple segmented control (sprung layoutId thumb)
    Sidebar.tsx      chrome-glass nav: views, evals + ledger level dots, theme
    Toolbar.tsx      sticky translucent toolbar; large title condenses on scroll
    TrendChart.tsx   metric-over-time SVG: CI band, gold line, glass tooltip
    ErrorBar.tsx     compact CI whisker that springs outward from the estimate
    PairedErrorBars  baseline vs candidate CIs on one shared axis (Compare)
    VerdictPill.tsx  REGRESSED / IMPROVED / NO CHANGE / INSUFFICIENT DATA
    ProgressArc.tsx  thin gold uses-vs-budget arc (Discipline)
    RunPicker.tsx    glass popover run selector
    CountUp.tsx      spring-driven number count-up
    Icons.tsx        hand-drawn 1.5px-stroke outline icon set
  views/
    Timeline.tsx     home: trend chart + run cards for the selected eval
    Compare.tsx      the signature view: verdict + paired error bars + stats
    RunDetailView    metrics + per-case results table (errors highlighted)
    Discipline.tsx   ledger budgets per eval, Dwork et al. framing
```

Design notes:

- Three glass tiers (`.glass`, `.glass-elev`, `.glass-chrome`) with real
  `backdrop-filter` blur/saturate, a 1px specular top edge, hairline border,
  and depth shadow; ambient color blobs sit behind everything so the glass
  has something to refract.
- All data comes from the live API; the SPA renders loading/empty/error
  states as glass cards. Nothing is mocked.
- Motion is spring-physics only (framer-motion) and animates transform /
  opacity exclusively; `prefers-reduced-motion` disables it globally via
  `MotionConfig reducedMotion="user"` plus CSS media queries.
- Axis domains: run-card error bars share a per-metric domain across the
  eval so positions are comparable card-to-card; Compare zooms to the pair.
