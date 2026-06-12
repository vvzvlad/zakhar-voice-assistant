# Zakhar Control Panel — React

Admin panel for the Zakhar Voice Assistant, exported as a Vite + React app.

## Run

```bash
npm install
npm run dev      # start dev server (http://localhost:5173)
npm run build    # production build -> dist/
npm run preview  # preview the production build
```

Requires Node 18+.

## Structure

```
index.html              Vite entry; loads IBM Plex fonts + /src/main.jsx
src/
  main.jsx              React root
  App.jsx               App shell: sidebar + topbar + active-section router
  styles.css            Design-system stylesheet (CSS variables, all components)
  data.js               Mock data model (ZAPP) — swap for your API
  nav.js                Navigation config (groups, titles)
  navStore.js           Module-level navigate() so any page can switch sections
  components/
    icons.jsx           SVG icon set + <Ic/>
    primitives.jsx      Field, Seg, Selector, Toggle, Slider, Stepper, Select,
                        Pill, StatusPill, Waterfall, PageHeader, SaveBar, Modal,
                        Player, KV, Spark, total/segsFor helpers
    Sidebar.jsx         Left navigation rail
    Topbar.jsx          Top bar (uptime, version, restart)
  pages/
    index.js            id -> page component map
    dashboard.jsx       Pipeline overview (KPIs, service map, recent runs)
    stages.jsx          VAD, STT, LLM, RUAccent, TTS settings
    log.jsx             Request log + detail drawer (waterfall, tool calls, audio)
    integrations.jsx    MCP servers, System prompt, Context, Devices
    network.jsx         Network / audio server
```

## Notes

- All data is mock (`src/data.js`). Wire the forms, selectors and tables to your
  backend; controls already hold local React state.
- Styling is plain CSS with custom properties in `:root` (no CSS framework).
- Env-var names from the backend (e.g. `INTENT_MODEL`) are passed to `<Field param=.../>`
  for reference in source but intentionally not rendered in the UI.
