# Zakhar Control Panel — React

Admin panel for the Zakhar Voice Assistant, a Vite + React app. The backend serves its
production build from `dist/` (`make frontend` builds it, `make run` builds it first).

## Run

```bash
npm install
npm run dev      # dev server (http://localhost:5173), proxies /api to the backend on :8201
npm run build    # production build -> dist/
npm run preview  # preview the production build
npm test         # vitest, tests in src/__tests__/
```

Requires Node 18+.

## Structure

```
index.html              Vite entry; loads /src/main.jsx
src/
  main.jsx              React root
  App.jsx               App shell: sidebar + topbar + active-page router (the page id is the URL path)
  styles.css            Design-system stylesheet (CSS variables, all components)
  fonts.css, fonts/     IBM Plex Sans / Mono, bundled
  api.js                Fetch wrapper around the backend's /api routes
  appData.jsx           Data layer: catalog, config and system state, patch(), WS heartbeat
  schema.js             Helpers for the pydantic JSON Schema the backend emits
  useStageForm.js       Editing state of one schema-driven form (draft, dirty, 422 errors)
  usePageForm.js        Editing state of a page whose cards save together in one PATCH
  nav.js                Navigation config (groups, titles)
  navStore.js           Module-level navigate() so any page can switch pages
  components/
    icons.jsx           SVG icon set + <Ic/>
    primitives.jsx      Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, Card,
                        PageHeader, SaveBar, FormPage, Modal, Player, Waterfall and the rest
    SchemaForm.jsx      Generic form for a pydantic JSON Schema object
    Sidebar.jsx         Left navigation rail
    Topbar.jsx          Top bar (running/offline with uptime, version, stages reloading)
    Toasts.jsx          Popup notifications of pipeline runs
    DeviceModal.jsx     Add / Edit speaker dialog of the Devices page
    DeviceLog.jsx       One device's own log, by day
    NluCatalogEditor.jsx  Editor of the command catalog (core.commands)
  pages/
    index.js            id -> page component map
    dashboard.jsx       Pipeline overview (KPIs, service map, recent runs)
    devices.jsx         Devices
    pipeline.jsx        Capture and Speech: which stages run
    chains.jsx          Logic: the ordered recognition chains (core.chains)
    stages.jsx          Providers: VAD, STT, LLM, Commands, TTS settings
    prompt.jsx          System prompt profiles
    tools.jsx           Tools: external MCP servers and built-in tool sources
    log.jsx             Request log + detail drawer (waterfall, tool calls, audio)
    system.jsx          System: network, audio server, agent MCP, run log, log level
```

## Notes

- Every page reads and writes the live backend config through `appData.jsx`; there is no
  mock data.
- Styling is plain CSS with custom properties in `:root` (no CSS framework).
