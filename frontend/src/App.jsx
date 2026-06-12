import React, { useState, useEffect } from "react";
import { TITLES } from "./nav.js";
import { setNavigate } from "./navStore.js";
import { Sidebar } from "./components/Sidebar.jsx";
import { Topbar } from "./components/Topbar.jsx";
import { PAGES } from "./pages/index.js";
import { useAppData } from "./appData.jsx";
import { Loading, ErrorBox } from "./components/primitives.jsx";
import { Toasts } from "./components/Toasts.jsx";

// Set of valid section ids derived from the nav config.
const VALID = new Set(Object.keys(TITLES));

// Extract the section id from a URL pathname; returns null if it is not a known section.
function idFromPath(pathname) {
  const seg = pathname.replace(/^\/+/, "").split("/")[0];
  return VALID.has(seg) ? seg : null;
}

// Resolve the initial section: prefer the URL, then localStorage, then the default.
function initialId() {
  const fromPath = idFromPath(window.location.pathname);
  if (fromPath) return fromPath;
  // Validate the persisted id against the known sections so a stale/removed id
  // (e.g. the old "device" before it was renamed to "system") falls back cleanly.
  try {
    const stored = localStorage.getItem("z-active");
    return VALID.has(stored) ? stored : "dashboard";
  } catch { return "dashboard"; }
}

export default function App() {
  const { loading, error, reload } = useAppData();
  const [active, setActive] = useState(initialId);

  // Apply a section switch without touching browser history (used by clicks and popstate).
  const applyActive = (id) => {
    setActive(id);
    try { localStorage.setItem("z-active", id); } catch { /* ignore */ }
    const sc = document.querySelector(".z-scroll");
    if (sc) sc.scrollTop = 0;
  };

  // Navigate from user intent (sidebar / page links): push a new history entry, then apply.
  const navigate = (id) => {
    if (id === active) return; // avoid duplicate history entries on re-click
    try { window.history.pushState({ id }, "", "/" + id); } catch { /* ignore */ }
    applyActive(id);
  };

  // Register the navigate fn so pages can call nav(id) from the module-level store.
  useEffect(() => { setNavigate(navigate); });

  // On mount: sync the address bar with the actually selected section (no history entry),
  // and handle browser Back/Forward via popstate.
  useEffect(() => {
    if (idFromPath(window.location.pathname) !== active) {
      try { window.history.replaceState({ id: active }, "", "/" + active); } catch { /* ignore */ }
    }
    const onPop = () => {
      const id = idFromPath(window.location.pathname) || "dashboard";
      applyActive(id); // no pushState here, otherwise history would loop
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const Page = PAGES[active] || (() => (
    <div className="z-page"><div className="z-empty"><b>{TITLES[active]}</b>Section coming up.</div></div>
  ));

  return (
    <div className="z-app">
      <Sidebar active={active} onNav={navigate} />
      <div className="z-main">
        <Topbar active={active} />
        <div className="z-scroll">
          {loading
            ? <div className="z-page"><div className="z-card"><Loading /></div></div>
            : error
              ? <div className="z-page"><div className="z-card"><ErrorBox error={error} onRetry={reload} /></div></div>
              : <Page />}
        </div>
      </div>
      <Toasts active={active} />
    </div>
  );
}
