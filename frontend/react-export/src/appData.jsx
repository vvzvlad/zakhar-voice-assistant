// App-wide data layer. Loads catalog/config/system on mount and exposes them via
// context, plus a `patch()` helper that PATCHes the backend and refreshes state.
import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from "react";
import * as api from "./api.js";

const Ctx = createContext(null);

const HEARTBEAT_TIMEOUT_MS = 4000; // mark Offline if no WS heartbeat within this window (≈4 missed 1 s beats)

export function AppDataProvider({ children }) {
  const [catalog, setCatalog] = useState(null);
  const [config, setConfig] = useState(null);
  const [system, setSystem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Optimistic: assume connected until proven otherwise, so the always-mounted
  // Sidebar doesn't flash a red "Offline" on every load before the first heartbeat.
  // A dead backend is caught quickly when the WS fails to connect (onStatus(false))
  // or when the watchdog fires after HEARTBEAT_TIMEOUT_MS without a heartbeat.
  const [connected, setConnected] = useState(true);

  const loadAll = useCallback(async () => {
    const [cat, cfg, sys] = await Promise.all([
      api.getCatalog(),
      api.getConfig(),
      api.getSystem(),
    ]);
    setCatalog(cat);
    setConfig(cfg);
    setSystem(sys);
  }, []);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    loadAll()
      .then(() => { if (alive) { setError(null); setConnected(true); } })
      .catch((e) => { if (alive) setError(e); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [loadAll]);

  // Subscribe to run frames arriving on the app-wide WS tunnel. Returns an
  // unsubscribe fn. A ref registry (not state) so run frames don't re-render
  // the whole provider tree.
  const runListenersRef = useRef(new Set());
  const subscribeRuns = useCallback((fn) => {
    runListenersRef.current.add(fn);
    return () => runListenersRef.current.delete(fn);
  }, []);

  // Liveness via the WS tunnel: the backend pushes a {type:"system",...} heartbeat
  // every second. Each one refreshes uptime and (re)arms a watchdog; if no
  // heartbeat arrives within HEARTBEAT_TIMEOUT_MS — or the socket closes — the
  // backend is considered down and the panel shows "Offline" instead of a frozen
  // "Running". This is the single app-wide tunnel kept always active.
  const watchdogRef = useRef(null);
  useEffect(() => {
    const clearWatchdog = () => { if (watchdogRef.current) { clearTimeout(watchdogRef.current); watchdogRef.current = null; } };
    const armWatchdog = () => {
      clearWatchdog();
      watchdogRef.current = setTimeout(() => setConnected(false), HEARTBEAT_TIMEOUT_MS);
    };
    const stop = api.openPanelStream({
      onMessage: (msg) => {
        if (msg && msg.type === "system") {
          // Merge so fields the heartbeat omits (e.g. db_size_bytes from the
          // initial HTTP load) are preserved; drop the transport-only `type`.
          const { type, ...sys } = msg;
          setSystem((prev) => ({ ...(prev || {}), ...sys }));
          setConnected(true);
          armWatchdog();
        }
        if (msg && msg.type === "run" && msg.run) {
          for (const fn of runListenersRef.current) {
            try { fn(msg.run); } catch { /* a bad listener must not break the stream */ }
          }
        }
      },
      onStatus: (up) => { if (!up) { clearWatchdog(); setConnected(false); } },
    });
    return () => { stop(); clearWatchdog(); };
  }, []);

  // Apply a partial patch, then refresh catalog + config + system so the UI reflects
  // the persisted state. Re-throws so callers (forms) can surface 422 validation.
  const patch = useCallback(async (patchObj) => {
    const updated = await api.patchConfig(patchObj);
    setConfig(updated);
    const [cat, sys] = await Promise.all([api.getCatalog(), api.getSystem()]);
    setCatalog(cat);
    setSystem(sys);
    return updated;
  }, []);

  const value = {
    catalog, config, system, loading, error, connected,
    patch, reload: loadAll, subscribeRuns,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppData() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppData must be used within AppDataProvider");
  return v;
}
