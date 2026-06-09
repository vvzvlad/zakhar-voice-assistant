// App-wide data layer. Loads catalog/config/system on mount and exposes them via
// context, plus a `patch()` helper that PATCHes the backend and refreshes state.
import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import * as api from "./api.js";

const Ctx = createContext(null);

export function AppDataProvider({ children }) {
  const [catalog, setCatalog] = useState(null);
  const [config, setConfig] = useState(null);
  const [system, setSystem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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
      .then(() => { if (alive) setError(null); })
      .catch((e) => { if (alive) setError(e); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [loadAll]);

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
    catalog, config, system, loading, error,
    patch, reload: loadAll,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppData() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppData must be used within AppDataProvider");
  return v;
}
