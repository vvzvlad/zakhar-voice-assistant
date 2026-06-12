import React from "react";
import { NAV, TITLES } from "../nav.js";
import { useAppData } from "../appData.jsx";
import { fmtUptime } from "../format.js";
import { STAGES } from "../stageMeta.js";

// Map a backend category id to its short stage name for the loading badge.
const RELOAD_LABELS = Object.fromEntries(STAGES.map((s) => [s.cat, s.name]));

export function Topbar({ active }) {
  const { system, connected } = useAppData();
  const reloading = system?.reloading || [];
  // Match both top-level item ids and nested children ids so nested
  // sections (e.g. mcp/prompt under llm) keep their group breadcrumb.
  const grp = NAV.find((g) =>
    g.items.some(([i, , kids]) => i === active || (kids || []).some(([ci]) => ci === active)),
  );

  return <div className="z-top">
    <div className="z-bcrumb"><s>{grp ? grp.grp + " / " : ""}</s>{TITLES[active] || active}</div>
    <div className="sp" />
    {/* Global loading badge: shown whenever a backend category is being hot-reloaded
        (model load in flight), surfaced via the system heartbeat's `reloading` list. */}
    {reloading.length > 0 && (
      <span className="z-env loading" title={`Loading: ${reloading.join(", ")}`}>
        <span className="z-spin" />
        Loading… {reloading.map((c) => RELOAD_LABELS[c] || c).join(", ")}
      </span>
    )}
    {/* Main liveness indicator: driven by the WS heartbeat (connected), shows
        "Offline" with a red dot when the backend stops sending heartbeats. */}
    <span className="z-env">
      <span className={"z-pulse" + (connected ? "" : " off")} />
      {connected ? `Running · ${fmtUptime(system?.uptime_seconds)}` : "Offline"}
    </span>
    <div className="z-env">{system?.version || "—"}</div>
  </div>;
}
