import React from "react";
import { NAV, TITLES } from "../nav.js";
import { useAppData } from "../appData.jsx";
import { fmtUptime } from "../format.js";

export function Topbar({ active }) {
  const { system, connected } = useAppData();
  // Match both top-level item ids and nested children ids so nested
  // sections (e.g. mcp/prompt under llm) keep their group breadcrumb.
  const grp = NAV.find((g) =>
    g.items.some(([i, , kids]) => i === active || (kids || []).some(([ci]) => ci === active)),
  );

  return <div className="z-top">
    <div className="z-bcrumb"><s>{grp ? grp.grp + " / " : ""}</s>{TITLES[active] || active}</div>
    <div className="sp" />
    {/* Main liveness indicator: driven by the WS heartbeat (connected), shows
        "Offline" with a red dot when the backend stops sending heartbeats. */}
    <span className="z-env">
      <span className={"z-pulse" + (connected ? "" : " off")} />
      {connected ? `Running · ${fmtUptime(system?.uptime_seconds)}` : "Offline"}
    </span>
    <div className="z-env">{system?.version || "—"}</div>
  </div>;
}
