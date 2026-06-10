import React from "react";
import { NAV, TITLES } from "../nav.js";
import { useAppData } from "../appData.jsx";
import { fmtUptime } from "../format.js";

export function Topbar({ active }) {
  const { system, connected } = useAppData();
  const grp = NAV.find((g) => g.items.some(([i]) => i === active));

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
