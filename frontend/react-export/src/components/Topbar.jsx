import React from "react";
import { NAV, TITLES } from "../nav.js";
import { useAppData } from "../appData.jsx";
import { fmtUptime } from "../format.js";

export function Topbar({ active }) {
  const { system } = useAppData();
  const grp = NAV.find((g) => g.items.some(([i]) => i === active));

  return <div className="z-top">
    <div className="z-bcrumb"><s>{grp ? grp.grp + " / " : ""}</s>{TITLES[active] || active}</div>
    <div className="sp" />
    <span className="z-env"><span className="z-pulse" />Running · {fmtUptime(system?.uptime_seconds)}</span>
    <div className="z-env">{system?.version || "—"}</div>
  </div>;
}
