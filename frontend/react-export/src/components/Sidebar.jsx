import React from "react";
import { NAV } from "../nav.js";
import { Ic } from "./icons.jsx";
import { useAppData } from "../appData.jsx";

export function Sidebar({ active, onNav }) {
  const { config, system } = useAppData();
  const mcpCount = (config?.core?.mcp_servers || []).length;
  const devCfg = config?.core?.devices || [];

  // Device statuses arrive live over the panel WS heartbeat (system.devices),
  // so the online/total badge updates without polling or page reloads.
  const devStatus = system?.devices || [];
  const online = devStatus.filter((d) => d.online).length;
  const devTotal = devCfg.length || devStatus.length;

  return <div className="z-side">
    <div className="z-brand">
      <div className="logo">Z</div>
      <div><b>Zakhar</b></div>
    </div>
    <div className="z-nav">
      {NAV.map((g) => <div key={g.grp}>
        <div className="z-navgrp">{g.grp}</div>
        {g.items.map(([id, label]) => (
          <div key={id} className={"z-navi" + (id === active ? " on" : "")} role="button" tabIndex={0}
            onClick={() => onNav(id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onNav(id); } }}>
            <Ic n={id} />{label}
            {id === "mcp" && <span className="badge">{mcpCount}</span>}
            {id === "devices" && devTotal > 0 && <span className="badge">{online}/{devTotal}</span>}
          </div>
        ))}
      </div>)}
    </div>
  </div>;
}
