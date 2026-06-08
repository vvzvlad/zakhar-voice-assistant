import React, { useEffect, useState } from "react";
import { NAV } from "../nav.js";
import { Ic } from "./icons.jsx";
import { useAppData } from "../appData.jsx";
import { getDevices } from "../api.js";
import { fmtUptime } from "../format.js";

export function Sidebar({ active, onNav }) {
  const { system, config } = useAppData();
  const version = system?.version || "—";
  const mcpUrl = config?.core?.mcp?.url;
  const mcpCount = mcpUrl ? 1 : 0;
  const devCfg = config?.core?.devices || [];

  const [devStatus, setDevStatus] = useState([]);
  useEffect(() => {
    let alive = true;
    getDevices().then((d) => { if (alive && Array.isArray(d)) setDevStatus(d); }).catch(() => {});
    return () => { alive = false; };
  }, []);
  const online = devStatus.filter((d) => d.online).length;
  const devTotal = devCfg.length || devStatus.length;

  return <div className="z-side">
    <div className="z-brand">
      <div className="logo">Z</div>
      <div><b>Zakhar</b><div className="ver">{version}</div></div>
    </div>
    <div className="z-nav">
      {NAV.map((g) => <div key={g.grp}>
        <div className="z-navgrp">{g.grp}</div>
        {g.items.map(([id, label]) => (
          <div key={id} className={"z-navi" + (id === active ? " on" : "")} onClick={() => onNav(id)}>
            <Ic n={id} />{label}
            {id === "mcp" && <span className="badge">{mcpCount}</span>}
            {id === "devices" && devTotal > 0 && <span className="badge">{online}/{devTotal}</span>}
          </div>
        ))}
      </div>)}
    </div>
    <div className="z-side-foot"><span className="z-pulse" />Running · {fmtUptime(system?.uptime_seconds)}</div>
  </div>;
}
