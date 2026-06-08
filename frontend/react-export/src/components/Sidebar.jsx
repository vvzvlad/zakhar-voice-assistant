import React from "react";
import Z from "../data.js";
import { NAV } from "../nav.js";
import { Ic } from "./icons.jsx";

export function Sidebar({ active, onNav }) {
  return <div className="z-side">
    <div className="z-brand">
      <div className="logo">Z</div>
      <div><b>Zakhar</b><div className="ver">{Z.meta.version}</div></div>
    </div>
    <div className="z-nav">
      {NAV.map((g) => <div key={g.grp}>
        <div className="z-navgrp">{g.grp}</div>
        {g.items.map(([id, label]) => {
          const st = Z.stages.find((s) => s.key === id);
          return <div key={id} className={"z-navi" + (id === active ? " on" : "")} onClick={() => onNav(id)}>
            <Ic n={id} />{label}
            {id === "mcp" && <span className="badge">{Z.mcp.filter((m) => m.enabled).length}</span>}
            {id === "devices" && <span className="badge">{Z.devices.filter((d) => d.status === "online").length}/{Z.devices.length}</span>}
            {st && st.status === "ok" && <span className="z-dot ok" />}
            {st && st.status === "off" && <span className="z-dot off" />}
          </div>;
        })}
      </div>)}
    </div>
    <div className="z-side-foot"><span className="z-pulse" />Running · {Z.meta.uptime}</div>
  </div>;
}
