import React, { useState } from "react";
import { NAV, TITLES } from "../nav.js";
import { Ic } from "./icons.jsx";
import { Modal } from "./primitives.jsx";
import { useAppData } from "../appData.jsx";
import { fmtUptime } from "../format.js";

export function Topbar({ active }) {
  const { system, pendingRestart, restart, refreshSystem } = useAppData();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const grp = NAV.find((g) => g.items.some(([i]) => i === active));

  const doRestart = async () => {
    setBusy(true);
    try { await restart(); } catch { /* ignore */ }
    setBusy(false);
    setOpen(false);
    setTimeout(refreshSystem, 1500);
  };

  return <div className="z-top">
    <div className="z-bcrumb"><s>{grp ? grp.grp + " / " : ""}</s>{TITLES[active] || active}</div>
    <div className="sp" />
    <span className="z-env"><span className="z-pulse" />Running · {fmtUptime(system?.uptime_seconds)}</span>
    <div className="z-env">{system?.version || "—"}</div>
    {pendingRestart && <span className="z-pill warn" style={{ cursor: "pointer" }} onClick={() => setOpen(true)}><Ic n="restart" w={12} />Restart pending</span>}
    {open && <Modal title="Restart service?" onClose={() => setOpen(false)}
      footer={<><button className="z-btn g" onClick={() => setOpen(false)}>Cancel</button><button className="z-btn warn" disabled={busy} onClick={doRestart}>{busy ? "Restarting…" : "Restart now"}</button></>}>
      <div className="z-note" style={{ padding: "8px 0 12px" }}>Backends, device connections and the audio server will be recreated. Active dialogs are preserved. Expected downtime <b>~3 s</b>.</div>
    </Modal>}
  </div>;
}
