import React, { useState } from "react";
import Z from "../data.js";
import { NAV, TITLES } from "../nav.js";
import { Ic } from "./icons.jsx";
import { Modal } from "./primitives.jsx";

export function Topbar({ active }) {
  const [restart, setRestart] = useState(false);
  return <div className="z-top">
    <div className="z-bcrumb"><s>{NAV.find((g) => g.items.some(([i]) => i === active)).grp} / </s>{TITLES[active]}</div>
    <div className="sp" />
    <span className="z-env"><span className="z-pulse" />Running · {Z.meta.uptime}</span>
    <div className="z-env">{Z.meta.version}</div>
    {Z.meta.pendingRestart && <span className="z-pill warn" style={{ cursor: "pointer" }} onClick={() => setRestart(true)}><Ic n="restart" w={12} />Restart pending</span>}
    {restart && <Modal title="Restart service?" onClose={() => setRestart(false)} footer={<><button className="z-btn g" onClick={() => setRestart(false)}>Cancel</button><button className="z-btn warn" onClick={() => setRestart(false)}>Restart now</button></>}>
      <div className="z-note" style={{ padding: "8px 0 12px" }}>Backends, device connections and the audio server will be recreated. Active dialogs are preserved. Expected downtime <b>~3 s</b>.</div>
    </Modal>}
  </div>;
}
