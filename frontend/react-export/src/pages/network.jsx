import React, { useState, useEffect, useRef } from "react";
import Z from "../data.js";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { Spark, Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, StatusPill, Waterfall, segsFor, total, PageHeader, SaveBar, Modal, Player, KV } from "../components/primitives.jsx";
  function Card({ title, sub, children, foot }) {
    return <div className="z-card">
      {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
      {children !== undefined && <div className="z-card-b">{children}</div>}
      {foot}
    </div>;
  }

  // ── Network ────────────────────────────────────────────
  function Network() {
    const n = Z.network;
    return <div className="z-page">
      <PageHeader title="Network & integrations" crumb="Operations · advanced" desc="Outbound routing for cloud APIs and the audio server that feeds speakers." />
      <div>
          <Card title="External proxy" foot={<SaveBar restart />}>
            <Field label="Proxy" param="EXTERNAL_PROXY" hint="SOCKS/HTTP for cloud APIs (STT / LLM / Yandex TTS). Empty = direct — all cloud calls go straight out."><div className="z-inp mono"><input value={n.EXTERNAL_PROXY} readOnly /><button className="z-mini">TEST</button></div></Field>
          </Card>
          <div style={{ height: 16 }} />
          <Card title="Audio server" sub="serves generated audio to speakers" foot={<SaveBar restart noTest />}>
            <Field label="Bind address" param="AUDIO_HOST:AUDIO_PORT" hint="Host and port the audio server listens on."><div className="z-inp mono"><input value={n.AUDIO_HOST + ":" + n.AUDIO_PORT} readOnly /></div></Field>
            <Field label="Cache TTL" param="AUDIO_TTL" hint="How long an mp3 lives in cache, seconds." row><Stepper value={n.AUDIO_TTL} step={30} onChange={() => {}} unit="s" /></Field>
          </Card>
      </div>
    </div>;
  }

  // ── System ─────────────────────────────────────────────
  function System() {
    const [level, setLevel] = useState(Z.meta.logLevel);
    const [restarting, setRestarting] = useState(false);
    const pending = [
      { stage: "TTS", what: "voice → zahar, emotion neutral" },
      { stage: "VAD", what: "trailing silence 800 → 700 ms" }
    ];
    return <div className="z-page">
      <PageHeader title="System" crumb="Operations" desc="Service status, logging and lifecycle." />
      <div className="z-cols">
        <div>
          <Card title="Application status">
            <div style={{ padding: "8px 0 6px", display: "flex", gap: 22, flexWrap: "wrap" }}>
              <div><div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>State</div><div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 16, fontWeight: 600, marginTop: 3 }}><span className="z-pulse" />Running</div></div>
              <div><div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Version</div><div className="mono" style={{ fontSize: 16, fontWeight: 600, marginTop: 3 }}>{Z.meta.version}</div></div>
              <div><div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Uptime</div><div className="mono" style={{ fontSize: 16, fontWeight: 600, marginTop: 3 }}>{Z.meta.uptime}</div></div>
              <div><div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Started</div><div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--ink2)" }}>{Z.meta.started}</div></div>
            </div>
          </Card>
          <div style={{ height: 16 }} />
          <Card title="Logging" foot={<SaveBar restart />}>
            <Field label="Log level" param="LOG_LEVEL" hint="Verbosity of server logs."><div style={{ maxWidth: 220 }}><Select value={level} options={["DEBUG", "INFO", "WARNING", "ERROR"]} onChange={setLevel} /></div></Field>
          </Card>
        </div>
        <div className="z-aside">
          <Card title="Lifecycle">
            <div style={{ padding: "10px 0 12px" }}>
              {Z.meta.pendingRestart && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span><b>{pending.length} changes</b> need a restart to take effect.</span></div>}
              <div style={{ marginBottom: 12 }}>
                {pending.map((p, i) => <div key={i} style={{ display: "flex", gap: 9, padding: "7px 0", borderBottom: i < pending.length - 1 ? "1px solid var(--line2)" : "none", fontSize: 12 }}>
                  <span className="z-paramtag" style={{ flex: "0 0 auto" }}>{p.stage}</span><span style={{ color: "var(--ink2)" }}>{p.what}</span>
                </div>)}
              </div>
              <button className="z-btn warn" style={{ width: "100%", justifyContent: "center" }} onClick={() => setRestarting(true)}><Ic n="restart" w={14} />Restart service</button>
            </div>
          </Card>
        </div>
      </div>
      {restarting && <Modal title="Restart service?" onClose={() => setRestarting(false)}
        footer={<><button className="z-btn g" onClick={() => setRestarting(false)}>Cancel</button><button className="z-btn warn" onClick={() => setRestarting(false)}>Restart now</button></>}>
        <div className="z-note" style={{ padding: "8px 0 12px" }}>Backends, device connections and the audio server will be recreated. Active dialogs are preserved. Expected downtime: <b>~3 s</b>.</div>
      </Modal>}
    </div>;
  }

export default Network;
