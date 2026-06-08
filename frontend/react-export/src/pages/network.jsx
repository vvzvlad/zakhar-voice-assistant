import React, { useState, useEffect } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Field, PageHeader, FormSaveBar, Select, Stepper, Modal,
} from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { fmtUptime, fmtStarted } from "../format.js";

function Card({ title, sub, children, foot }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
    {children !== undefined && <div className="z-card-b">{children}</div>}
    {foot}
  </div>;
}

// ── Network (core.network + core.audio) ───────────────────────────────────
function Network() {
  const { catalog, patch } = useAppData();
  const defs = catalog.core.schema.$defs || {};
  const netSchema = defs.NetworkConfig;
  const audioSchema = defs.AudioConfig;

  const netValues = catalog.core.values.network || {};
  const audioValues = catalog.core.values.audio || {};

  const net = useStageForm(netValues, (d) => ({ core: { network: d } }), patch);
  const audio = useStageForm(audioValues, (d) => ({ core: { audio: d } }), patch);

  return <div className="z-page">
    <PageHeader title="Network & integrations" crumb="Operations · advanced"
      desc="Outbound routing for cloud APIs and the audio server that feeds speakers." />
    <Card title="External proxy" foot={<FormSaveBar dirty={net.dirty} saving={net.saving} onSave={net.save} restart errors={errorLines(net.err)} />}>
      {netSchema
        ? <SchemaForm schema={netSchema} values={net.draft} onChange={net.onChange} />
        : <Field label="Proxy" hint="SOCKS/HTTP for cloud APIs. Empty = direct."><div className="z-inp mono"><input value={net.draft.external_proxy ?? ""} onChange={(e) => net.onChange("external_proxy", e.target.value)} /></div></Field>}
    </Card>
    <div style={{ height: 16 }} />
    <Card title="Audio server" sub="serves generated audio to speakers" foot={<FormSaveBar dirty={audio.dirty} saving={audio.saving} onSave={audio.save} restart errors={errorLines(audio.err)} />}>
      {audioSchema
        ? <SchemaForm schema={audioSchema} values={audio.draft} onChange={audio.onChange} />
        : <>
          <Field label="Bind host"><div className="z-inp mono"><input value={audio.draft.host ?? ""} onChange={(e) => audio.onChange("host", e.target.value)} /></div></Field>
          <Field label="Port" row><Stepper value={audio.draft.port ?? 8200} onChange={(v) => audio.onChange("port", v)} /></Field>
          <Field label="Cache TTL" hint="Seconds." row><Stepper value={audio.draft.ttl ?? 300} step={30} onChange={(v) => audio.onChange("ttl", v)} unit="s" /></Field>
        </>}
    </Card>
    <div className="z-sl">System<div className="ln" /></div>
    <System />
  </div>;
}

// ── System (from /api/system; log_level editable; restart) ────────────────
export function System() {
  const { system, patch, restart, refreshSystem, catalog } = useAppData();
  const curLevel = catalog.core.values.log_level || system?.log_level || "INFO";
  const [level, setLevel] = useState(curLevel);
  const [savingLvl, setSavingLvl] = useState(false);
  const [lvlErr, setLvlErr] = useState(null);
  const [restarting, setRestarting] = useState(false);
  const [busy, setBusy] = useState(false);

  // Re-seed the select whenever the persisted level changes (e.g. after a save).
  useEffect(() => { setLevel(curLevel); }, [curLevel]);
  const dirty = level !== curLevel;

  const saveLevel = async () => {
    setSavingLvl(true); setLvlErr(null);
    try { await patch({ core: { log_level: level } }); }
    catch (e) { setLvlErr(e); }
    finally { setSavingLvl(false); }
  };

  const doRestart = async () => {
    setBusy(true);
    try { await restart(); } catch { /* ignore */ }
    setBusy(false); setRestarting(false);
    setTimeout(refreshSystem, 1500);
  };

  const cell = (label, value, mono) => (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{value}</div>
    </div>
  );

  return <>
    <div className="z-cols">
      <div>
        <Card title="Application status">
          <div style={{ padding: "8px 0 6px", display: "flex", gap: 22, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>State</div>
              <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 16, fontWeight: 600, marginTop: 3 }}><span className="z-pulse" />{system?.running ? "Running" : "Stopped"}</div>
            </div>
            {cell("Version", system?.version || "—", true)}
            {cell("Uptime", fmtUptime(system?.uptime_seconds), true)}
            {cell("Started", fmtStarted(system?.started), true)}
          </div>
        </Card>
        <div style={{ height: 16 }} />
        <Card title="Logging" foot={<FormSaveBar dirty={dirty} saving={savingLvl} onSave={saveLevel} errors={errorLines(lvlErr)} />}>
          <Field label="Log level" hint="Verbosity of server logs."><div style={{ maxWidth: 220 }}><Select value={level} options={["DEBUG", "INFO", "WARNING", "ERROR"]} onChange={setLevel} /></div></Field>
        </Card>
      </div>
      <div className="z-aside">
        <Card title="Lifecycle">
          <div style={{ padding: "10px 0 12px" }}>
            {system?.pending_restart && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span>There are changes not yet applied to runtime — a restart is required.</span></div>}
            <button className="z-btn warn" style={{ width: "100%", justifyContent: "center" }} onClick={() => setRestarting(true)}><Ic n="restart" w={14} />Restart service</button>
          </div>
        </Card>
      </div>
    </div>
    {restarting && <Modal title="Restart service?" onClose={() => setRestarting(false)}
      footer={<><button className="z-btn g" onClick={() => setRestarting(false)}>Cancel</button><button className="z-btn warn" disabled={busy} onClick={doRestart}>{busy ? "Restarting…" : "Restart now"}</button></>}>
      <div className="z-note" style={{ padding: "8px 0 12px" }}>Backends, device connections and the audio server will be recreated. Active dialogs are preserved. Expected downtime: <b>~3 s</b>.</div>
    </Modal>}
  </>;
}

export default Network;
