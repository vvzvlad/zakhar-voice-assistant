import React, { useState, useEffect } from "react";
import { Field, PageHeader, FormSaveBar, Select, Stepper } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { fmtUptime, fmtStarted, fmtBytes } from "../format.js";

function Card({ title, sub, children, foot }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
    {children !== undefined && <div className="z-card-b">{children}</div>}
    {foot}
  </div>;
}

// ── System page (core.network + core.audio + core.agent_mcp + core.runs) ──
// Named SystemPage to avoid clashing with the exported System status card below.
function SystemPage() {
  const { catalog, patch } = useAppData();
  const defs = catalog.core.schema.$defs || {};
  const netSchema = defs.NetworkConfig;
  const audioSchema = defs.AudioConfig;
  const mcpSchema = defs.AgentMcpConfig;

  const netValues = catalog.core.values.network || {};
  const audioValues = catalog.core.values.audio || {};
  const mcpValues = catalog.core.values.agent_mcp || {};

  const net = useStageForm(netValues, (d) => ({ core: { network: d } }), patch);
  const audio = useStageForm(audioValues, (d) => ({ core: { audio: d } }), patch);
  const amcp = useStageForm(mcpValues, (d) => ({ core: { agent_mcp: d } }), patch);

  // The MCP endpoint is served by this very panel, so it is same-origin: no
  // host/port math, just the page's own origin plus /mcp.
  const mcpEndpoint = `${window.location.origin}/mcp`;

  return <div className="z-page">
    <PageHeader title="System" crumb="Operations · advanced"
      desc="Outbound routing for cloud APIs, the audio server that feeds speakers, the MCP server for other agents, and recorded-utterance storage." />
    {/* Two equal columns: routing/audio on the left, system status + recordings + logging on the right */}
    <div className="z-cols even">
      <div className="z-grid">
        <Card title="External proxy" foot={<FormSaveBar dirty={net.dirty} saving={net.saving} onSave={net.save} errors={errorLines(net.err)} />}>
          {netSchema
            ? <SchemaForm schema={netSchema} values={net.draft} onChange={net.onChange} />
            : <Field label="Proxy" hint="SOCKS/HTTP for cloud APIs. Empty = direct."><div className="z-inp mono"><input value={net.draft.external_proxy ?? ""} onChange={(e) => net.onChange("external_proxy", e.target.value)} /></div></Field>}
        </Card>
        <Card title="Audio server" sub="serves generated audio to speakers" foot={<FormSaveBar dirty={audio.dirty} saving={audio.saving} onSave={audio.save} errors={errorLines(audio.err)} />}>
          {audioSchema
            ? <SchemaForm schema={audioSchema} values={audio.draft} onChange={audio.onChange} />
            : <>
              <Field label="Bind host"><div className="z-inp mono"><input value={audio.draft.host ?? ""} onChange={(e) => audio.onChange("host", e.target.value)} /></div></Field>
              <Field label="Port" row><Stepper value={audio.draft.port ?? 8200} onChange={(v) => audio.onChange("port", v)} /></Field>
              <Field label="Cache TTL" hint="Seconds." row><Stepper value={audio.draft.ttl ?? 300} step={30} onChange={(v) => audio.onChange("ttl", v)} unit="s" /></Field>
            </>}
        </Card>
        <Card title="MCP Server for other agents" sub="streamable-HTTP MCP endpoint on this panel's port" foot={<FormSaveBar dirty={amcp.dirty} saving={amcp.saving} onSave={amcp.save} errors={errorLines(amcp.err)} />}>
          {mcpSchema && <SchemaForm schema={mcpSchema} values={amcp.draft} onChange={amcp.onChange} />}
          <div style={{ marginTop: 6, fontSize: 12.5, color: "var(--mut)", lineHeight: 1.45 }}>
            A connected agent can: read the request/reply log, read &amp; live-patch the
            config, list speakers, speak text on a speaker, and send text commands
            through the assistant (LLM + smart-home tools).
          </div>
          <div style={{ marginTop: 4 }}>
            <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Endpoint</div>
            <div className="mono" style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{mcpEndpoint}</div>
          </div>
        </Card>
      </div>
      {/* Right column: system status + recordings storage + logging. */}
      <div className="z-grid">
        <System />
        <Recordings />
      </div>
    </div>
  </div>;
}

// ── Recordings (core.runs.audio_keep) + DB-on-disk indicator ───────────────
function Recordings() {
  const { catalog, patch, system } = useAppData();
  const runsSchema = (catalog.core.schema.$defs || {}).RunsConfig;
  const runsValues = catalog.core.values.runs || {};
  // useStageForm keeps the WHOLE runs object as the draft; we render only the
  // audio_keep field, so the other runs fields are saved back unchanged.
  const runs = useStageForm(runsValues, (d) => ({ core: { runs: d } }), patch);

  return <Card title="Recordings" sub="utterance audio kept for playback in the log"
    foot={<FormSaveBar dirty={runs.dirty} saving={runs.saving} onSave={runs.save} errors={errorLines(runs.err)} />}>
    {runsSchema
      && <SchemaForm schema={runsSchema} values={runs.draft} onChange={runs.onChange}
        skip={["enabled", "retention_days", "store_audio"]} />}
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Database on disk</div>
      <div className="mono" style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{fmtBytes(system?.db_size_bytes)}</div>
    </div>
  </Card>;
}

// ── System (from /api/system; log_level editable) ─────────────────────────
export function System() {
  const { system, patch, catalog, connected } = useAppData();
  const curLevel = catalog.core.values.log_level || system?.log_level || "INFO";
  const [level, setLevel] = useState(curLevel);
  const [savingLvl, setSavingLvl] = useState(false);
  const [lvlErr, setLvlErr] = useState(null);

  // Re-seed the select whenever the persisted level changes (e.g. after a save).
  useEffect(() => { setLevel(curLevel); }, [curLevel]);
  const dirty = level !== curLevel;

  const saveLevel = async () => {
    setSavingLvl(true); setLvlErr(null);
    try { await patch({ core: { log_level: level } }); }
    catch (e) { setLvlErr(e); }
    finally { setSavingLvl(false); }
  };

  const cell = (label, value, mono) => (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{value}</div>
    </div>
  );

  return <div className="z-grid">
    <Card title="Application status">
      <div style={{ padding: "8px 0 6px", display: "flex", gap: 22, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>State</div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 16, fontWeight: 600, marginTop: 3 }}>
            <span className={"z-pulse" + (connected ? "" : " off")} />{connected ? "Running" : "Disconnected"}
          </div>
        </div>
        {cell("Version", system?.version || "—", true)}
        {cell("Uptime", fmtUptime(system?.uptime_seconds), true)}
        {cell("Started", fmtStarted(system?.started), true)}
      </div>
    </Card>
    <Card title="Logging" foot={<FormSaveBar dirty={dirty} saving={savingLvl} onSave={saveLevel} errors={errorLines(lvlErr)} />}>
      <Field label="Log level" hint="Verbosity of server logs."><div style={{ maxWidth: 220 }}><Select value={level} options={["DEBUG", "INFO", "WARNING", "ERROR"]} onChange={setLevel} /></div></Field>
    </Card>
  </div>;
}

export default SystemPage;
