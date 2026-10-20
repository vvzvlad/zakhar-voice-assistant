import React from "react";
import { Card, Field, FormPage, PageHeader, Select, Stepper } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { usePageForm } from "../usePageForm.js";
import { fmtUptime, fmtStarted, fmtBytes } from "../format.js";

// ── System page (core.network + core.audio + core.agent_mcp + core.runs + core.log_level) ──
// Every card is a part of one page form: the page's one Save sends them as one PATCH.
function SystemPage() {
  const { catalog, patch, system } = useAppData();
  const defs = catalog.core.schema.$defs || {};
  const netSchema = defs.NetworkConfig;
  const audioSchema = defs.AudioConfig;
  const mcpSchema = defs.AgentMcpConfig;
  const values = catalog.core.values;
  const section = (key) => ({ key, values: values[key] || {}, toPatch: (d) => ({ core: { [key]: d } }) });
  const form = usePageForm([
    section("network"), section("audio"), section("agent_mcp"), section("runs"),
    {
      key: "log_level",
      values: { log_level: values.log_level || system?.log_level || "INFO" },
      toPatch: (d) => ({ core: { log_level: d.log_level } }),
    },
  ], patch);
  const net = { draft: form.draftOf("network"), onChange: form.onChangeOf("network") };
  const audio = { draft: form.draftOf("audio"), onChange: form.onChangeOf("audio") };

  // The MCP endpoint is served by this very panel, so it is same-origin: no
  // host/port math, just the page's own origin plus /mcp.
  const mcpEndpoint = `${window.location.origin}/mcp`;

  return <FormPage form={form}>
    <PageHeader title="System"
      desc="Outbound routing for cloud APIs, the device server that voice devices connect to, the MCP server for other agents, and recorded-utterance storage." />
    {/* Two equal columns: routing/audio on the left, system status + recordings + logging on the right */}
    <div className="z-cols even">
      <div className="z-grid">
        <Card title="External proxy">
          {netSchema
            ? <SchemaForm schema={netSchema} values={net.draft} onChange={net.onChange} />
            : <Field label="Proxy" hint="SOCKS or HTTP proxy for cloud APIs; empty connects directly."><div className="z-inp mono"><input value={net.draft.external_proxy ?? ""} onChange={(e) => net.onChange("external_proxy", e.target.value)} /></div></Field>}
        </Card>
        <Card title="Device server" sub="voice devices connect to /device over WebSocket">
          {audioSchema
            ? <SchemaForm schema={audioSchema} values={audio.draft} onChange={audio.onChange} />
            : <>
              <Field label="Bind host"><div className="z-inp mono"><input value={audio.draft.host ?? ""} onChange={(e) => audio.onChange("host", e.target.value)} /></div></Field>
              <Field label="Port" row><Stepper value={audio.draft.port ?? 8200} onChange={(v) => audio.onChange("port", v)} /></Field>
            </>}
        </Card>
        <Card title="MCP Server for other agents" sub="streamable-HTTP MCP endpoint on this panel's port">
          {mcpSchema && <SchemaForm schema={mcpSchema} values={form.draftOf("agent_mcp")} onChange={form.onChangeOf("agent_mcp")} />}
          <div style={{ marginTop: 6, fontSize: 12.5, color: "var(--mut)", lineHeight: 1.45 }}>
            A connected agent can: read the request/reply log, read &amp; live-patch the
            config, list devices, speak text on a device, and send text commands
            through the assistant (LLM + smart-home tools).
          </div>
          <div style={{ marginTop: 4 }}>
            <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Endpoint</div>
            <div className="mono" style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{mcpEndpoint}</div>
          </div>
        </Card>
      </div>
      {/* Right column: system status + logging + recordings storage. */}
      <div className="z-grid">
        <Status />
        <Card title="Logging">
          <Field label="Log level" hint="Verbosity of server logs."><div style={{ maxWidth: 220 }}>
            <Select value={form.draftOf("log_level").log_level} options={["DEBUG", "INFO", "WARNING", "ERROR"]}
              onChange={(v) => form.onChangeOf("log_level")("log_level", v)} />
          </div></Field>
        </Card>
        <Recordings draft={form.draftOf("runs")} onChange={form.onChangeOf("runs")} />
      </div>
    </div>
  </FormPage>;
}

// ── Recordings (core.runs.audio_keep) + DB-on-disk indicator ───────────────
// The draft is the WHOLE runs object; only audio_keep is rendered, so the other
// runs fields are saved back unchanged.
function Recordings({ draft, onChange }) {
  const { catalog, system } = useAppData();
  const runsSchema = (catalog.core.schema.$defs || {}).RunsConfig;
  return <Card title="Recordings" sub="utterance audio kept for playback in the log">
    {runsSchema
      && <SchemaForm schema={runsSchema} values={draft} onChange={onChange}
        skip={["enabled", "retention_days", "store_audio"]} />}
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>Database on disk</div>
      <div className="mono" style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{fmtBytes(system?.db_size_bytes)}</div>
    </div>
  </Card>;
}

// ── Application status (from /api/system) ──────────────────────────────────
function Status() {
  const { system, connected } = useAppData();
  const cell = (label, value, mono) => (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: 15, fontWeight: 600, marginTop: 3 }}>{value}</div>
    </div>
  );
  return <Card title="Application status">
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
  </Card>;
}

export default SystemPage;
