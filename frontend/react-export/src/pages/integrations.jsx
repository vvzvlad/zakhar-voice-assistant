import React, { useEffect, useState } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Field, PageHeader, FormSaveBar, StatusPill, Modal, Stepper, Loading,
} from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getPrompt, putPrompt, getDevices } from "../api.js";

function Card({ title, sub, children, foot, right }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}{right}</div>}
    {children !== undefined && <div className="z-card-b">{children}</div>}
    {foot}
  </div>;
}

// ── MCP (single server bound to core.mcp) ─────────────────────────────────
export function MCP() {
  const { catalog, patch } = useAppData();
  const mcpSchema = catalog.core.schema.$defs?.McpConfig;
  const mcpValues = catalog.core.values.mcp || { url: "", token: "" };
  const buildPatch = (draft) => ({ core: { mcp: draft } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(mcpValues, buildPatch, patch);

  return <div className="z-page">
    <PageHeader title="MCP server" desc="Smart-home integration the LLM calls for tools. One server for now — multi-server support is coming." />
    <div className="z-banner warn" style={{ margin: "0 0 14px" }}>
      <Ic n="mcp" w={15} />
      <span><b>Один сервер.</b> Несколько MCP-серверов, список инструментов и пер-серверные промпты появятся позже.</span>
    </div>
    <Card title="node-red.home" foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
      {mcpSchema
        ? <SchemaForm schema={mcpSchema} values={draft} onChange={onChange} />
        : <>
          <Field label="Endpoint URL" hint="Empty = smart home disabled.">
            <div className="z-inp mono"><input value={draft.url ?? ""} placeholder="http://10.0.0.5:8001/mcp" onChange={(e) => onChange("url", e.target.value)} /></div>
          </Field>
          <Field label="Bearer token" hint="Empty = no auth.">
            <div className="z-inp mono"><input type="password" value={draft.token ?? ""} placeholder="— none —" onChange={(e) => onChange("token", e.target.value)} /></div>
          </Field>
        </>}
    </Card>
  </div>;
}

// ── System prompt ─────────────────────────────────────────────────────────
export function Prompt() {
  const [text, setText] = useState(null);
  const [loaded, setLoaded] = useState("");
  const [path, setPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);

  const load = () => {
    getPrompt().then((r) => { setText(r.text); setLoaded(r.text); setPath(r.path); setErr(null); })
      .catch((e) => setErr(e));
  };
  useEffect(() => { load(); }, []);

  const dirty = text != null && text !== loaded;
  const save = async () => {
    setSaving(true); setErr(null);
    try { await putPrompt(text); setLoaded(text); }
    catch (e) { setErr(e); }
    finally { setSaving(false); }
  };

  if (text == null) return <div className="z-page"><div className="z-card"><Loading /></div></div>;

  return <div className="z-page">
    <PageHeader title="System prompt" crumb="Integrations"
      desc="Zahar's character, rules and answer format. The placeholder is replaced with live date/time at request time."
      actions={<>
        <button className="z-btn g" onClick={load}>Reload</button>
        <button className="z-btn p" disabled={!dirty || saving} onClick={save}>{saving ? "Saving…" : "Save"}</button>
      </>} />
    <div className="z-cols wide">
      <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{text.length} chars · {path}</span>} title="Editor"
        foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
        <div style={{ padding: "8px 0" }}>
          <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
            style={{ width: "100%", minHeight: 560, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
        </div>
      </Card>
      <div className="z-aside">
        <Card title="The date/time variable">
          <div className="z-info" style={{ padding: "10px 0 12px" }}>
            <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<TDW>>>>>"}</span> is substituted at request time with the current <b>date and time</b>.
          </div>
        </Card>
        <Card title="Notation">
          <div className="z-info" style={{ padding: "10px 0 12px" }}>Stress marks use <b>«+»</b> before the stressed vowel (e.g. «з+амок»).</div>
        </Card>
      </div>
    </div>
  </div>;
}

// ── Context (core.context) ────────────────────────────────────────────────
export function Context() {
  const { catalog, patch } = useAppData();
  const ctxSchema = catalog.core.schema.$defs?.ContextConfig;
  const ctxValues = catalog.core.values.context || {};
  const buildPatch = (draft) => ({ core: { context: draft } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(ctxValues, buildPatch, patch);

  return <div className="z-page">
    <PageHeader title="Dialog context" crumb="Integrations"
      desc="How many past turns the assistant remembers, and how quickly it forgets. Stored separately per speaker." />
    <Card title="Memory" foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
      {ctxSchema
        ? <SchemaForm schema={ctxSchema} values={draft} onChange={onChange} skip={["dir"]} />
        : <>
          <Field label="Context depth" hint="Recent Q&A pairs sent to the model." row><Stepper value={draft.max_turns ?? 5} min={1} onChange={(v) => onChange("max_turns", v)} unit="turns" /></Field>
          <Field label="Dialog TTL" hint="Idle time before a dialog resets (0 = always fresh)." row><Stepper value={draft.ttl_seconds ?? 300} min={0} step={30} onChange={(v) => onChange("ttl_seconds", v)} unit="s" /></Field>
        </>}
    </Card>
  </div>;
}

// ── Devices ───────────────────────────────────────────────────────────────
function DeviceModal({ initial, onSave, onClose, title }) {
  const [name, setName] = useState(initial?.name || "");
  const [host, setHost] = useState(initial?.host || "");
  const [psk, setPsk] = useState(initial?.psk || "");
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!name || !host} onClick={() => onSave({ name, host, psk })}>Save</button></>}>
    <Field label="Name" hint="Unique — also keys the dialog context."><div className="z-inp"><input value={name} placeholder="e.g. hallway" onChange={(e) => setName(e.target.value)} /></div></Field>
    <Field label="Host / IP"><div className="z-inp mono"><input value={host} placeholder="10.0.0.25" onChange={(e) => setHost(e.target.value)} /></div></Field>
    <Field label="PSK" hint="ESPHome API encryption key."><div className="z-inp mono"><input value={psk} placeholder="base64 key…" onChange={(e) => setPsk(e.target.value)} /></div></Field>
  </Modal>;
}

export function Devices() {
  const { catalog, patch } = useAppData();
  const devices = catalog.core.values.devices || [];
  const esphomePort = catalog.core.values.esphome?.port ?? 6053;

  const [live, setLive] = useState([]);
  useEffect(() => {
    let alive = true;
    getDevices().then((d) => { if (alive && Array.isArray(d)) setLive(d); }).catch(() => {});
    return () => { alive = false; };
  }, [catalog]);
  const statusOf = (name) => {
    const m = live.find((d) => d.name === name);
    if (!m) return "off";
    return m.online ? "online" : "offline";
  };

  const [modal, setModal] = useState(null); // { mode:'add'|'edit', index }
  const [busyErr, setBusyErr] = useState(null);

  const saveList = async (list) => {
    setBusyErr(null);
    try { await patch({ core: { devices: list } }); setModal(null); }
    catch (e) { setBusyErr(e); }
  };
  const onAdd = (d) => saveList([...devices, d]);
  const onEdit = (i, d) => saveList(devices.map((x, idx) => (idx === i ? d : x)));
  const onDelete = (i) => saveList(devices.filter((_, idx) => idx !== i));

  // Common params (esphome.port + audio.public_base_url) as their own save form.
  const commonValues = {
    port: esphomePort,
    public_base_url: catalog.core.values.audio?.public_base_url ?? "",
  };
  const buildCommon = (d) => ({ core: { esphome: { port: d.port }, audio: { public_base_url: d.public_base_url } } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(commonValues, buildCommon, patch);

  return <div className="z-page">
    <PageHeader title="Devices" desc="ESPHome speakers the server connects to. Each name also keys its own dialog context."
      actions={<button className="z-btn p" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add speaker</button>} />
    {busyErr && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span>{errorLines(busyErr).join(" · ")}</span></div>}
    <Card>
      <table className="z-tbl">
        <thead><tr><th>Name</th><th>Host / IP</th><th>PSK</th><th>Status</th><th></th></tr></thead>
        <tbody>
          {devices.length === 0
            ? <tr><td colSpan={5} style={{ color: "var(--mut)", padding: "14px 0" }}>No speakers configured.</td></tr>
            : devices.map((d, i) => <tr key={i} style={{ cursor: "default" }}>
              <td style={{ fontWeight: 600 }}>{d.name}</td>
              <td className="mono" style={{ fontSize: 11.5 }}>{d.host}<span style={{ color: "var(--mut2)" }}>:{esphomePort}</span></td>
              <td className="mono" style={{ fontSize: 11.5, color: "var(--mut)" }}>{d.psk ? "••••••••" : "—"}</td>
              <td><StatusPill status={statusOf(d.name)} /></td>
              <td style={{ textAlign: "right" }}><div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                <button className="z-mini" onClick={() => setModal({ mode: "edit", index: i })}>Edit</button>
                <button className="z-mini" onClick={() => onDelete(i)}>Delete</button>
              </div></td>
            </tr>)}
        </tbody>
      </table>
    </Card>
    <div className="z-sl">Common parameters<div className="ln" /></div>
    <Card foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
      <Field label="ESPHome API port" hint="Default 6053." row><Stepper value={draft.port ?? 6053} onChange={(v) => onChange("port", v)} /></Field>
      <Field label="Public base URL" hint="Where speakers download generated audio. If wrong — the speaker won't play the reply.">
        <div className="z-inp mono"><input value={draft.public_base_url ?? ""} placeholder="http://10.0.0.10:8200" onChange={(e) => onChange("public_base_url", e.target.value)} /></div>
      </Field>
    </Card>
    {modal?.mode === "add" && <DeviceModal title="Add speaker" onSave={onAdd} onClose={() => setModal(null)} />}
    {modal?.mode === "edit" && <DeviceModal title="Edit speaker" initial={devices[modal.index]} onSave={(d) => onEdit(modal.index, d)} onClose={() => setModal(null)} />}
  </div>;
}
