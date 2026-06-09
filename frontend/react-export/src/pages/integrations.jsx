import React, { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Field, KeyInput, PageHeader, FormSaveBar, StatusPill, Pill, Modal, Stepper, Loading, Select,
} from "../components/primitives.jsx";
import SchemaForm, { schemaNeedsRestart } from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { deref } from "../schema.js";
import { getPrompt, putPrompt, getDevices, getTools, startCapture, getCaptureStatus, downloadCaptureResult } from "../api.js";

function Card({ title, sub, children, foot, right }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}{right}</div>}
    {children !== undefined && <div className="z-card-b">{children}</div>}
    {foot}
  </div>;
}

// ── Tool sources (multi-source view of the ToolHub) ───────────────────────
// Three integration cards driven by CONFIG (so each shows even before it is
// configured), each enriched with LIVE status/tools from GET /api/tools matched
// by source id. Sources are hot-reloaded — rebuilt live on change (rebuild_tools),
// so enabling one takes effect immediately, no restart needed.

// Read-only chip row that always renders every advertised tool name.
function ToolChips({ tools }) {
  const list = tools || [];
  return <div className="z-f" style={{ borderBottom: "none" }}>
    <div className="z-fl"><b style={{ fontSize: 12 }}>Advertised tools <span style={{ color: "var(--mut2)", fontWeight: 400 }}>(read-only)</span></b></div>
    {list.length === 0
      ? <div className="z-fh">No tools — the source is not responding or disabled.</div>
      : <div className="z-chiprow">
          {list.map((t) => <span className="z-toolchip" key={t.name} title={t.description || ""}>{t.name}</span>)}
        </div>}
  </div>;
}

// One integration source card: header (name + kind badge + status pill), an
// editable SchemaForm bound to its core.* sub-section, and the live tool chips.
//   id        — source id matched against /api/tools ("home"/"openweathermap"/"calendar")
//   name      — human title; sub — short caption under it
//   schema    — resolved JSON sub-schema (from core.schema, $defs available on root)
//   root      — full core schema (holds $defs for the SchemaForm)
//   values    — current core.<section> values
//   buildPatch(draft) -> patch object; configured(values) -> bool ("configured")
//   live      — matching /api/tools entry, or null when absent
function SourceCard({ id, name, sub, schema, root, values, buildPatch, configured, live, patch }) {
  const { draft, onChange, dirty, saving, err, save } = useStageForm(values, buildPatch, patch);
  const isConfigured = configured(draft);
  const kind = live?.kind || "builtin";

  // Status: online/offline come from the live source; otherwise "not configured"
  // when the relevant config is empty (a configured source absent from /api/tools
  // failed to start — shown as offline rather than "not configured").
  let pill;
  if (live) pill = <StatusPill status={live.online ? "online" : "offline"} />;
  else if (!isConfigured) pill = <Pill tone="muted">not configured</Pill>;
  else pill = <StatusPill status="offline" />;

  return <div className="z-card" style={{ marginBottom: 14 }}>
    <div className="z-card-h">
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <Ic n={kind === "http" ? "mcp" : "network"} w={17} />
        <div style={{ minWidth: 0 }}>
          <b style={{ display: "block" }}>{name}</b>
          {sub && <span style={{ fontSize: 11, color: "var(--mut)" }}>{sub}</span>}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginLeft: "auto" }}>
        <Pill tone={kind === "http" ? "warn" : "muted"}>{kind === "http" ? "external" : "built-in"}</Pill>
        {pill}
      </div>
    </div>
    <div className="z-card-b">
      {schema
        ? <SchemaForm schema={schema} root={root} values={draft} onChange={onChange} />
        : <div className="z-fh">Schema unavailable.</div>}
      <ToolChips tools={live?.tools} />
    </div>
    {/* Sources are hot-reloaded — rebuilt live on save (rebuild_tools), so no restart is needed. */}
    <FormSaveBar dirty={dirty} saving={saving} onSave={save} restart={schemaNeedsRestart(schema)} errors={errorLines(err)} />
  </div>;
}

// ── External MCP server modal (add / edit) ────────────────────────────────
// Fields mirror McpServerConfig: name (unique source id), url, token (masked),
// transport (Literal), prompt (describes the server's tools to the model).
const TRANSPORTS = ["auto", "streamable_http", "sse"];
// Built-in ToolHub source ids — an external server may not shadow them, or it
// would hide the openweathermap/calendar status in /api/tools.
const RESERVED_NAMES = ["openweathermap", "calendar"];
function McpServerModal({ initial, onSave, onClose, title, takenNames }) {
  const [name, setName] = useState(initial?.name || "");
  const [url, setUrl] = useState(initial?.url || "");
  const [token, setToken] = useState(initial?.token || "");
  const [transport, setTransport] = useState(initial?.transport || "auto");
  const [prompt, setPrompt] = useState(initial?.prompt || "");
  // Name must be non-empty, valid URL non-empty, unique among the OTHER servers,
  // and not collide with a reserved built-in source id (case-insensitive).
  const dup = !!name && takenNames.includes(name);
  const reserved = !!name && RESERVED_NAMES.includes(name.trim().toLowerCase());
  const valid = !!name && !!url && !dup && !reserved;
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!valid} onClick={() => onSave({ name, url, token, transport, prompt })}>Save</button></>}>
    <Field label="Name" hint="Unique name — it is also the source id in /api/tools.">
      <div className="z-inp"><input value={name} placeholder="e.g. home" onChange={(e) => setName(e.target.value)} /></div>
      {dup && <div className="z-fh" style={{ color: "#b91c1c" }}>Name is already in use.</div>}
      {!dup && reserved && <div className="z-fh" style={{ color: "#b91c1c" }}>Name is reserved by a built-in source.</div>}
    </Field>
    <Field label="URL"><div className="z-inp mono"><input value={url} placeholder="http://10.0.0.5:8123/mcp_server/sse" onChange={(e) => setUrl(e.target.value)} /></div></Field>
    <Field label="Token" hint="Optional Bearer token.">
      <KeyInput value={token} placeholder="optional…" onChange={setToken} />
    </Field>
    <Field label="Transport" hint="auto detects sse from a URL ending in /sse.">
      <Select value={transport} options={TRANSPORTS} onChange={setTransport} />
    </Field>
    <Field label="Prompt" hint="Describes this server's tools to the model (appended to the system prompt).">
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false}
        style={{ width: "100%", minHeight: 90, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "10px 12px", fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.6, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
    </Field>
  </Modal>;
}

// One external MCP server card: header (name + external tag + status pill), the
// url in mono, and the live tool chips matched from /api/tools by source id ===
// server name. Edit / Delete buttons drive the CRUD flow above.
function McpServerCard({ server, live, onEdit, onDelete }) {
  let pill;
  if (live) pill = <StatusPill status={live.online ? "online" : "offline"} />;
  else pill = <StatusPill status="offline" />;
  return <div className="z-card" style={{ marginBottom: 14 }}>
    <div className="z-card-h">
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <Ic n="mcp" w={17} />
        <div style={{ minWidth: 0 }}>
          <b style={{ display: "block" }}>{server.name}</b>
          <span className="mono" style={{ fontSize: 11, color: "var(--mut)" }}>{server.url}</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginLeft: "auto" }}>
        <Pill tone="warn">external</Pill>
        {pill}
        <div style={{ display: "flex", gap: 6 }}>
          <button className="z-mini" onClick={onEdit}>Edit</button>
          <button className="z-mini" onClick={onDelete}>Delete</button>
        </div>
      </div>
    </div>
    <div className="z-card-b">
      <ToolChips tools={live?.tools} />
    </div>
  </div>;
}

// ── MCP / Integrations: external servers + built-in tool sources ──────────
export function MCP() {
  const { catalog, patch } = useAppData();
  const coreSchema = catalog.core.schema;
  const coreValues = catalog.core.values;
  // properties.<section> is a $ref into $defs; resolve it so SchemaForm gets a
  // schema with `.properties` (and pass the full core schema as `root` for $defs).
  const sub = (key) => deref(coreSchema.properties?.[key] || {}, coreSchema);

  const servers = coreValues.mcp_servers || [];   // [{name,url,token,transport,prompt}]

  const [tools, setTools] = useState(null);   // null = loading; [] = loaded/empty
  const [toolsErr, setToolsErr] = useState(null);
  const loadTools = useCallback(() => {
    getTools()
      .then((r) => { setTools(Array.isArray(r?.sources) ? r.sources : []); setToolsErr(null); })
      .catch((e) => { setTools([]); setToolsErr(e); });
  }, []);
  useEffect(() => { loadTools(); }, [loadTools]);
  const liveOf = (id) => (tools || []).find((s) => s.id === id) || null;

  // Wrap the shared patch() so a successful save also refreshes /api/tools (the
  // source list is rebuilt live on change, so this picks up the new state).
  const patchAndRefresh = useCallback(async (p) => {
    const r = await patch(p);
    loadTools();
    return r;
  }, [patch, loadTools]);

  // External-servers CRUD: full-array replace via patch({ core: { mcp_servers } }),
  // mirroring the Devices page. Sources rebuild live on change, so no restart needed.
  const [modal, setModal] = useState(null); // { mode:'add'|'edit', index }
  const [busyErr, setBusyErr] = useState(null);
  const saveList = async (list) => {
    setBusyErr(null);
    try { await patchAndRefresh({ core: { mcp_servers: list } }); setModal(null); }
    catch (e) { setBusyErr(e); }
  };
  const onAdd = (s) => saveList([...servers, s]);
  const onEdit = (i, s) => saveList(servers.map((x, idx) => (idx === i ? s : x)));
  const onDelete = (i) => saveList(servers.filter((_, idx) => idx !== i));

  return <div className="z-page">
    <PageHeader title="Tool sources" desc="Tool sources the model calls: external smart-home MCP servers and built-in weather/calendar. Sources are applied live — rebuilt on save, no restart needed."
      actions={<button className="z-btn p" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add server</button>} />
    {toolsErr && <div className="z-banner warn" style={{ margin: "0 0 14px" }}>
      <Ic n="restart" w={15} />
      <span><b>Status unavailable.</b> Failed to fetch the tool list: {errorLines(toolsErr).join(" · ")}</span>
    </div>}
    {busyErr && <div className="z-banner warn" style={{ margin: "0 0 12px" }}>
      <Ic n="restart" w={15} /><span>{errorLines(busyErr).join(" · ")}</span>
    </div>}
    {tools === null
      ? <Card><Loading /></Card>
      : <>
        <div className="z-sl">External MCP servers<div className="ln" /></div>
        {servers.length === 0
          ? <Card><div className="z-fh" style={{ padding: "6px 0" }}>No external MCP servers — smart home is unavailable.</div></Card>
          : servers.map((s, i) => <McpServerCard key={s.name || i} server={s} live={liveOf(s.name)}
              onEdit={() => setModal({ mode: "edit", index: i })} onDelete={() => onDelete(i)} />)}
        <div className="z-sl">Built-in sources<div className="ln" /></div>
        <SourceCard
          id="openweathermap" name="OpenWeatherMap (built-in)" sub="core.openweathermap · built-in MCP"
          schema={sub("openweathermap")} root={coreSchema} values={coreValues.openweathermap || { api_key: "", city: "Moscow" }}
          buildPatch={(d) => ({ core: { openweathermap: d } })}
          configured={(v) => !!(v && v.api_key)} live={liveOf("openweathermap")} patch={patchAndRefresh} />
        <SourceCard
          id="calendar" name="Calendar (built-in)" sub="core.calendar · built-in MCP (CalDAV)"
          schema={sub("calendar")} root={coreSchema}
          values={coreValues.calendar || { url: "", username: "", password: "", calendar: "" }}
          buildPatch={(d) => ({ core: { calendar: d } })}
          configured={(v) => !!(v && v.url && v.username)} live={liveOf("calendar")} patch={patchAndRefresh} />
      </>}
    {modal?.mode === "add" && <McpServerModal title="Add MCP server" onSave={onAdd} onClose={() => setModal(null)}
      takenNames={servers.map((s) => s.name)} />}
    {modal?.mode === "edit" && <McpServerModal title="Edit MCP server" initial={servers[modal.index]}
      onSave={(s) => onEdit(modal.index, s)} onClose={() => setModal(null)}
      takenNames={servers.filter((_, idx) => idx !== modal.index).map((s) => s.name)} />}
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

  // Resize the editor textarea to fit its content so the whole prompt is visible
  // without an inner scrollbar. Reset to "auto" first so scrollHeight reflects the
  // content height, not the current box; then add the vertical border so border-box
  // sizing leaves no clipped or overflowing pixels.
  const taRef = useRef(null);
  const autoGrow = useCallback(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    const border = el.offsetHeight - el.clientHeight; // top+bottom borders (no scrollbar with overflow hidden)
    el.style.height = `${el.scrollHeight + border}px`;
  }, []);
  // Recompute height whenever text changes (initial load and every edit) and on
  // window resize, since wrapping (and thus height) depends on the textarea width.
  useLayoutEffect(() => { autoGrow(); }, [text, autoGrow]);
  useEffect(() => {
    window.addEventListener("resize", autoGrow);
    return () => window.removeEventListener("resize", autoGrow);
  }, [autoGrow]);

  const dirty = text != null && text !== loaded;
  const save = async () => {
    setSaving(true); setErr(null);
    try { await putPrompt(text); setLoaded(text); }
    catch (e) { setErr(e); }
    finally { setSaving(false); }
  };

  // Dialog context (core.context) — an independent save form rendered in the aside.
  // All hooks must run before the early-return below to satisfy the rules of hooks.
  const { catalog, patch } = useAppData();
  const ctxSchema = catalog.core.schema.$defs?.ContextConfig;
  const ctxValues = catalog.core.values.context || {};
  const ctxBuildPatch = (d) => ({ core: { context: d } });
  const { draft: ctxDraft, onChange: ctxOnChange, dirty: ctxDirty, saving: ctxSaving, err: ctxErr, save: ctxSave } = useStageForm(ctxValues, ctxBuildPatch, patch);

  // A single header Save persists both the prompt text and the dialog-context
  // settings. Each underlying save handles its own error/saving state, so this
  // never rejects; we just await whichever forms are dirty.
  const anyDirty = dirty || ctxDirty;
  const busy = saving || ctxSaving;
  const saveAll = async () => {
    await Promise.all([dirty ? save() : null, ctxDirty ? ctxSave() : null].filter(Boolean));
  };

  if (text == null) return <div className="z-page"><div className="z-card"><Loading /></div></div>;

  return <div className="z-page">
    <PageHeader title="System prompt" crumb="Integrations"
      desc="Zahar's character, rules and answer format. The placeholder is replaced with live date/time at request time."
      actions={<>
        <button className="z-btn g" onClick={load}>Reload</button>
        <button className="z-btn p" disabled={!anyDirty || busy} onClick={saveAll}>{busy ? "Saving…" : "Save"}</button>
      </>} />
    <div className="z-cols wide">
      <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{text.length} chars · {path}</span>} title="Editor"
        foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
        <div style={{ padding: "8px 0" }}>
          <textarea ref={taRef} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
            style={{ width: "100%", minHeight: 560, resize: "none", overflow: "hidden", boxSizing: "border-box", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
        </div>
      </Card>
      <div className="z-aside">
        <Card title="Dialog context"
          foot={ctxErr ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(ctxErr).join(" · ")}</span></div> : undefined}>
          {ctxSchema
            ? <SchemaForm schema={ctxSchema} values={ctxDraft} onChange={ctxOnChange} skip={["dir"]} />
            : <>
                <Field label="Context depth" hint="Recent Q&A pairs sent to the model." row><Stepper value={ctxDraft.max_turns ?? 5} min={1} onChange={(v) => ctxOnChange("max_turns", v)} unit="turns" /></Field>
                <Field label="Dialog TTL" hint="Idle time before a dialog resets (0 = always fresh)." row><Stepper value={ctxDraft.ttl_seconds ?? 300} min={0} step={30} onChange={(v) => ctxOnChange("ttl_seconds", v)} unit="s" /></Field>
              </>}
        </Card>
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

// ── Devices ───────────────────────────────────────────────────────────────
// The Edit modal also hosts the capture control, but only for an EXISTING device
// (not in Add mode): `device` is the saved name to capture from, `online` whether
// it is reachable. Add mode passes neither, so the control is hidden.
function DeviceModal({ initial, onSave, onClose, title, device, online }) {
  const [name, setName] = useState(initial?.name || "");
  const [host, setHost] = useState(initial?.host || "");
  const [psk, setPsk] = useState(initial?.psk || "");
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!name || !host} onClick={() => onSave({ name, host, psk })}>Save</button></>}>
    <Field label="Name" hint="Unique — also keys the dialog context."><div className="z-inp"><input value={name} placeholder="e.g. hallway" onChange={(e) => setName(e.target.value)} /></div></Field>
    <Field label="Host / IP"><div className="z-inp mono"><input value={host} placeholder="10.0.0.25" onChange={(e) => setHost(e.target.value)} /></div></Field>
    <Field label="PSK" hint="ESPHome API encryption key."><KeyInput value={psk} placeholder="base64 key…" onChange={setPsk} /></Field>
    {device && <CaptureControl device={device} online={online} />}
  </Modal>;
}

// "Record X seconds" control shown inside the Edit-speaker modal. Each recording
// runs as a SERVER-SIDE background task (start -> poll countdown -> download WAV).
// With a count > 1 the browser loops that full cycle back-to-back, downloading one
// WAV per take — so you collect many wake-word samples without re-clicking. A Stop
// button ends the batch after the current take. Disabled (tooltip) while offline.
function CaptureControl({ device, online }) {
  const [seconds, setSeconds] = useState(30);
  const [count, setCount] = useState(1);      // how many takes to record back-to-back
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState("");     // progress / status line
  const [err, setErr] = useState(null);
  const [stopping, setStopping] = useState(false); // Stop pressed, waiting for current take to finish
  const cancelRef = useRef(false);            // Stop button -> end batch after current take

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // One full capture: start the server-side job, poll until terminal, download the WAV.
  // Returns true if a WAV was downloaded, false if the job ended without a result.
  async function captureOnce(idx, total) {
    await startCapture(device, seconds);
    for (;;) {
      const s = await getCaptureStatus(device);
      if (s.state === "recording" || s.state === "cancelled") {
        setPhase(`Recording ${idx}/${total}… ${s.remaining > 0 ? s.remaining + "s left" : "processing"}`);
        await sleep(1000);
        continue;
      }
      if (s.state === "done") { await downloadCaptureResult(device); return true; }
      if (s.state === "error") { throw new Error(s.error || "capture failed"); }
      return false;  // idle / unknown -> no result for this take
    }
  }

  // Run `count` takes sequentially. Stops early on the Stop button or a hard error.
  const run = async () => {
    setErr(null);
    cancelRef.current = false;
    setRunning(true);
    setStopping(false);
    let ok = 0;
    try {
      for (let i = 1; i <= count; i++) {
        if (cancelRef.current) break;
        setPhase(`Recording ${i}/${count}…`);
        if (await captureOnce(i, count)) ok += 1;
      }
      setPhase(cancelRef.current ? `Stopped: ${ok}/${count}` : `Done: ${ok}/${count}`);
    } catch (e) {
      setErr(e.message || "failed");
      setPhase(`Stopped at ${ok}/${count}`);
    } finally {
      setRunning(false);
      setStopping(false);
    }
  };

  return <Field label="Capture sample" hint={`Records ${seconds}s of mic audio and downloads a WAV per take${count > 1 ? ` (×${count})` : ""}. Used for wake-word training.`}>
    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
      {running
        ? <>
            <span className="z-fh">{(phase || "…") + (stopping ? " — will stop after this take" : "")}</span>
            <button
              className={stopping ? "z-btn warn" : "z-btn d"}
              disabled={stopping}
              onClick={() => { cancelRef.current = true; setStopping(true); }}
            >{stopping ? "Stopping…" : "Stop"}</button>
          </>
        : <>
            <Stepper value={seconds} min={1} max={300} onChange={setSeconds} unit="s" />
            <Stepper value={count} min={1} max={1000} onChange={setCount} unit="takes" />
            <button className="z-btn p" disabled={!online} title={online ? "" : "Speaker offline"}
              onClick={run}>{count > 1 ? `Record ×${count}` : "Record sample"}</button>
          </>}
    </div>
    {!running && phase && <div className="z-fh">{phase}</div>}
    {err && <div className="z-fh" style={{ color: "#b91c1c" }}>{err}</div>}
  </Field>;
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
    {modal?.mode === "edit" && <DeviceModal title="Edit speaker" initial={devices[modal.index]} onSave={(d) => onEdit(modal.index, d)} onClose={() => setModal(null)}
      device={devices[modal.index]?.name} online={statusOf(devices[modal.index]?.name) === "online"} />}
  </div>;
}
