import React, { useEffect, useState, useCallback } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Card, Field, KeyInput, PageHeader, FormPage, StatusPill, Pill, Modal, Loading, Select,
} from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { errorLines } from "../useStageForm.js";
import { usePageForm } from "../usePageForm.js";
import { deref } from "../schema.js";
import { BUILTIN_SOURCES } from "../toolSources.js";
import { getTools } from "../api.js";

// ── Tool sources (multi-source view of the ToolHub) ───────────────────────
// Integration cards driven by CONFIG (so each shows even before it is
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

// One built-in source card, folded to its header until opened: header (name + kind
// badge + status pill), an editable SchemaForm bound to its core.* sub-section, and
// the live tool chips. The card's draft belongs to the page form, saved with the
// page's one Save.
//   name      — human title; sub — short caption under it
//   schema    — resolved JSON sub-schema (from core.schema, $defs available on root)
//   root      — full core schema (holds $defs for the SchemaForm)
//   draft, onChange — the card's draft core.<section> values and its setter
//   configured(values) -> bool ("configured")
//   live      — matching /api/tools entry, or null when absent
function SourceCard({ name, sub, schema, root, draft, onChange, configured, live }) {
  const isConfigured = configured(draft);
  const kind = live?.kind || "builtin";

  // Status: online/offline come from the live source; a source switched off via
  // its `enabled` flag shows "disabled"; otherwise "not configured" when the
  // relevant config is empty (a configured source absent from /api/tools failed
  // to start — shown as offline rather than "not configured").
  let pill;
  if (live) pill = <StatusPill status={live.online ? "online" : "offline"} />;
  else if (draft.enabled === false) pill = <Pill tone="muted">disabled</Pill>;
  else if (!isConfigured) pill = <Pill tone="muted">not configured</Pill>;
  else pill = <StatusPill status="offline" />;

  return <details className="z-card" style={{ marginBottom: 14 }}>
    <summary className="z-card-h">
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <svg className="z-chev" width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M2 4l3.5 3.5L9 4" /></svg>
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
    </summary>
    <div className="z-card-b">
      {schema
        ? <SchemaForm schema={schema} root={root} values={draft} onChange={onChange} />
        : <div className="z-fh">Schema unavailable.</div>}
      <ToolChips tools={live?.tools} />
    </div>
  </details>;
}

// ── External MCP server modal (add / edit) ────────────────────────────────
// Fields mirror McpServerConfig: name (unique source id), url, token (masked),
// transport (Literal), prompt (describes the server's tools to the model). Its
// button puts the server into the page's draft; the page's Save writes it.
const TRANSPORTS = [
  { value: "auto", label: "Auto" }, { value: "streamable_http", label: "Streamable HTTP" }, { value: "sse", label: "SSE" },
];
// Built-in ToolHub source ids — an external server may not shadow them, or it
// would hide the openweathermap/calendar/reminders/assistant_tasks/volume status in /api/tools.
const RESERVED_NAMES = ["openweathermap", "calendar", "reminders", "assistant_tasks", "volume"];
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
  // Edit writes back only the fields the user changed: the rest of the entry (enabled,
  // slow, and fields a saved entry may omit) stays as it was, so Done without changes
  // leaves the server equal to its saved copy. Add writes every field, enabled and not slow.
  const done = () => {
    const fields = { name, url, token, transport, prompt };
    if (!initial) return onSave({ ...fields, enabled: true, slow: false });
    const start = { name: initial.name || "", url: initial.url || "", token: initial.token || "", transport: initial.transport || "auto", prompt: initial.prompt || "" };
    const next = { ...initial };
    for (const k of Object.keys(fields)) if (fields[k] !== start[k]) next[k] = fields[k];
    onSave(next);
  };
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!valid} onClick={done}>{initial ? "Done" : "Add"}</button></>}>
    {/* The name is also the source id in /api/tools. */}
    <Field label="Name" hint="Unique name of this server among the tool sources.">
      <div className="z-inp"><input value={name} placeholder="e.g. home" onChange={(e) => setName(e.target.value)} /></div>
      {dup && <div className="z-fh" style={{ color: "#b91c1c" }}>Name is already in use.</div>}
      {!dup && reserved && <div className="z-fh" style={{ color: "#b91c1c" }}>Name is reserved by a built-in source.</div>}
    </Field>
    <Field label="URL"><div className="z-inp mono"><input value={url} placeholder="http://10.0.0.5:8123/mcp_server/sse" onChange={(e) => setUrl(e.target.value)} /></div></Field>
    <Field label="Token" hint="Optional Bearer token.">
      <KeyInput value={token} placeholder="optional…" onChange={setToken} />
    </Field>
    <Field label="Transport" hint="Auto picks SSE for a URL ending in /sse, otherwise Streamable HTTP.">
      <Select value={transport} options={TRANSPORTS} onChange={setTransport} />
    </Field>
    <Field label="Prompt" hint="Describes this server's tools to the model (appended to the system prompt).">
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false}
        style={{ width: "100%", minHeight: 90, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "10px 12px", fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.6, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
    </Field>
  </Modal>;
}

// One external MCP server card: header (name + status pill), the url in mono, the
// Enabled switch (`enabledSchema`, rendered like the built-in sources' switch) and the
// live tool chips matched from /api/tools by source id === server name. The switch and
// the Edit / Delete buttons change the page's draft; a disabled server is dimmed except
// its controls and shows a "disabled" pill (missing flag = enabled).
function McpServerCard({ server, saved, live, enabledSchema, root, onChange, onEdit, onDelete }) {
  const enabled = server.enabled ?? true;
  const dim = enabled ? "" : "z-dim";
  let pill;
  if (!enabled) pill = <Pill tone="muted">disabled</Pill>;
  // A server added or renamed in the draft has never been connected to: no status yet.
  else if (!saved) pill = <Pill tone="muted">not saved</Pill>;
  else if (live) pill = <StatusPill status={live.online ? "online" : "offline"} />;
  else pill = <StatusPill status="offline" />;
  // On a narrow card the header wraps: the status and buttons drop under the name and url, a
  // long url breaks anywhere. The name block's 240px basis keeps a long url from wrapping them
  // on a wide card.
  return <div className="z-card" style={{ marginBottom: 14 }}>
    <div className="z-card-h" style={{ flexWrap: "wrap" }}>
      <div className={dim} style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, flex: "1 1 240px" }}>
        <span style={{ display: "flex", flex: "none" }}><Ic n="mcp" w={17} /></span>
        <div style={{ minWidth: 0 }}>
          <b style={{ display: "block" }}>{server.name}</b>
          <span className="mono" style={{ fontSize: 11, color: "var(--mut)", overflowWrap: "anywhere" }}>{server.url}</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 12, marginLeft: "auto" }}>
        <div className={dim} style={{ display: "flex" }}>{pill}</div>
        <div style={{ display: "flex", gap: 6 }}>
          <button className="z-btn g sm" onClick={onEdit}>Edit</button>
          <button className="z-btn d sm" onClick={onDelete}>Delete</button>
        </div>
      </div>
    </div>
    <div className="z-card-b">
      <SchemaForm schema={enabledSchema} root={root} values={{ enabled }} onChange={onChange} />
      <div className={dim}><ToolChips tools={live?.tools} /></div>
    </div>
  </div>;
}

// ── Tools: external MCP servers + built-in tool sources ──────────────────
export function MCP() {
  const { catalog, patch } = useAppData();
  const coreSchema = catalog.core.schema;
  const coreValues = catalog.core.values;
  // properties.<section> is a $ref into $defs; resolve it so SchemaForm gets a
  // schema with `.properties` (and pass the full core schema as `root` for $defs).
  const sub = (key) => deref(coreSchema.properties?.[key] || {}, coreSchema);
  // An external server's Enabled switch: the `enabled` field of its schema alone.
  const enabledSchema = { properties: { enabled: deref(sub("mcp_servers").items, coreSchema).properties?.enabled } };

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

  // One page form holds every source: the external servers' list (core.mcp_servers,
  // replaced whole) and the built-in sections. Adding, editing, deleting and switching
  // a server only changes the draft; the page's one Save writes it all in one patch,
  // and the sources are rebuilt live (rebuild_tools), no restart needed.
  const form = usePageForm([
    { key: "mcp_servers", values: { mcp_servers: coreValues.mcp_servers || [] }, toPatch: (d) => ({ core: d }) },
    ...BUILTIN_SOURCES.map((s) => ({
      key: s.id, values: coreValues[s.id] || s.fallback, toPatch: (d) => ({ core: { [s.id]: d } }),
    })),
  ], patchAndRefresh);
  const servers = form.draftOf("mcp_servers").mcp_servers;   // [{enabled,name,url,token,transport,prompt,slow}]
  const setServers = (list) => form.onChangeOf("mcp_servers")("mcp_servers", list);

  const [modal, setModal] = useState(null); // { mode:'add'|'edit', index }
  const onAdd = (s) => { setServers([...servers, s]); setModal(null); };
  const onEdit = (i, s) => { setServers(servers.map((x, idx) => (idx === i ? s : x))); setModal(null); };
  const onDelete = (i) => {
    if (!window.confirm(`Delete MCP server "${servers[i].name}"?`)) return;
    setServers(servers.filter((_, idx) => idx !== i));
  };
  // A card's Enabled switch edits its server in the draft; once saved, a disabled
  // server is dropped from the ToolHub and its prompt block.
  const onServerChange = (i) => (field, value) => setServers(servers.map((x, idx) => (idx === i ? { ...x, [field]: value } : x)));

  return <FormPage form={form} narrow>
    <PageHeader title="Tools" desc="Tool sources the model calls: external smart-home MCP servers and built-in weather/calendar/reminders/assistant tasks, plus the always-on device volume tool. Sources are applied live — rebuilt on save, no restart needed."
      actions={<button className="z-btn p" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add server</button>} />
    {toolsErr && <div className="z-banner warn" style={{ margin: "0 0 14px" }}>
      <Ic n="restart" w={15} />
      <span><b>Status unavailable.</b> Failed to fetch the tool list: {errorLines(toolsErr).join(" · ")}</span>
    </div>}
    {tools === null
      ? <Card><Loading /></Card>
      : <>
        <div className="z-sl">External MCP servers<div className="ln" /></div>
        {servers.length === 0
          ? <Card><div className="z-fh" style={{ padding: "6px 0" }}>No external MCP servers — smart home is unavailable.</div></Card>
          : servers.map((s, i) => <McpServerCard key={s.name || i} server={s} live={liveOf(s.name)}
              saved={(coreValues.mcp_servers || []).some((x) => x.name === s.name)}
              enabledSchema={enabledSchema} root={coreSchema} onChange={onServerChange(i)}
              onEdit={() => setModal({ mode: "edit", index: i })} onDelete={() => onDelete(i)} />)}
        <div className="z-sl">Built-in sources<div className="ln" /></div>
        {BUILTIN_SOURCES.map((s) => <SourceCard key={s.id} name={s.name} sub={s.sub} schema={sub(s.id)} root={coreSchema}
          draft={form.draftOf(s.id)} onChange={form.onChangeOf(s.id)} configured={s.configured} live={liveOf(s.id)} />)}
      </>}
    {modal?.mode === "add" && <McpServerModal title="Add MCP server" onSave={onAdd} onClose={() => setModal(null)}
      takenNames={servers.map((s) => s.name)} />}
    {modal?.mode === "edit" && <McpServerModal title="Edit MCP server" initial={servers[modal.index]}
      onSave={(s) => onEdit(modal.index, s)} onClose={() => setModal(null)}
      takenNames={servers.filter((_, idx) => idx !== modal.index).map((s) => s.name)} />}
  </FormPage>;
}
