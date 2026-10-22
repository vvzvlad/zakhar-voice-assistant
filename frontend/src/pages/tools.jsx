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
import { deviceLabel } from "../format.js";
import { getTools, getMemory, putMemory, getReminders, patchReminder, deleteReminder } from "../api.js";

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
//   children  — the source's own editor, shown above the tool chips
function SourceCard({ name, sub, schema, root, draft, onChange, configured, live, children }) {
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
      {children}
      <ToolChips tools={live?.tools} />
    </div>
  </details>;
}

// A mono textarea, as the MCP server's Prompt and the memory card take it.
const TEXTAREA_STYLE = { width: "100%", minHeight: 90, resize: "vertical", border: "1px solid var(--line)", borderRadius: 8, padding: "10px 12px", fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.6, color: "var(--ink)", outline: "none", background: "var(--panel2)" };

// The memory card's draft: the saved facts as one text, one fact per line, loaded on
// mount. The assistant may remember or forget facts while the page is open, so save()
// applies the owner's edit (lines removed from and added to the loaded text) to the
// facts as they are now, then shows the list as the server stored it; it keeps its own
// error, so it never rejects.
function useMemoryDraft() {
  const [text, setText] = useState(null);       // null until loaded
  const [loaded, setLoaded] = useState("");     // the text as last loaded or saved
  const [loadErr, setLoadErr] = useState(null);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const show = (r) => { const t = r.facts.join("\n"); setText(t); setLoaded(t); };
  useEffect(() => { getMemory().then(show).catch(setLoadErr); }, []);
  const save = async () => {
    setSaving(true); setErr(null);
    try {
      const before = loaded.split("\n"), after = text.split("\n");
      const { facts } = await getMemory();
      show(await putMemory([...facts.filter((f) => after.includes(f) || !before.includes(f)),
        ...after.filter((f) => !before.includes(f))]));
    }
    catch (e) { setErr(e); }
    finally { setSaving(false); }
  };
  return { text, setText, loadErr, err, saving, save, dirty: text != null && text !== loaded };
}

// The memory source's editor of the saved facts (data/memory.json, not the config),
// one per line, inside its card. The draft belongs to the page, saved with the page's
// one Save.
function MemoryFacts({ text, onChange, loadErr }) {
  return <div className="z-f">
    <div className="z-fh" style={{ marginBottom: 8 }}>
      Facts the assistant saved with its remember tool, one per line; they replace <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<MEMORY>>>>>"}</span> in every request. Edit or delete a line and Save. Off, the assistant neither sees nor changes them.
    </div>
    {text != null
      ? <textarea value={text} onChange={(e) => onChange(e.target.value)} spellCheck={false}
          aria-label="Memory facts" style={{ ...TEXTAREA_STYLE, minHeight: 140 }} />
      : loadErr
        ? <div className="z-fh" style={{ color: "#b91c1c" }}>{errorLines(loadErr).join(" · ")}</div>
        : <Loading />}
  </div>;
}

const KINDS = { reminder: "Reminder", timer: "Timer", alarm: "Alarm", command: "Command" };
// An alarm's repeat day buttons, ISO weekdays 1 (Mon) .. 7 (Sun).
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const pad2 = (n) => String(n).padStart(2, "0");
// Epoch seconds <-> the local "YYYY-MM-DDTHH:MM:SS" of a datetime-local input, in whole
// seconds; a value the input left empty or half-typed reads as NaN.
const toLocalInput = (ts) => {
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
};
const fromLocalInput = (s) => new Date(s).getTime() / 1000;
// Seconds as "1h 02m 05s" / "4m 30s" / "12s".
const fmtLeft = (sec) => {
  const s = Math.max(0, Math.round(sec)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${pad2(m)}m ${pad2(s % 60)}s` : m ? `${m}m ${pad2(s % 60)}s` : `${s}s`;
};

// The PATCH body of the owner's edit of a scheduled row: only the fields that differ
// from the row as loaded ({} = nothing changed). `edit` holds what the inputs show:
// text, due (the datetime-local value) and days (sorted; [] = one-shot).
function rowChanges(row, edit = {}) {
  const out = {};
  if (edit.text !== undefined && edit.text !== (row.text ?? "")) out.text = edit.text;
  if (edit.due !== undefined && fromLocalInput(edit.due) !== Math.floor(row.due_ts)) out.due_ts = fromLocalInput(edit.due);
  if (edit.days !== undefined && edit.days.join() !== (row.days || []).join()) out.days = edit.days.length ? edit.days : null;
  return out;
}

// The reminders card's draft: the scheduler's pending rows, loaded on mount with the
// time they were loaded at, and the owner's edits to them by row id ({text, due, days,
// deleted}, each once touched). save() sends a DELETE or a PATCH per changed row, drops
// the edits that went through, then loads the rows again; it keeps its own error (the
// first failure), so it never rejects.
function useRemindersDraft() {
  const [list, setList] = useState(null);       // {items, at}; null until loaded
  const [loadErr, setLoadErr] = useState(null);
  const [edits, setEdits] = useState({});
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const load = () => getReminders()
    .then((r) => { setList({ items: r.items, at: Date.now() / 1000 }); setLoadErr(null); })
    .catch(setLoadErr);
  useEffect(() => { load(); }, []);
  const edit = (id, change) => setEdits((e) => ({ ...e, [id]: { ...e[id], ...change } }));
  const changed = (list?.items || []).filter((r) => edits[r.id]?.deleted || Object.keys(rowChanges(r, edits[r.id])).length);
  const save = async () => {
    setSaving(true); setErr(null);
    const left = { ...edits };
    let first = null;
    for (const r of changed) {
      try {
        if (edits[r.id].deleted) await deleteReminder(r.id);
        else await patchReminder(r.id, rowChanges(r, edits[r.id]));
        delete left[r.id];
      } catch (e) { first ??= e; }
    }
    setEdits(left); setErr(first);
    await load();
    setSaving(false);
  };
  return { items: list?.items ?? null, at: list?.at, loadErr, edits, edit, err, saving, save, load, dirty: changed.length > 0 };
}

// One scheduled row: its kind, speaker, a timer's time left as of loading and Delete
// (Undo while marked), then its text and due time, and an alarm's repeat days.
function ScheduledRow({ row, edit, onEdit, device, left }) {
  const days = edit.days ?? row.days ?? [];
  const toggleDay = (d) => onEdit({ days: days.includes(d) ? days.filter((x) => x !== d) : [...days, d].sort((a, b) => a - b) });
  const dim = edit.deleted ? "z-dim" : "";
  return <div data-testid="scheduled-row" style={{ borderTop: "1px solid var(--line2)", padding: "10px 0" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div className={dim} style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, minWidth: 0, flex: 1 }}>
        <Pill tone="muted">{KINDS[row.kind] || row.kind}</Pill>
        <span style={{ fontSize: 12, color: "var(--mut)" }}>{device ?? "—"}</span>
        {row.kind === "timer" && <span className="mono" style={{ fontSize: 11.5, color: "var(--mut)" }}>{fmtLeft(left)} left at load</span>}
      </div>
      <button className={"z-btn sm " + (edit.deleted ? "g" : "d")} onClick={() => onEdit({ deleted: !edit.deleted })}>
        {edit.deleted ? "Undo" : "Delete"}
      </button>
    </div>
    <div className={dim} style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
      <div className="z-inp sm" style={{ flex: "1 1 220px" }}>
        <input aria-label="Text" value={edit.text ?? row.text ?? ""} onChange={(e) => onEdit({ text: e.target.value })} />
      </div>
      <div className="z-inp sm mono">
        <input type="datetime-local" step="1" aria-label="Due time" value={edit.due ?? toLocalInput(row.due_ts)}
          onChange={(e) => onEdit({ due: e.target.value })} />
      </div>
    </div>
    {row.kind === "alarm" && <div className={dim} style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 10, marginTop: 8 }}>
      <div className="z-seg wrap">
        {WEEKDAYS.map((name, i) => <button key={name} type="button" className={days.includes(i + 1) ? "on" : ""}
          aria-pressed={days.includes(i + 1)} onClick={() => toggleDay(i + 1)}>{name}</button>)}
      </div>
      <span className="z-fh">{days.length ? "Repeats on these days" : "Rings once"}</span>
    </div>}
  </div>;
}

// The reminders source's list of what its scheduler holds (data/schedule.db, not the
// config), inside its card. The edits belong to the page, saved with the page's one Save.
function ScheduledItems({ draft, devices }) {
  const { items, at, loadErr, edits, edit } = draft;
  return <div className="z-f">
    <div className="z-fh" style={{ marginBottom: 8 }}>
      Reminders, timers, alarms and deferred commands waiting to fire, soonest first. Edit or delete one and Save.
    </div>
    {items == null
      ? loadErr
        ? <div className="z-fh" style={{ color: "#b91c1c" }}>{errorLines(loadErr).join(" · ")}</div>
        : <Loading />
      : items.length === 0
        ? <div className="z-fh">Nothing is scheduled.</div>
        : items.map((r) => <ScheduledRow key={r.id} row={r} edit={edits[r.id] || {}} onEdit={(c) => edit(r.id, c)}
            device={deviceLabel(devices, r.device)} left={r.due_ts - at} />)}
  </div>;
}

// ── External MCP server modal (add / edit) ────────────────────────────────
// Fields mirror McpServerConfig: name (unique source id), url, token (masked),
// transport (Literal), prompt (describes the server's tools to the model). Its
// button puts the server into the page's draft; the page's Save writes it.
const TRANSPORTS = [
  { value: "auto", label: "Auto" }, { value: "streamable_http", label: "Streamable HTTP" }, { value: "sse", label: "SSE" },
];
// Built-in ToolHub source ids — an external server may not shadow them, or it
// would hide the openweathermap/calendar/reminders/assistant_tasks/volume/memory status in /api/tools.
const RESERVED_NAMES = ["openweathermap", "calendar", "reminders", "assistant_tasks", "volume", "memory"];
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
        style={TEXTAREA_STYLE} />
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
  // The saved facts and the scheduled rows live outside the config: the page's one Save
  // sends the config patch, the facts and the rows' edits, each only when dirty; the save
  // bar shows the errors of all three.
  const memory = useMemoryDraft();
  const reminders = useRemindersDraft();
  const page = {
    dirty: form.dirty || memory.dirty || reminders.dirty,
    saving: form.saving || memory.saving || reminders.saving,
    // A config save may switch the reminders source on or off: the rows are read again.
    save: () => Promise.all([form.dirty && form.save().then(reminders.load), memory.dirty && memory.save(),
      reminders.dirty && reminders.save()]),
  };

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

  return <FormPage form={page} narrow errors={[...(form.dirty ? errorLines(form.err) : []), ...(memory.dirty ? errorLines(memory.err) : []), ...(reminders.dirty ? errorLines(reminders.err) : [])]}>
    <PageHeader title="Tools" desc="Tool sources the model calls: external smart-home MCP servers and built-in weather/calendar/reminders/assistant tasks/memory, plus the always-on device volume tool. Sources are applied live — rebuilt on save, no restart needed."
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
          draft={form.draftOf(s.id)} onChange={form.onChangeOf(s.id)} configured={s.configured} live={liveOf(s.id)}>
          {s.id === "memory" && <MemoryFacts text={memory.text} onChange={memory.setText} loadErr={memory.loadErr} />}
          {s.id === "reminders" && <ScheduledItems draft={reminders} devices={coreValues.devices} />}
        </SourceCard>)}
      </>}
    {modal?.mode === "add" && <McpServerModal title="Add MCP server" onSave={onAdd} onClose={() => setModal(null)}
      takenNames={servers.map((s) => s.name)} />}
    {modal?.mode === "edit" && <McpServerModal title="Edit MCP server" initial={servers[modal.index]}
      onSave={(s) => onEdit(modal.index, s)} onClose={() => setModal(null)}
      takenNames={servers.filter((_, idx) => idx !== modal.index).map((s) => s.name)} />}
  </FormPage>;
}
