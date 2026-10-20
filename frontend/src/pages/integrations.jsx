import React, { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Field, KeyInput, PageHeader, FormPage, StatusPill, Pill, Modal, Stepper, Loading, Select, Slider, Toggle,
  Seg, ScaleSeg,
} from "../components/primitives.jsx";
import { parseVoiceMarker, voiceField, voiceName, differsFromVoice } from "../voiceMarker.js";
import SchemaForm from "../components/SchemaForm.jsx";
import DeviceLog from "../components/DeviceLog.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { usePageForm } from "../usePageForm.js";
import { confirmDiscard } from "../navStore.js";
import { deref } from "../schema.js";
import {
  getPromptProfiles, getPromptProfile, createPromptProfile, updatePromptProfile,
  deletePromptProfile, activatePromptProfile,
  getTools, getOptions, startCapture, getCaptureStatus, downloadCaptureResult, getDeviceControls, setDeviceControl,
  sayOnDevice, getWakeProbability, setWakeProbabilityStream, getConfig, getWakeModels, uploadWakeModel,
} from "../api.js";
import { deviceLabel, deviceVersion, fmtUptime } from "../format.js";

function Card({ title, sub, children, foot, right }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}{right}</div>}
    {children !== undefined && <div className="z-card-b">{children}</div>}
    {foot}
  </div>;
}

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

// One integration source card: header (name + kind badge + status pill), an
// editable SchemaForm bound to its core.* sub-section, and the live tool chips.
// The card's draft belongs to the page form, saved with the page's one Save.
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
  </div>;
}

// The built-in tool sources; `id` is both the core.* section and the source id
// in /api/tools, `fallback` the values shown before the section exists.
const BUILTIN_SOURCES = [
  { id: "openweathermap", name: "OpenWeatherMap (built-in)", sub: "core.openweathermap · built-in MCP",
    fallback: { enabled: true, api_key: "", city: "Moscow" }, configured: (v) => !!(v && v.api_key) },
  { id: "calendar", name: "Calendar (built-in)", sub: "core.calendar · built-in MCP (CalDAV)",
    fallback: { enabled: true, url: "", username: "", password: "", calendar: "" }, configured: (v) => !!(v && v.url && v.username) },
  { id: "reminders", name: "Reminders (built-in)", sub: "core.reminders · built-in MCP",
    fallback: { enabled: true }, configured: (v) => !!(v && v.enabled) },
  { id: "assistant_tasks", name: "Assistant tasks (built-in)", sub: "core.assistant_tasks · built-in MCP (ZeroClaw webhook)",
    fallback: { enabled: true, url: "", secret: "", task_suffix: "", prompt: "" }, configured: (v) => !!(v && v.url && v.secret) },
];

// ── External MCP server modal (add / edit) ────────────────────────────────
// Fields mirror McpServerConfig: name (unique source id), url, token (masked),
// transport (Literal), prompt (describes the server's tools to the model).
const TRANSPORTS = ["auto", "streamable_http", "sse"];
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
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      {/* Preserve the enabled/slow flags across edits (they have no modal fields);
          Add mode defaults to enabled + not slow. */}
      <button className="z-btn p" disabled={!valid} onClick={() => onSave({ name, url, token, transport, prompt, enabled: initial?.enabled ?? true, slow: initial?.slow ?? false })}>Save</button></>}>
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

// One external MCP server card: header (name + status pill), the
// url in mono, and the live tool chips matched from /api/tools by source id ===
// server name. Enable/Disable / Edit / Delete buttons drive the CRUD flow above;
// a disabled server is dimmed except its buttons and shows a "disabled" pill (missing
// flag = enabled).
function McpServerCard({ server, live, onToggle, onEdit, onDelete }) {
  const enabled = server.enabled ?? true;
  const dim = enabled ? "" : "z-dim";
  let pill;
  if (!enabled) pill = <Pill tone="muted">disabled</Pill>;
  else if (live) pill = <StatusPill status={live.online ? "online" : "offline"} />;
  else pill = <StatusPill status="offline" />;
  return <div className="z-card" style={{ marginBottom: 14 }}>
    <div className="z-card-h">
      <div className={dim} style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <Ic n="mcp" w={17} />
        <div style={{ minWidth: 0 }}>
          <b style={{ display: "block" }}>{server.name}</b>
          <span className="mono" style={{ fontSize: 11, color: "var(--mut)" }}>{server.url}</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginLeft: "auto" }}>
        <div className={dim} style={{ display: "flex" }}>{pill}</div>
        <div style={{ display: "flex", gap: 6 }}>
          <button className="z-mini" onClick={onToggle}>{enabled ? "Disable" : "Enable"}</button>
          <button className="z-mini" onClick={onEdit}>Edit</button>
          <button className="z-mini" onClick={onDelete}>Delete</button>
        </div>
      </div>
    </div>
    <div className={`z-card-b ${dim}`}>
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

  const servers = coreValues.mcp_servers || [];   // [{enabled,name,url,token,transport,prompt}]

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
  const onDelete = (i) => {
    if (!window.confirm(`Delete MCP server "${servers[i].name}"?`)) return;
    saveList(servers.filter((_, idx) => idx !== i));
  };
  // Flip the persisted enabled flag (missing = true on old configs). A disabled
  // server is dropped from the ToolHub and its prompt block — both hot-applied.
  const onToggle = (i) => saveList(servers.map((x, idx) => (idx === i ? { ...x, enabled: !(x.enabled ?? true) } : x)));

  // The built-in source cards are one page form; sources are hot-reloaded —
  // rebuilt live on save (rebuild_tools).
  const form = usePageForm(BUILTIN_SOURCES.map((s) => ({
    key: s.id, values: coreValues[s.id] || s.fallback, toPatch: (d) => ({ core: { [s.id]: d } }),
  })), patchAndRefresh);

  return <FormPage form={form} narrow>
    <PageHeader title="Tools" desc="Tool sources the model calls: external smart-home MCP servers and built-in weather/calendar/reminders/assistant tasks/speaker volume. Sources are applied live — rebuilt on save, no restart needed."
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
              onToggle={() => onToggle(i)}
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

// ── System prompt ─────────────────────────────────────────────────────────
// Small name-entry modal shared by the New / Rename profile actions.
function ProfileNameModal({ title, initial, onSave, onClose, busyErr }) {
  const [name, setName] = useState(initial || "");
  const valid = !!name.trim();
  return <Modal title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!valid} onClick={() => onSave(name.trim())}>Save</button></>}>
    <Field label="Name" hint="Unique profile name.">
      <div className="z-inp"><input value={name} placeholder="e.g. concise" autoFocus
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && valid) onSave(name.trim()); }} /></div>
      {busyErr && <div className="z-fh" style={{ color: "#b91c1c" }}>{errorLines(busyErr).join(" · ")}</div>}
    </Field>
  </Modal>;
}

export function Prompt() {
  // Named prompt profiles: the list (summaries + active_id) drives the selector;
  // `selected` is the full profile whose text sits in the editor. The ACTIVE
  // profile is what the pipeline uses — selecting/editing never switches it,
  // only the explicit Activate button does.
  const [profiles, setProfiles] = useState(null);  // null = loading
  const [activeId, setActiveId] = useState(null);
  const [selected, setSelected] = useState(null);  // {id,name,...} of the editor's profile
  const [text, setText] = useState(null);
  const [loaded, setLoaded] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);            // load and profile-action errors
  const [saveErr, setSaveErr] = useState(null);    // the text save's error, shown in the save bar
  const [modal, setModal] = useState(null);        // {mode:'new'|'rename'} | null
  const [modalErr, setModalErr] = useState(null);
  const [profileBusy, setProfileBusy] = useState(false);
  // Default ON when a marker is present: activating that profile then also applies
  // its preferred voice (convenient). Hidden entirely when no marker exists.
  const [applyVoice, setApplyVoice] = useState(true);

  const refreshProfiles = useCallback(async () => {
    const r = await getPromptProfiles();
    setProfiles(r.profiles || []);
    setActiveId(r.active_id);
    return r;
  }, []);

  const openProfile = useCallback(async (id) => {
    const p = await getPromptProfile(id);
    setSelected(p); setText(p.text); setLoaded(p.text); setSaveErr(null);
  }, []);

  // Initial load + Reload: refresh the list, then open `preferId` when it still
  // exists, otherwise the active profile.
  const load = useCallback((preferId = null) => {
    refreshProfiles()
      .then((r) => {
        const keep = preferId != null && (r.profiles || []).some((p) => p.id === preferId);
        const id = keep ? preferId : r.active_id;
        return id != null ? openProfile(id) : null;
      })
      .then(() => setErr(null))
      .catch((e) => setErr(e));
  }, [refreshProfiles, openProfile]);
  useEffect(() => { load(); }, [load]);

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
  // Save persists the SELECTED profile's text (not necessarily the active one),
  // then refreshes the list so the char counts stay current.
  const save = async () => {
    if (!selected) return;
    setSaving(true); setSaveErr(null);
    try {
      await updatePromptProfile(selected.id, { text });
      setLoaded(text);
      await refreshProfiles();
    } catch (e) { setSaveErr(e); }
    finally { setSaving(false); }
  };

  // --- profile bar actions ---------------------------------------------------
  const selectProfile = async (id) => {
    if (!selected || id === selected.id) return;
    if (!confirmDiscard(dirty)) return;
    try { await openProfile(id); setErr(null); }
    catch (e) { setErr(e); }
  };

  const doActivate = async () => {
    if (!selected) return;
    setProfileBusy(true);
    try {
      // Activation applies the STORED text server-side, so the voice decision is
      // based on `loaded` (the last-saved text), not the possibly-dirty `text`.
      const voice = parseVoiceMarker(loaded);
      const resp = await activatePromptProfile(selected.id, { applyVoice: applyVoice && !!voice });
      await refreshProfiles();
      if (resp && resp.voice_error) {
        setErr(new Error(resp.voice_error));
      } else {
        setErr(null);
        // When a voice was applied, refresh the global catalog so Providers → TTS
        // and Speech show the new voice without a manual reload.
        if (resp && resp.voice_applied) await reload();
      }
    } catch (e) { setErr(e); }
    finally { setProfileBusy(false); }
  };

  // New: text omitted -> the server copies the ACTIVE profile's text; the new
  // profile is then selected into the editor.
  const doCreate = async (name) => {
    setModalErr(null);
    try {
      const p = await createPromptProfile({ name });
      setModal(null);
      setSelected(p); setText(p.text); setLoaded(p.text); setSaveErr(null);
      await refreshProfiles();
      setErr(null);
    } catch (e) { setModalErr(e); }
  };

  const doRename = async (name) => {
    setModalErr(null);
    try {
      const p = await updatePromptProfile(selected.id, { name });
      setModal(null);
      setSelected((prev) => ({ ...prev, name: p.name }));
      await refreshProfiles();
      setErr(null);
    } catch (e) { setModalErr(e); }
  };

  const doDelete = async () => {
    if (!selected || selected.id === activeId) return; // the active profile is undeletable
    if (!window.confirm(`Delete profile "${selected.name}"?`)) return;
    setProfileBusy(true);
    try {
      await deletePromptProfile(selected.id);
      load(); // falls back to the active profile
    } catch (e) { setErr(e); }
    finally { setProfileBusy(false); }
  };

  // Dialog context (core.context) — its own draft in the aside, saved by the
  // page's one Save together with the text.
  // All hooks must run before the early-return below to satisfy the rules of hooks.
  const { catalog, patch, reload } = useAppData();
  const ctxSchema = catalog.core.schema.$defs?.ContextConfig;
  const ctxValues = catalog.core.values.context || {};
  const ctxBuildPatch = (d) => ({ core: { context: d } });
  const { draft: ctxDraft, onChange: ctxOnChange, dirty: ctxDirty, saving: ctxSaving, err: ctxErr, save: ctxSave } = useStageForm(ctxValues, ctxBuildPatch, patch);

  // Preferred voice parsed from the SELECTED profile's last-saved text (`loaded`),
  // the TTS provider it names and the one speaking now (first enabled variant).
  const voice = parseVoiceMarker(loaded);
  const ttsCat = catalog.categories?.find((c) => c.id === "tts");
  const ttsProv = (id) => ttsCat?.providers.find((p) => p.id === id);
  const markerProv = voice && ttsProv(voice.provider);
  const current = ttsCat?.variants?.find((v) => v.enabled)?.provider;
  // An opaque voice id (one whose name the TTS form keeps in "<field>_label") that
  // is not the provider's current voice is named by the options the form picks it from.
  const vField = voiceField(voice);
  const vId = vField ? voice.fields[vField] : "";
  const vValues = markerProv?.values || {};
  const lookup = !!(vField && markerProv?.schema?.properties?.[`${vField}_label`])
    && !(vValues[vField] === vId && vValues[`${vField}_label`]);
  const vProvider = voice?.provider;
  const [found, setFound] = useState(null);   // {id, name} from the options endpoint
  useEffect(() => {
    if (!lookup) return undefined;
    let alive = true;
    getOptions("tts", vProvider, vField)
      .then((r) => {
        const hit = (r.options || []).find((o) => o && o.value === vId);
        if (alive && hit) setFound({ id: vId, name: hit.label });
      })
      .catch(() => { /* named by its id */ });
    return () => { alive = false; };
  }, [lookup, vProvider, vField, vId]);

  // The page's one Save persists both the prompt text and the dialog-context
  // settings. Each underlying save handles its own error/saving state, so this
  // never rejects; we just await whichever forms are dirty.
  const form = {
    dirty: dirty || ctxDirty,
    saving: saving || ctxSaving,
    save: async () => {
      await Promise.all([dirty ? save() : null, ctxDirty ? ctxSave() : null].filter(Boolean));
    },
  };

  if (text == null || profiles == null) {
    if (err) {
      // Surface a load failure (e.g. prompt store unavailable) instead of an
      // endless spinner.
      return <div className="z-page"><div className="z-card"><div className="z-card-b">
        <div className="z-fh" style={{ color: "#b91c1c", padding: "10px 0" }}>{errorLines(err).join(" · ")}</div>
      </div></div></div>;
    }
    return <div className="z-page"><div className="z-card"><Loading /></div></div>;
  }

  // Selector options: the active profile is visibly marked.
  const isActive = selected?.id === activeId;
  const profileOptions = profiles.map((p) => ({
    value: p.id,
    label: p.id === activeId ? `● ${p.name} (active)` : p.name,
  }));
  const voiceText = voice && [markerProv?.label || voice.provider,
    voiceName(voice, vValues, found?.id === vId ? found.name : undefined)].join(" · ");
  const voiceDiffers = differsFromVoice(voice, current, ttsProv(current)?.values);

  return <FormPage form={form} errors={[...errorLines(saveErr), ...errorLines(ctxErr)]}>
    <PageHeader title="System prompt"
      desc="Zahar's character, rules and answer format. Keep several named profiles and activate the one the assistant should use. The placeholder is replaced with live date/time at request time."
      actions={<button className="z-btn g" onClick={() => { if (confirmDiscard(dirty)) load(selected?.id); }}>Reload</button>} />
    {/* Profile bar: pick a profile to edit; Activate switches what the pipeline uses. */}
    <div className="z-card" style={{ marginBottom: 14 }}>
      <div className="z-card-b" style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", padding: "10px 14px" }}>
        <Select w={260} value={selected?.id} options={profileOptions} onChange={selectProfile} />
        <button className="z-btn p" disabled={isActive || profileBusy} onClick={doActivate}
          title={isActive ? "This profile is already active" : "Make this profile the one the assistant uses"}>Activate</button>
        {voice && <span style={{ display: "flex", alignItems: "center", gap: 6 }}
          title="Apply this profile's preferred TTS voice when activating">
          <Toggle on={applyVoice} onChange={setApplyVoice} sm />
          <span className="sub" data-testid="marker-voice">Apply voice: {voiceText}</span>
          {voiceDiffers && <Pill tone="warn">differs from current TTS voice</Pill>}
        </span>}
        <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
          <button className="z-btn g" onClick={() => { if (confirmDiscard(dirty)) { setModalErr(null); setModal({ mode: "new" }); } }}><Ic n="add" w={14} />New</button>
          <button className="z-btn g" onClick={() => { setModalErr(null); setModal({ mode: "rename" }); }}>Rename</button>
          <button className="z-btn d" disabled={isActive || profileBusy} onClick={doDelete}
            title={isActive ? "The active profile cannot be deleted" : ""}>Delete</button>
        </div>
      </div>
    </div>
    <div className="z-cols wide">
      <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{selected?.name} · {text.length} chars</span>} title="Editor"
        foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
        <div style={{ padding: "8px 0" }}>
          <textarea ref={taRef} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
            style={{ width: "100%", minHeight: 560, resize: "none", overflow: "hidden", boxSizing: "border-box", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
        </div>
      </Card>
      <div className="z-aside">
        <Card title="Dialog context">
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
        <Card title="Preferred voice">
          <div className="z-info" style={{ padding: "10px 0 12px" }}>
            A single line <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=… field=value>>>>>"}</span> records a preferred TTS voice for this profile.
            The first pair <b>provider=</b> picks the TTS provider; the rest map to its fields (e.g. <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=yandex voice=zahar>>>>>"}</span>).
            When you activate the profile with the toggle on, this voice is applied to the TTS config. The marker is <b>stripped</b> from the prompt sent to the model.
          </div>
        </Card>
        <Card title="Notation">
          <div className="z-info" style={{ padding: "10px 0 12px" }}>Stress marks use <b>«+»</b> before the stressed vowel (e.g. «з+амок»).</div>
        </Card>
      </div>
    </div>
    {modal?.mode === "new" && <ProfileNameModal title="New profile (copy of the active prompt)"
      onSave={doCreate} onClose={() => setModal(null)} busyErr={modalErr} />}
    {modal?.mode === "rename" && <ProfileNameModal title="Rename profile" initial={selected?.name}
      onSave={doRename} onClose={() => setModal(null)} busyErr={modalErr} />}
  </FormPage>;
}

// ── Devices ───────────────────────────────────────────────────────────────
const DEVICE_TABS = ["General", "Wake word", "Sound", "Stop word", "Log"];

// Add mode (`onSave({name, enabled})`) asks for a device id. Edit mode (`device`, the saved id)
// shows the device's status, its id — what the firmware sends in hello — and its versions above
// five tabs: General (the display label and any live control no other tab places), Wake word
// (the wake word model, the live wake probability, the cutoffs, sample capture), Sound (the
// speaker volume, a test phrase, the auto volume switch and, while it is on, its settings and
// readings, the reply compression switch, the playback reserve), Stop word (the echo
// cancellation) and Log (the device log switch and the log the server keeps). `onSave` gets only
// the changed ones of {label, wake_model, log, auto_volume, compress_replies} (wake_model "" =
// the release default; an uploaded .tflite becomes the selected model), whichever tabs they were
// edited on; a tab holding such a change carries a dot. The live controls are written at once,
// polled for the whole life of the dialog. All tabs stay mounted and only the open one shows, so
// a running capture or a typed phrase outlives a tab switch; the log alone mounts with its tab,
// so it is polled only while shown. `online` says whether the device is reachable, `logDays` how
// many days of device log the server keeps.
function DeviceModal({ onSave, onClose, title, device, label: savedLabel = "", wakeModel: savedWakeModel = "", log: savedLog = false, autoVolume: savedAutoVolume = false, compressReplies: savedCompressReplies = false, logDays, online }) {
  const [name, setName] = useState("");
  const [label, setLabel] = useState(savedLabel);
  const [wakeModel, setWakeModel] = useState(savedWakeModel);
  const [log, setLog] = useState(savedLog);
  const [autoVolume, setAutoVolume] = useState(savedAutoVolume);
  const [compressReplies, setCompressReplies] = useState(savedCompressReplies);
  const [wakeModels, setWakeModels] = useState(null); // {options, release} from GET /api/wake-models
  const [uploadErr, setUploadErr] = useState(null);
  const [tab, setTab] = useState("General");
  const ctl = useDeviceControls(device);
  const fileRef = useRef(null);
  const tabsRef = useRef(null);
  // A new tab opens at its first row. Once the sticky tabs bar has caught on, the scroll past
  // that point shows as a gap between the bar and the header it normally touches; drop just that.
  function pickTab(t) {
    const bar = tabsRef.current;
    if (bar) bar.parentElement.scrollTop -= bar.getBoundingClientRect().top - bar.previousElementSibling.getBoundingClientRect().bottom;
    setTab(t);
  }
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    getWakeModels()
      .then((r) => { if (alive) setWakeModels(r); })
      .catch(() => { /* the model field stays hidden */ });
    return () => { alive = false; };
  }, [device]);
  // Upload the picked .tflite, re-read the list and select the new model (Save sends it).
  async function upload(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // picking the same file again fires another change
    if (!file) return;
    setUploadErr(null);
    try {
      const { name: uploaded } = await uploadWakeModel(file);
      setWakeModels(await getWakeModels());
      setWakeModel(uploaded);
    } catch (err) {
      setUploadErr(err.message);
    }
  }
  if (device) {
    // The tabs holding a field Save would send.
    const marked = DEVICE_TABS.filter((t) => ({
      General: label.trim() !== savedLabel,
      "Wake word": wakeModel !== savedWakeModel,
      Sound: autoVolume !== savedAutoVolume || compressReplies !== savedCompressReplies,
      Log: log !== savedLog,
    })[t]);
    const snapControls = ctl.snap?.controls || [];
    const controlField = (c) => <ControlField key={c.id} control={c} value={ctl.valueOf(c)}
      onChange={(v) => ctl.write(c, v)} />;
    // The live controls of `ids`, in that order, each once the snapshot lists it.
    const controls = (...ids) => ids.map((id) => snapControls.find((c) => c.id === id)).filter(Boolean)
      .map(controlField);
    const loading = !ctl.snap && !ctl.err && <div className="z-f"><div className="z-fh">Loading…</div></div>;
    return <Modal top wide title={title} onClose={onClose}
      footer={<><button className="z-btn g" onClick={onClose}>Close</button>
        <button className="z-btn p" disabled={marked.length === 0} onClick={() => onSave({
          // Only the edited fields: the rest stays as in the server's fresh list (saveList).
          ...(label.trim() !== savedLabel && { label: label.trim() }),
          ...(wakeModel !== savedWakeModel && { wake_model: wakeModel }),
          ...(log !== savedLog && { log }),
          ...(autoVolume !== savedAutoVolume && { auto_volume: autoVolume }),
          ...(compressReplies !== savedCompressReplies && { compress_replies: compressReplies }),
        })}>Save</button></>}>
      <div className="z-dev-stats">
        <Stat label="Status"><StatusPill status={online ? "online" : "offline"} /></Stat>
        <Stat label="Device id" title="Sent by the firmware when it connects — keys the device's settings and dialog context.">{device}</Stat>
        <Stat label="Firmware" title={deviceVersion(ctl.snap, "config_version")}>{deviceVersion(ctl.snap, "config_version") ?? "—"}</Stat>
        <Stat label="Wake word model" title={deviceVersion(ctl.snap, "model_version")}>{deviceVersion(ctl.snap, "model_version") ?? "—"}</Stat>
      </div>
      <div className="z-tabs" ref={tabsRef}><Seg full options={DEVICE_TABS} value={tab} onChange={pickTab} marked={marked} /></div>
      {ctl.err && <div className="z-fh" style={{ color: "#b91c1c", marginTop: 8 }}>{ctl.err.message || String(ctl.err)}</div>}
      <div hidden={tab !== "General"}>
        <Field row label="Name" hint="Shown in the panel instead of the device id; empty shows the id."><div className="z-rowctl"><div className="z-inp"><input value={label} placeholder={device} onChange={(e) => setLabel(e.target.value)} /></div></div></Field>
        {snapControls.filter((c) => !PLACED_CONTROLS.has(c.id)).map(controlField)}
      </div>
      <div hidden={tab !== "Wake word"}>
        {wakeModels && <Field row label="Wake word model" hint="Sent to the device on save and at every connection. An uploaded .tflite is kept in the server's data volume; a new file of the same name replaces it and goes at once to every device using it.">
          <div className="z-rowctl">
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <Select value={wakeModel} onChange={setWakeModel} options={[
                  { value: "", label: `Release default (${wakeModels.release})` },
                  ...wakeModels.options.map((n) => ({ value: n, label: n })),
                ]} />
              </div>
              <button className="z-btn g" title="Upload a .tflite model" onClick={() => fileRef.current.click()}>Upload</button>
              <input ref={fileRef} type="file" accept=".tflite" hidden aria-label="Wake word model file" onChange={upload} />
            </div>
            {uploadErr && <div className="z-fh" style={{ color: "#b91c1c" }}>{uploadErr}</div>}
          </div>
        </Field>}
        <WakeProbabilityMeter device={device} online={online} />
        {loading}
        {controls("wake_probability_cutoff", "vad_cutoff")}
        <CaptureControl device={device} online={online} />
      </div>
      <div hidden={tab !== "Sound"}>
        {loading}
        {controls("speaker_volume")}
        <SayControl device={device} online={online} />
        <Field row label="Auto volume" hint="On: every 15 s of quiet the speaker measures the room's noise on its raw microphone; the reply's volume is the dial's plus (noise − threshold) × gain: louder above the threshold, quieter below it. Applies on save, no reflash.">
          <Toggle on={autoVolume} onChange={setAutoVolume} />
        </Field>
        {autoVolume && <>
          {controls("auto_volume_threshold", "auto_volume_gain")}
          {ctl.snap?.readings.length > 0 && <Field row label="Auto volume now" hint="The room's noise the speaker measured and the boost the server sends for it.">
            <div className="z-rowctl" style={{ display: "flex", gap: 24 }}>
              {ctl.snap.readings.map((r) => <Stat key={r.id} label={r.name}>
                {r.value == null ? "—" : withUnit(Number.isInteger(r.value) ? r.value : r.value.toFixed(1), r.unit)}
              </Stat>)}
            </div>
          </Field>}
        </>}
        <Field row label="Compress replies" hint="On: the server compresses everything it plays on this speaker — quiet parts up to 20 dB louder, peaks held at −11.5 dBFS. Louder for a TTS whose peaks sit below full scale (about +2–3 dB with Fish Audio); one that already peaks at full scale, as Piper does, plays up to ~3 dB quieter. Applies on save, no reflash.">
          <Toggle on={compressReplies} onChange={setCompressReplies} />
        </Field>
        {controls("play_prime_ms")}
      </div>
      <div hidden={tab !== "Stop word"}>
        {loading}
        {controls("echo_cancel", "echo_cancel_mode", "echo_cancel_filter")}
      </div>
      <div hidden={tab !== "Log"} className="z-logpane">
        <Field row label="Device log" hint={`On: the speaker streams its whole log to the server, which keeps the last ${logDays} days in data/device_logs/${device}/. Off: the speaker sends nothing. Applies on save, no reflash.`}>
          <Toggle on={log} onChange={setLog} />
        </Field>
        {tab === "Log" && <DeviceLog device={device} />}
      </div>
    </Modal>;
  }
  return <Modal top title={title} onClose={onClose}
    footer={<><button className="z-btn g" onClick={onClose}>Cancel</button>
      <button className="z-btn p" disabled={!name} onClick={() => onSave({ name, enabled: true })}>Add</button></>}>
    <Field label="Device id" hint="The id its firmware sends when it connects (its Wi-Fi MAC) — also keys the dialog context."><div className="z-inp"><input value={name} placeholder="e.g. fc012cd1e5ec" onChange={(e) => setName(e.target.value)} /></div></Field>
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
      if (s.state === "starting" || s.state === "recording") {
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

  return <Field row label="Capture sample" hint={`Records ${seconds}s of mic audio and downloads a WAV per take${count > 1 ? ` (×${count})` : ""}. Used for wake-word training.`}>
    <div className="z-rowctl">
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
              <Stepper sm value={seconds} min={1} max={300} onChange={setSeconds} unit="s" />
              <Stepper sm value={count} min={1} max={1000} onChange={setCount} unit="takes" />
              <button className="z-btn p" style={{ marginLeft: "auto" }} disabled={!online} title={online ? "" : "Speaker offline"}
                onClick={run}>Record</button>
            </>}
      </div>
      {!running && phase && <div className="z-fh">{phase}</div>}
      {err && <div className="z-fh" style={{ color: "#b91c1c" }}>{err}</div>}
    </div>
  </Field>;
}

// Short per-control hint shown under the control's label, keyed by control id.
const CONTROL_HINTS = {
  wake_probability_cutoff: "Higher = fewer false-fires, may miss wake word",
  vad_cutoff: "Lower = fewer dropped wakes, slightly more non-speech leak",
  echo_cancel: "Off: the stop word hears the wake word's input; 1..3: residual echo suppression, mild to strongest (FD modes only)",
  echo_cancel_mode: "SR / FD, low cost / high performance; a change re-creates the canceller",
  echo_cancel_filter: "Filter length 1..8; a change re-creates the canceller",
  play_prime_ms: "Audio the speaker holds before it starts a phrase, and again after the stream runs dry; more = fewer stutters, the answer starts later",
  auto_volume_threshold: "The room noise at which the reply plays at the dial's volume, with Auto volume on: louder above it, quieter below",
  auto_volume_gain: "Volume % per dB of noise away from the threshold; 2.5 %/dB moves the reply 1 dB per dB",
};

// The live controls DeviceModal places on its Wake word, Sound and Stop word tabs; any other
// control the server lists shows on General.
const PLACED_CONTROLS = new Set([
  "wake_probability_cutoff", "vad_cutoff",
  "speaker_volume", "auto_volume_threshold", "auto_volume_gain", "play_prime_ms",
  "echo_cancel", "echo_cancel_mode", "echo_cancel_filter",
]);

// Labels of the controls on the Stop word tab, whose server names repeat the tab's subject;
// every other control shows the server's name.
const CONTROL_LABELS = {
  echo_cancel: "Echo cancellation",
  echo_cancel_mode: "Mode",
  echo_cancel_filter: "Filter length",
};

// Controls shown as a segmented scale: the button labels, the stored value is the button's index.
const CONTROL_SCALES = {
  echo_cancel: ["Off", "1", "2", "3"],
  echo_cancel_mode: ["SR low", "SR high", "FD low", "FD high"],
};

// A value with its unit: "%" sticks to the number, any other unit follows a space.
const withUnit = (text, unit) => (!unit || unit === "%" ? `${text}${unit || ""}` : `${text} ${unit}`);

// A read-only figure: a small uppercase label over its mono value; `title` is the value's tooltip.
function Stat({ label, title, children }) {
  return <div className="z-stat">
    <div style={{ fontSize: 10.5, color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600 }}>{label}</div>
    <div className="mono" title={title} style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{children}</div>
  </div>;
}

// One live control: a ScaleSeg for the ids in CONTROL_SCALES, else a Slider over the range and
// step the server gives it.
function ControlField({ control: c, value, onChange }) {
  const scale = CONTROL_SCALES[c.id];
  return <Field row label={CONTROL_LABELS[c.id] || c.name} hint={CONTROL_HINTS[c.id]}>
    <div className="z-rowctl">
      {scale
        ? <ScaleSeg options={scale.map((label, i) => ({ value: i, label }))} value={value} onChange={onChange} />
        : <Slider min={c.min} max={c.max} step={c.step || 1} value={value} onChange={onChange}
            fmt={(val) => withUnit(c.step < 1 ? Number(val).toFixed(1) : Math.round(val), c.unit)} />}
    </div>
  </Field>;
}

// The per-device live state of the Edit-speaker dialog, used once by DeviceModal so it keeps
// polling while the tabs switch: `snap` is the last snapshot {controls, readings, versions}, null
// before the first. The controls are the wake-word probability cutoff, the VAD cutoff, the stop
// word's echo cancellation, the playback reserve, the auto volume's threshold and gain (all from
// the server config) and the speaker volume (read from the connected device, listed only while a
// device with a speaker is connected); the readings are the auto volume's (the room's noise, the
// boost), each null before its first value; the versions are the connected firmware's. `write`
// sends a control's new value debounced, `valueOf` is the value to show, `err` the last failed
// load or write. A background poll keeps the snapshot in sync with external changes (e.g. the
// physical volume dial) without stomping a control the user is actively dragging.
function useDeviceControls(device) {
  const { refreshConfig } = useAppData();
  const [snap, setSnap] = useState(null);
  const [err, setErr] = useState(null);
  const [local, setLocal] = useState({});        // id -> in-flight value (responsive drag)
  const pending = useRef({});                     // id -> bool (freeze poll-sync while writing)
  const timers = useRef({});                      // id -> debounce timeout handle

  // Apply a fresh snapshot, skipping controls the user is actively writing.
  const applySnapshot = (s) => {
    setSnap({ controls: s.controls || [], readings: s.readings || [], versions: s.versions || [] });
    setLocal((prev) => {
      const next = { ...prev };
      for (const c of s.controls || []) {
        if (!pending.current[c.id] && c.value != null) next[c.id] = c.value;
      }
      return next;
    });
  };

  // Initial load + background poll while the modal is open.
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    const load = () => getDeviceControls(device)
      .then((snap) => { if (alive) { applySnapshot(snap); setErr(null); } })
      .catch((e) => { if (alive) setErr(e); });
    load();
    const iv = setInterval(load, 2500);
    return () => { alive = false; clearInterval(iv); };
  }, [device]);

  // Clear any pending debounce timers on unmount (modal close).
  useEffect(() => () => { for (const t of Object.values(timers.current)) clearTimeout(t); }, []);

  const write = (c, v) => {
    setLocal((prev) => ({ ...prev, [c.id]: v }));
    pending.current[c.id] = true;
    if (timers.current[c.id]) clearTimeout(timers.current[c.id]);
    timers.current[c.id] = setTimeout(() => {
      // `device` is a stable prop for the modal's lifetime, so no stale-closure risk.
      // A cutoff, an echo cancellation setting, the playback reserve or an auto volume setting is
      // stored in the server config: refresh the config snapshot so a later save of the device
      // list carries the new value.
      setDeviceControl(device, c.id, v)
        .then((snap) => { applySnapshot(snap); setErr(null); return refreshConfig(); })
        .catch((e) => setErr(e))
        .finally(() => { pending.current[c.id] = false; });
    }, 250);
  };

  return { snap, err, write, valueOf: (c) => local[c.id] ?? c.value ?? c.min ?? 0 };
}

// "Say on speaker" control shown on the Sound tab of the Edit-speaker modal, under the speaker
// volume: speaks the typed text on the speaker through the TTS, so the volume can be checked by
// ear. The request resolves once the speech has played. Disabled (tooltip) while offline.
function SayControl({ device, online }) {
  const [text, setText] = useState("Проверка громкости.");
  const [saying, setSaying] = useState(false);
  const [err, setErr] = useState(null);

  const say = async () => {
    setErr(null);
    setSaying(true);
    try {
      await sayOnDevice(device, text);
    } catch (e) {
      setErr(e.message || "failed");
    } finally {
      setSaying(false);
    }
  };

  return <Field row label="Say on speaker" hint="Speaks the text on the speaker through the TTS — to check the volume.">
    <div className="z-rowctl">
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <div className="z-inp" style={{ flex: 1 }}><input value={text} onChange={(e) => setText(e.target.value)} /></div>
        <button className="z-btn p" disabled={!online || !text.trim() || saying} title={online ? "" : "Speaker offline"}
          onClick={say}>{saying ? "Saying…" : "Say"}</button>
      </div>
      {err && <div className="z-fh" style={{ color: "#b91c1c" }}>{err}</div>}
    </div>
  </Field>;
}

// Live, read-only Wake Probability meter shown on the Wake word tab of the Edit-speaker modal
// (only for an EXISTING device). The stream is gated ON THE DEVICE: opening the modal enables it
// (POST enabled=true), closing it disables it (POST enabled=false), so the firmware only streams
// while the panel is open. While open we poll once a second and render the
// on-device PEAK wake-word probability over the last second as a percent + a thin bar.
// The enable/disable are idempotent and tolerate React StrictMode's double-invoke (the
// server forwards `wake_peak_stream {on}` to the device; failures are swallowed since
// the device may be offline). `online` seeds reachability and is refreshed from each
// snapshot.
function WakeProbabilityMeter({ device, online: onlineProp }) {
  const [online, setOnline] = useState(onlineProp);
  const [value, setValue] = useState(null);         // latest percent (number) or null
  const [err, setErr] = useState(null);

  // Single effect owns the whole lifecycle: enable on open, poll while open, disable on
  // close. `device` is stable for the modal's lifetime, so the cleanup disables the same
  // device it enabled. Both enable and disable are best-effort (offline -> ignore).
  useEffect(() => {
    if (!device) return undefined;
    let alive = true;
    // Best-effort enable; ignore errors (the device may be offline).
    setWakeProbabilityStream(device, true).catch(() => {});
    const poll = () => getWakeProbability(device)
      .then((snap) => {
        if (!alive) return;
        setOnline(snap.online);
        setValue(snap.value ?? null);
        setErr(null);
      })
      .catch((e) => { if (alive) setErr(e); });
    poll();
    const iv = setInterval(poll, 1000);
    return () => {
      alive = false;
      clearInterval(iv);
      // Best-effort disable so the device stops streaming once nobody is watching.
      setWakeProbabilityStream(device, false).catch(() => {});
    };
  }, [device]);

  let body;
  if (!online) {
    body = <div className="z-fh">Device offline.</div>;
  } else {
    const pct = value == null ? null : Math.max(0, Math.min(100, value));
    // The slider's track and value markup without a thumb, so it lines up with the sliders below.
    body = <div className="z-slider">
      <div className="z-trk" style={{ cursor: "default" }}><i style={{ width: `${pct == null ? 0 : pct}%`, transition: "width .2s" }} /></div>
      <span className="z-sval">{pct == null ? "…" : `${Math.round(pct)}%`}</span>
    </div>;
  }

  return <Field row label="Wake probability" hint="On-device wake-word probability — peak over the last second.">
    <div className="z-rowctl">
      {body}
      {err && <div className="z-fh" style={{ color: "#b91c1c", marginTop: 6 }}>
        {(err && (err.message || String(err))) || "Failed to read wake probability."}
      </div>}
    </div>
  </Field>;
}

export function Devices() {
  const { catalog, patch, system, refreshConfig } = useAppData();
  const devices = catalog.core.values.devices || [];
  const devicePort = catalog.core.values.audio?.port ?? 8200;

  // Statuses arrive live over the panel WS heartbeat (system.devices is refreshed
  // every second), so no one-shot fetch / polling is needed here.
  const live = system?.devices || [];
  const liveOf = (name) => live.find((d) => d.name === name) || null;
  const statusOf = (name) => {
    const m = liveOf(name);
    if (!m) return "off";
    return m.online ? "online" : "offline";
  };
  // A device connecting for the first time is added to the config on the server; the
  // heartbeat lists it before this page's config snapshot does, so re-read the config.
  const unlisted = live.filter((m) => !devices.some((d) => d.name === m.name)).map((m) => m.name).join(",");
  useEffect(() => {
    if (unlisted) refreshConfig().catch(() => {});
  }, [unlisted]);

  const [modal, setModal] = useState(null); // { mode:'add'|'edit', index }
  const [busyErr, setBusyErr] = useState(null);

  // Apply `change` to the server's CURRENT device list, fetched right before the
  // write, so a cutoff set from a slider or a device that connected (and was added)
  // while the page was open is not rolled back. Devices are matched by name.
  const saveList = async (change) => {
    setBusyErr(null);
    try {
      const fresh = await getConfig();
      await patch({ core: { devices: change(fresh.core?.devices || []) } });
      setModal(null);
    } catch (e) { setBusyErr(e); }
  };
  const onAdd = (d) => saveList((list) => [...list, d]);
  const onEdit = (name, edit) => saveList((list) => list.map((x) => (
    x.name === name ? { ...x, ...edit } : x
  )));
  const onDelete = (name) => {
    if (!window.confirm(`Delete speaker "${deviceLabel(devices, name)}"?`)) return;
    saveList((list) => list.filter((x) => x.name !== name));
  };
  // Flip the persisted enabled flag (missing = true on old configs). Disabling closes
  // the device's connection and refuses it from then on; re-enabling admits it — both hot.
  const onToggle = (name) => saveList((list) => list.map((x) => (
    x.name === name ? { ...x, enabled: !(x.enabled ?? true) } : x
  )));

  return <div className="z-page">
    <PageHeader title="Devices" desc={`Devices connect to ws://<server>:${devicePort}/device; an unknown device is added on its first connection. Each device id also keys its own dialog context.`}
      actions={<button className="z-btn p" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add speaker</button>} />
    {busyErr && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span>{errorLines(busyErr).join(" · ")}</span></div>}
    <Card>
      <table className="z-tbl">
        <thead><tr><th>Name</th><th>Firmware</th><th>Wake word model</th><th>Uptime</th><th>Status</th><th></th></tr></thead>
        <tbody>
          {devices.length === 0
            ? <tr><td colSpan={6} style={{ color: "var(--mut)", padding: "14px 0" }}>No speakers configured.</td></tr>
            : devices.map((d, i) => {
              const enabled = d.enabled ?? true;
              const dim = enabled ? "" : "z-dim";
              return <tr key={i} style={{ cursor: "default" }}>
                <td className={dim} style={{ fontWeight: 600 }}>{d.label || d.name}</td>
                {/* Firmware/model versions come from the live WS snapshot; "—" while offline/disabled. */}
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{deviceVersion(liveOf(d.name), "config_version") ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{deviceVersion(liveOf(d.name), "model_version") ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{fmtUptime(liveOf(d.name)?.uptime_seconds)}</td>
                <td className={dim}>{enabled ? <StatusPill status={statusOf(d.name)} /> : <Pill tone="muted">disabled</Pill>}</td>
                <td style={{ textAlign: "right" }}><div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                  <button className="z-mini" onClick={() => onToggle(d.name)}>{enabled ? "Disable" : "Enable"}</button>
                  {/* Re-read the config first: the dialog starts from the device's saved label and
                      model, which may have changed elsewhere since this page loaded. */}
                  <button className="z-mini" onClick={() => refreshConfig().catch(() => {})
                    .finally(() => setModal({ mode: "edit", name: d.name }))}>Edit</button>
                  <button className="z-mini" onClick={() => onDelete(d.name)}>Delete</button>
                </div></td>
              </tr>;
            })}
        </tbody>
      </table>
    </Card>
    {modal?.mode === "add" && <DeviceModal title="Add speaker" onSave={onAdd} onClose={() => setModal(null)} />}
    {modal?.mode === "edit" && <DeviceModal key={modal.name} title="Edit speaker" onClose={() => setModal(null)}
      device={modal.name} label={devices.find((d) => d.name === modal.name)?.label || ""}
      wakeModel={devices.find((d) => d.name === modal.name)?.wake_model || ""}
      log={!!devices.find((d) => d.name === modal.name)?.log} logDays={catalog.core.values.device_log_days}
      autoVolume={!!devices.find((d) => d.name === modal.name)?.auto_volume}
      compressReplies={!!devices.find((d) => d.name === modal.name)?.compress_replies}
      onSave={(edit) => onEdit(modal.name, edit)} online={statusOf(modal.name) === "online"} />}
  </div>;
}
