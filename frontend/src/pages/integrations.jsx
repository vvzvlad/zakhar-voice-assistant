import React, { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Field, KeyInput, PageHeader, FormPage, StatusPill, Pill, Modal, Stepper, Loading, Select, Toggle,
} from "../components/primitives.jsx";
import { parseVoiceMarker, voiceField, voiceName, differsFromVoice } from "../voiceMarker.js";
import SchemaForm from "../components/SchemaForm.jsx";
import DeviceModal from "../components/DeviceModal.jsx";
import { useAppData } from "../appData.jsx";
import { errorLines } from "../useStageForm.js";
import { usePageForm } from "../usePageForm.js";
import { confirmDiscard } from "../navStore.js";
import { deref } from "../schema.js";
import { BUILTIN_SOURCES } from "../toolSources.js";
import {
  getPromptProfiles, getPromptProfile, createPromptProfile, updatePromptProfile,
  deletePromptProfile, activatePromptProfile,
  getTools, getOptions, getConfig,
} from "../api.js";
import { deviceLabel, deviceVersion, fmtUptime, fwVersion } from "../format.js";

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

// ── MCP / Integrations: external servers + built-in tool sources ──────────
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
    <PageHeader title="Tools" desc="Tool sources the model calls: external smart-home MCP servers and built-in weather/calendar/reminders/assistant tasks, plus the always-on speaker volume tool. Sources are applied live — rebuilt on save, no restart needed."
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

  // All hooks must run before the early-return below to satisfy the rules of hooks.
  const { catalog, reload } = useAppData();

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

  // The page's one Save persists the prompt text; save() keeps its own error, so it never rejects.
  const form = { dirty, saving, save };

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

  return <FormPage form={form} narrow errors={errorLines(saveErr)}>
    <PageHeader title="System Prompt"
      desc="Zakhar's character, rules and answer format. Keep several named profiles and activate the one the assistant should use. The placeholder is replaced with live date/time at request time."
      actions={<button className="z-btn g" onClick={() => { if (confirmDiscard(dirty)) load(selected?.id); }}>Reload</button>} />
    {/* Profile bar: pick a profile to edit; Activate switches what the pipeline uses.
        The profile's voice switch sits on its own line under the buttons. */}
    <div className="z-card" style={{ marginBottom: 14 }}>
      <div className="z-card-b" style={{ display: "flex", flexDirection: "column", gap: 10, padding: "10px 14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <Select w={260} value={selected?.id} options={profileOptions} onChange={selectProfile} />
          <button className="z-btn p" disabled={isActive || profileBusy} onClick={doActivate}
            title={isActive ? "This profile is already active" : "Make this profile the one the assistant uses"}>Activate</button>
          <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
            <button className="z-btn g" onClick={() => { if (confirmDiscard(dirty)) { setModalErr(null); setModal({ mode: "new" }); } }}><Ic n="add" w={14} />New</button>
            <button className="z-btn g" onClick={() => { setModalErr(null); setModal({ mode: "rename" }); }}>Rename</button>
            <button className="z-btn d" disabled={isActive || profileBusy} onClick={doDelete}
              title={isActive ? "The active profile cannot be deleted" : ""}>Delete</button>
          </div>
        </div>
        {voice && <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}
          title="Apply this profile's preferred TTS voice when activating">
          <Toggle on={applyVoice} onChange={setApplyVoice} sm />
          <span className="sub" data-testid="marker-voice">Apply voice on activate: {voiceText}</span>
          {voiceDiffers && <Pill tone="warn">differs from current TTS voice</Pill>}
        </div>}
      </div>
    </div>
    {/* The prompt syntax reference, collapsed: the date/time variable, the voice marker, stress marks. */}
    <div className="z-card" style={{ marginBottom: 14 }}><div className="z-card-b">
      <details>
        <summary className="z-disclosure">Syntax</summary>
        <div className="z-f">
          <div className="z-fl"><b>The date/time variable</b></div>
          <div className="z-info">
            <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<TDW>>>>>"}</span> is substituted at request time with the current <b>date and time</b>.
          </div>
        </div>
        <div className="z-f">
          <div className="z-fl"><b>Preferred voice</b></div>
          <div className="z-info">
            A single line <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=… field=value>>>>>"}</span> records a preferred TTS voice for this profile.
            The first pair <b>provider=</b> picks the TTS provider; the rest map to its fields (e.g. <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=yandex voice=zahar>>>>>"}</span>).
            When you activate the profile with the toggle on, this voice is applied to the TTS config. The marker is <b>stripped</b> from the prompt sent to the model.
          </div>
        </div>
        <div className="z-f">
          <div className="z-fl"><b>Notation</b></div>
          <div className="z-info">Stress marks use <b>«+»</b> before the stressed vowel (e.g. «з+амок»).</div>
        </div>
      </details>
    </div></div>
    <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{selected?.name} · {text.length} chars</span>} title="Editor"
      foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
      <div style={{ padding: "8px 0" }}>
        <textarea ref={taRef} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
          style={{ width: "100%", minHeight: 560, resize: "none", overflow: "hidden", boxSizing: "border-box", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" }} />
      </div>
    </Card>
    {modal?.mode === "new" && <ProfileNameModal title="New profile (copy of the active prompt)"
      onSave={doCreate} onClose={() => setModal(null)} busyErr={modalErr} />}
    {modal?.mode === "rename" && <ProfileNameModal title="Rename profile" initial={selected?.name}
      onSave={doRename} onClose={() => setModal(null)} busyErr={modalErr} />}
  </FormPage>;
}

// ── Devices ───────────────────────────────────────────────────────────────

export function Devices() {
  const { catalog, patch, system, refreshConfig } = useAppData();
  const devices = catalog.core.values.devices || [];

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

  return <div className="z-page">
    <PageHeader title="Devices" desc="An unknown device is added on its first connection. Each device id also keys its own dialog context."
      actions={<button className="z-btn g" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add speaker</button>} />
    {busyErr && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span>{errorLines(busyErr).join(" · ")}</span></div>}
    <Card>
      {/* On a narrow screen Firmware and Uptime hide (the dialog header has them) and what is
          still too wide scrolls inside the card, not the page. */}
      <div style={{ overflowX: "auto" }}>
      <table className="z-tbl z-devtbl">
        <thead><tr><th>Name</th><th>Firmware</th><th>Wake word model</th><th>Uptime</th><th>Status</th></tr></thead>
        <tbody>
          {devices.length === 0
            ? <tr><td colSpan={5} style={{ color: "var(--mut)", padding: "14px 0" }}>No speakers configured.</td></tr>
            : devices.map((d, i) => {
              const enabled = d.enabled ?? true;
              const dim = enabled ? "" : "z-dim";
              const fw = deviceVersion(liveOf(d.name), "config_version");
              // Re-read the config first: the dialog starts from the device's saved label and
              // model, which may have changed elsewhere since this page loaded.
              return <tr key={i} data-testid="device-row" onClick={() => refreshConfig().catch(() => {})
                .finally(() => setModal({ mode: "edit", name: d.name }))}>
                <td className={dim} style={{ fontWeight: 600 }}>{d.label || d.name}</td>
                {/* Firmware/model versions come from the live WS snapshot; "—" while offline/disabled.
                    The firmware's build hash is in its tooltip. */}
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }} title={fw ?? undefined}>{fwVersion(fw) ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{deviceVersion(liveOf(d.name), "model_version") ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{fmtUptime(liveOf(d.name)?.uptime_seconds)}</td>
                <td className={dim}>{enabled ? <StatusPill status={statusOf(d.name)} /> : <Pill tone="muted">disabled</Pill>}</td>
              </tr>;
            })}
        </tbody>
      </table>
      </div>
    </Card>
    {modal?.mode === "add" && <DeviceModal title="Add speaker" onSave={onAdd} onClose={() => setModal(null)} />}
    {modal?.mode === "edit" && <DeviceModal key={modal.name} title="Edit speaker" onClose={() => setModal(null)}
      device={modal.name} label={devices.find((d) => d.name === modal.name)?.label || ""}
      enabled={devices.find((d) => d.name === modal.name)?.enabled ?? true}
      wakeModel={devices.find((d) => d.name === modal.name)?.wake_model || ""}
      log={!!devices.find((d) => d.name === modal.name)?.log} logDays={catalog.core.values.device_log_days}
      autoVolume={!!devices.find((d) => d.name === modal.name)?.auto_volume}
      compressReplies={!!devices.find((d) => d.name === modal.name)?.compress_replies}
      onSave={(edit) => onEdit(modal.name, edit)} onDelete={() => onDelete(modal.name)}
      online={statusOf(modal.name) === "online"} />}
  </div>;
}
