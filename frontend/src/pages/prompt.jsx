import React, { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Card, Field, PageHeader, FormPage, Pill, Modal, Loading, Select, Toggle,
} from "../components/primitives.jsx";
import { parseVoiceMarker, voiceField, voiceName, differsFromVoice } from "../voiceMarker.js";
import { useAppData } from "../appData.jsx";
import { errorLines } from "../useStageForm.js";
import { confirmDiscard } from "../navStore.js";
import {
  getPromptProfiles, getPromptProfile, createPromptProfile, updatePromptProfile,
  deletePromptProfile, activatePromptProfile, getOptions,
} from "../api.js";

// ── System and character prompts ──────────────────────────────────────────
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

// The prompt editors' textarea: grown to its content by useAutoGrow.
const EDITOR_STYLE = { width: "100%", minHeight: 560, resize: "none", overflow: "hidden", boxSizing: "border-box", border: "1px solid var(--line)", borderRadius: 8, padding: "13px 15px", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.65, color: "var(--ink)", outline: "none", background: "var(--panel2)" };

// Resize an editor textarea to fit its content so the whole prompt is visible without
// an inner scrollbar; returns the textarea's ref. Reset to "auto" first so scrollHeight
// reflects the content height, not the current box; then add the vertical border so
// border-box sizing leaves no clipped or overflowing pixels. Recomputed whenever
// `value` changes (initial load and every edit) and on window resize, since wrapping
// (and thus height) depends on the textarea width.
function useAutoGrow(value) {
  const ref = useRef(null);
  const autoGrow = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    const border = el.offsetHeight - el.clientHeight; // top+bottom borders (no scrollbar with overflow hidden)
    el.style.height = `${el.scrollHeight + border}px`;
  }, []);
  useLayoutEffect(() => { autoGrow(); }, [value, autoGrow]);
  useEffect(() => {
    window.addEventListener("resize", autoGrow);
    return () => window.removeEventListener("resize", autoGrow);
  }, [autoGrow]);
  return ref;
}

// The prompt syntax reference, collapsed: the variables both the system prompt and a
// character profile take; `children` adds the page's own entries.
function SyntaxCard({ children }) {
  return <div className="z-card" style={{ marginBottom: 14 }}><div className="z-card-b">
    <details>
      <summary className="z-disclosure">Syntax</summary>
      <div className="z-f">
        <div className="z-fl"><b>The date/time variable</b></div>
        <div className="z-info">
          <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<TDW>>>>>"}</span> is substituted at request time with the current <b>date and time</b>.
        </div>
      </div>
      <div className="z-f">
        <div className="z-fl"><b>The wake-word variable</b></div>
        <div className="z-info">
          <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<WAKE_WORD>>>>>"}</span> is substituted at request time with the <b>wake word</b> of the device's wake-word model, from the model's manifest; for a model without one, the word the device reported.
        </div>
      </div>
      <div className="z-f">
        <div className="z-fl"><b>The memory variable</b></div>
        <div className="z-info">
          <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<MEMORY>>>>>"}</span> is substituted at request time with the <b>facts</b> the assistant saved with its remember tool (data/memory.json), one per line, or «(пусто)» when there are none.
        </div>
      </div>
      {children}
    </details>
  </div></div>;
}

// The system prompt (core.prompt.system): the rules of any character, put before the
// active character profile in every request.
export function SystemPrompt() {
  const { catalog, reload, patch } = useAppData();
  // `edit` is the unsaved text, null while it equals the stored value.
  const stored = catalog.core.values.prompt?.system ?? "";
  const [edit, setEdit] = useState(null);
  const draft = edit ?? stored;
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);            // the reload's error, shown under the editor
  const [saveErr, setSaveErr] = useState(null);    // the save's error, shown in the save bar
  const taRef = useAutoGrow(draft);

  const dirty = edit != null && edit !== stored;
  const save = async () => {
    setSaving(true); setSaveErr(null);
    try {
      await patch({ core: { prompt: { system: edit } } });
      setEdit(null);
    } catch (e) { setSaveErr(e); }
    finally { setSaving(false); }
  };
  // Reload drops the unsaved text and takes the stored one from a fresh catalog.
  const reloadPage = async () => {
    if (!confirmDiscard(dirty)) return;
    setEdit(null);
    try { await reload(); setErr(null); } catch (e) { setErr(e); }
  };

  // The page's one Save persists the text; save() keeps its own error, so it never rejects.
  const form = { dirty, saving, save };

  return <FormPage form={form} narrow errors={errorLines(saveErr)}>
    <PageHeader title="System Prompt"
      desc="The rules of any character: this text goes before the active character prompt in every request and is the same for every character. The placeholders are replaced at request time with the live date/time, the wake word and the assistant's memory."
      actions={<button className="z-btn g" onClick={reloadPage}>Reload</button>} />
    <SyntaxCard />
    <Card title="Editor"
      foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
      <div style={{ padding: "8px 0" }}>
        <textarea ref={taRef} value={draft} onChange={(e) => setEdit(e.target.value)} spellCheck={false}
          style={EDITOR_STYLE} />
      </div>
    </Card>
  </FormPage>;
}

// The character prompt: named profiles of Zakhar's character, one of them active.
export function Prompt() {
  const { catalog, reload } = useAppData();
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

  const taRef = useAutoGrow(text);

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
    <PageHeader title="Character Prompt"
      desc="Zakhar's character, added after the system prompt. Keep several named profiles and activate the one the assistant should use. The placeholders are replaced at request time with the live date/time, the wake word and the assistant's memory."
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
    {/* The syntax reference: the variables, then the profile's voice marker. */}
    <SyntaxCard>
      <div className="z-f">
        <div className="z-fl"><b>Preferred voice</b></div>
        <div className="z-info">
          A single line <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=… field=value>>>>>"}</span> records a preferred TTS voice for this profile.
          The first pair <b>provider=</b> picks the TTS provider; the rest map to its fields (e.g. <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=yandex voice=zahar>>>>>"}</span>).
          When you activate the profile with the toggle on, this voice is applied to the TTS config. The marker is <b>stripped</b> from the prompt sent to the model.
        </div>
      </div>
    </SyntaxCard>
    <Card right={<span className="sub" style={{ marginLeft: "auto" }}>{selected?.name} · {text.length} chars</span>} title="Editor"
      foot={err ? <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div> : undefined}>
      <div style={{ padding: "8px 0" }}>
        <textarea ref={taRef} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
          style={EDITOR_STYLE} />
      </div>
    </Card>
    {modal?.mode === "new" && <ProfileNameModal title="New profile (copy of the active prompt)"
      onSave={doCreate} onClose={() => setModal(null)} busyErr={modalErr} />}
    {modal?.mode === "rename" && <ProfileNameModal title="Rename profile" initial={selected?.name}
      onSave={doRename} onClose={() => setModal(null)} busyErr={modalErr} />}
  </FormPage>;
}
