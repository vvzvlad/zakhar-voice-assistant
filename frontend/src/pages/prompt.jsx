import React, { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import {
  Card, Field, PageHeader, FormPage, Pill, Modal, Loading,
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

// The editor header's "⋯" button and its menu of `items` ({label, onClick, danger,
// disabled}); the menu closes on a click outside it, when focus leaves it, on Escape
// (focus goes back to the button) and once an item is picked. A mousedown in the menu
// keeps the focus where it is, so a browser that does not focus a clicked button
// (Safari) does not blur the menu away before the click lands.
function MoreMenu({ items }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const btnRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const outside = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const escape = (e) => { if (e.key === "Escape") { setOpen(false); btnRef.current?.focus(); } };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [open]);
  return <div className="z-menuwrap" ref={ref}
    onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false); }}>
    <button ref={btnRef} type="button" className="z-btn g sm" aria-label="More actions" aria-haspopup="menu" aria-expanded={open}
      onClick={() => setOpen((o) => !o)}>⋯</button>
    {open && <div className="z-menu" role="menu" onMouseDown={(e) => e.preventDefault()}>
      {items.map((it) => <button key={it.label} type="button" role="menuitem" className={it.danger ? "d" : undefined}
        disabled={it.disabled} onClick={() => { setOpen(false); it.onClick(); }}>{it.label}</button>)}
    </div>}
  </div>;
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
    // The "auto" reset shortens the page for a moment, and the page's scroller clamps its
    // position to the shorter content: put the position back once the height is set.
    const scroller = el.closest(".z-scroll");
    const top = scroller?.scrollTop;
    el.style.height = "auto";
    const border = el.offsetHeight - el.clientHeight; // top+bottom borders (no scrollbar with overflow hidden)
    el.style.height = `${el.scrollHeight + border}px`;
    if (scroller) scroller.scrollTop = top;
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
          <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<MEMORY>>>>>"}</span> is substituted at request time with the <b>facts</b> the assistant saved with its remember tool (data/memory.json), one per line, or «(пусто)» when there are none or Memory is switched off on the Tools page.
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
  // Named prompt profiles: the list (summaries + active_id) drives the profiles column;
  // `selected` is the full profile whose text sits in the editor. The ACTIVE
  // profile is what the pipeline uses — selecting/editing never switches it,
  // only the explicit Make active button does.
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
  // The "with voice" box next to Make active, checked by default: activating the
  // profile then also applies its preferred voice. Absent when the profile has no marker.
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

  // --- profile actions --------------------------------------------------------
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
  // Preferred voice parsed from the SELECTED profile's last-saved text (`loaded`)
  // and the TTS provider speaking now (first enabled variant).
  const voice = parseVoiceMarker(loaded);
  const ttsCat = catalog.categories?.find((c) => c.id === "tts");
  const ttsProv = (id) => ttsCat?.providers.find((p) => p.id === id);
  const current = ttsCat?.variants?.find((v) => v.enabled)?.provider;
  // Each row names its profile's voice (the list carries every profile's parsed marker).
  // An opaque voice id (one whose name the TTS form keeps in "<field>_label") that is not
  // its provider's current voice is named by the options the form picks it from, fetched
  // once per provider and field.
  const lookupKey = (v) => `${v.provider}/${voiceField(v)}`;
  const needsLookup = (v) => {
    const f = voiceField(v);
    const prov = f && ttsProv(v.provider);
    const values = prov?.values || {};
    return !!prov?.schema?.properties?.[`${f}_label`] && !(values[f] === v.fields[f] && values[`${f}_label`]);
  };
  // The distinct keys as one string, so the effect runs again only when the set changes.
  const lookups = [...new Set((profiles || []).map((p) => p.voice).filter(needsLookup).map(lookupKey))].sort().join(" ");
  const [names, setNames] = useState({});   // lookupKey -> {voice id: name} from the options endpoint
  useEffect(() => {
    if (!lookups) return undefined;
    let alive = true;
    for (const key of lookups.split(" ")) {
      const [provider, field] = key.split("/");
      getOptions("tts", provider, field)
        .then((r) => {
          const byId = Object.fromEntries((r.options || []).filter(Boolean).map((o) => [o.value, o.label]));
          if (alive) setNames((n) => ({ ...n, [key]: byId }));
        })
        .catch(() => { /* named by its id */ });
    }
    return () => { alive = false; };
  }, [lookups]);

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

  const isActive = selected?.id === activeId;
  // A profile's voice for its row: `text` is "<provider label> · <voice name>", the way the
  // TTS form names it; a voice named only by a long id shows its first 8 characters, and
  // `longId` holds the whole id for the row's title.
  const voiceOf = (v) => {
    const prov = ttsProv(v.provider);
    const f = voiceField(v);
    const name = voiceName(v, prov?.values || {}, names[lookupKey(v)]?.[v.fields[f]]);
    const longId = f && name === v.fields[f] && name.length > 12 ? name : undefined;
    return { text: [prov?.label || v.provider, longId ? `${name.slice(0, 8)}…` : name].join(" · "), longId };
  };
  const voiceDiffers = differsFromVoice(voice, current, ttsProv(current)?.values);
  const menuItems = [{ label: "Rename…", onClick: () => { setModalErr(null); setModal({ mode: "rename" }); } }];
  if (!isActive) menuItems.push({ label: "Delete", danger: true, disabled: profileBusy, onClick: doDelete });

  return <FormPage form={form} errors={errorLines(saveErr)}>
    <PageHeader title="Character Prompt"
      desc="Zakhar's character, added after the system prompt. Keep several named profiles and activate the one the assistant should use. The placeholders are replaced at request time with the live date/time, the wake word and the assistant's memory."
      actions={<button className="z-btn g" onClick={() => { if (confirmDiscard(dirty)) load(selected?.id); }}>Reload</button>} />
    {/* The profiles column: a row opens its profile in the editor on the right. */}
    <div className="z-prompt">
      <div className="z-card z-profiles">
        <div className="z-profiles-h">Profiles
          <button type="button" className="z-btn g sm" aria-label="New profile"
            onClick={() => { if (confirmDiscard(dirty)) { setModalErr(null); setModal({ mode: "new" }); } }}><Ic n="add" w={13} /></button>
        </div>
        {profiles.map((p) => {
          const pv = p.voice && voiceOf(p.voice);
          return <button key={p.id} type="button" className={"z-prof" + (p.id === selected?.id ? " on" : "")}
            aria-current={p.id === selected?.id ? "true" : undefined} title={pv?.longId} onClick={() => selectProfile(p.id)}>
            <span className="n">{p.name}{p.id === activeId && <Pill tone="good">Active</Pill>}</span>
            <span className="v">{[pv?.text, `${p.chars} chars`].filter(Boolean).join(" · ")}</span>
          </button>;
        })}
      </div>
      <div style={{ minWidth: 0 }}>
        {/* The syntax reference: the variables, then the profile's voice marker. */}
        <SyntaxCard>
          <div className="z-f">
            <div className="z-fl"><b>Preferred voice</b></div>
            <div className="z-info">
              A single line <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=… field=value>>>>>"}</span> records a preferred TTS voice for this profile.
              The first pair <b>provider=</b> picks the TTS provider; the rest map to its fields (e.g. <span className="z-paramtag" style={{ fontSize: 11 }}>{"<<<<<VOICE provider=yandex voice=zahar>>>>>"}</span>).
              When you activate the profile with "with voice" checked, this voice is applied to the TTS config. The marker is <b>stripped</b> from the prompt sent to the model.
            </div>
          </div>
        </SyntaxCard>
        {/* The editor's header holds the selected profile's state and actions; on a narrow
            card the actions drop under the name. */}
        <div className="z-card">
          <div className="z-card-h" style={{ flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, minWidth: 0 }}>
              <b style={{ overflowWrap: "anywhere" }}>{selected?.name}</b>
              {isActive && <Pill tone="good">Active</Pill>}
              {voiceDiffers && <Pill tone="warn">differs from current TTS voice</Pill>}
            </div>
            <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, marginLeft: "auto" }}>
              <span className="sub">{text.length} chars</span>
              {!isActive && <button type="button" className="z-btn p sm" disabled={profileBusy} onClick={doActivate}
                title="Make this profile the one the assistant uses">Make active</button>}
              {!isActive && voice && <label className="z-check" title="Apply this profile's preferred TTS voice when activating">
                <input type="checkbox" checked={applyVoice} onChange={(e) => setApplyVoice(e.target.checked)} />with voice
              </label>}
              <MoreMenu items={menuItems} />
            </div>
          </div>
          <div className="z-card-b">
            <div style={{ padding: "8px 0" }}>
              <textarea ref={taRef} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
                style={EDITOR_STYLE} />
            </div>
          </div>
          {err && <div className="z-foot"><span className="z-dirty" style={{ color: "#b91c1c" }}>{errorLines(err).join(" · ")}</span></div>}
        </div>
      </div>
    </div>
    {modal?.mode === "new" && <ProfileNameModal title="New profile (copy of the active prompt)"
      onSave={doCreate} onClose={() => setModal(null)} busyErr={modalErr} />}
    {modal?.mode === "rename" && <ProfileNameModal title="Rename profile" initial={selected?.name}
      onSave={doRename} onClose={() => setModal(null)} busyErr={modalErr} />}
  </FormPage>;
}
