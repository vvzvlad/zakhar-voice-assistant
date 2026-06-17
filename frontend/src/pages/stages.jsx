import React, { useState, useEffect, useRef } from "react";
import { Selector, PageHeader, FormSaveBar, Seg } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getOptions, getChimes, playChime, streamTtsVoice, getTools } from "../api.js";
import { playAudioResponse } from "../streamAudio.js";
import { buildVoiceMarker } from "../voiceMarker.js";
import { parseActions, parseAliasText, serializeAliasText, serializeActions, allEnumSlots, classifySlotKind } from "../nluAliases.js";

function Card({ title, sub, children, foot }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
    <div className="z-card-b">{children}</div>
    {foot}
  </div>;
}

// Shared end-of-phrase chime preview: plays a chime on every online speaker via the
// same announce path as the real ack. `busy` holds the sound_path currently playing
// (or null); `msg` is the last error (success is silent). Used by the "Play now" button AND the
// per-item play buttons in the chime dropdown, so they share one busy flag + status.
function useChimePreview() {
  const [busy, setBusy] = useState(null); // sound_path being played, or null
  const [msg, setMsg] = useState(null);   // { tone: "ok" | "err", text }
  const play = async (soundPath) => {
    const sp = soundPath || "";
    setBusy(sp);
    setMsg(null);
    try {
      const r = await playChime(sp);
      const played = (r && r.played) || [];
      const offline = (r && r.offline) || [];
      // Success is self-evident (the chime plays through the speaker), so show no
      // confirmation text — only surface a message when nothing actually played.
      if (played.length === 0) {
        setMsg({ tone: "err", text: offline.length ? `No online speaker (offline: ${offline.join(", ")})` : "No speakers configured" });
      }
    } catch (e) {
      setMsg({ tone: "err", text: e.message || "Failed to play" });
    } finally {
      setBusy(null);
    }
  };
  return { busy, msg, play };
}

// Voice-test phrase persistence (survives page reloads). localStorage can throw
// in privacy modes, so every access is guarded — the state then lives in memory.
const TTS_TEST_TEXT_KEY = "zakhar.tts.test_text";
const TTS_TEST_DEFAULT_TEXT = "Привет! Это проверка голоса: раз, два, три.";

function loadTestText() {
  try {
    return localStorage.getItem(TTS_TEST_TEXT_KEY) || TTS_TEST_DEFAULT_TEXT;
  } catch {
    return TTS_TEST_DEFAULT_TEXT;
  }
}

// Universal TTS voice test: synthesizes the typed phrase with the CURRENT
// (possibly unsaved) provider form draft via POST /api/tts/test and plays the
// result in the browser. Provider-agnostic: the draft dict is sent verbatim and
// validated server-side by the provider's own ConfigModel, so any TTS provider
// works with zero per-provider code here.
function VoiceTestCard({ provider, settings }) {
  const [text, setText] = useState(loadTestText);
  // Button phase: null (idle) | "synth" (request in flight) | "playing" (clip audible).
  const [phase, setPhase] = useState(null);
  const [msg, setMsg] = useState(null); // last error text (success is silent)
  const ctlRef = useRef(null);   // current playAudioResponse controller, if any
  const abortRef = useRef(null); // AbortController for the in-flight request
  // Per-run identity token: callbacks capture the run that started them and bail
  // if it is no longer the active run, so a late event from a superseded clip (or
  // a stale request failure) can never touch a newer playback.
  const runRef = useRef(null);

  const onText = (v) => {
    setText(v);
    try { localStorage.setItem(TTS_TEST_TEXT_KEY, v); } catch { /* privacy mode: keep in memory */ }
  };

  // Stop playback (if any), abort any in-flight request and return to idle.
  // Idempotent (the controller's stop() and AbortController.abort() are both safe
  // to call repeatedly / after completion).
  const cleanup = () => {
    if (ctlRef.current) {
      try { ctlRef.current.stop(); } catch { /* already torn down — ignore */ }
      ctlRef.current = null;
    }
    if (abortRef.current) {
      try { abortRef.current.abort(); } catch { /* already aborted — ignore */ }
      abortRef.current = null;
    }
    setPhase(null);
  };

  const test = async () => {
    const myRun = {};
    runRef.current = myRun;
    const ac = new AbortController();
    abortRef.current = ac;
    setPhase("synth");
    setMsg(null);
    try {
      const resp = await streamTtsVoice(provider, settings, text, ac.signal);
      if (runRef.current !== myRun) return;           // superseded/stopped while awaiting
      const ctl = playAudioResponse(resp, {
        onPlaying: () => { if (runRef.current === myRun) setPhase("playing"); },
        onEnded:   () => { if (runRef.current === myRun) cleanup(); },
        onError:   (m) => { if (runRef.current !== myRun) return; setMsg(m); cleanup(); },
      });
      ctlRef.current = ctl;
    } catch (e) {
      if (runRef.current !== myRun) return;           // stale failure from a superseded run
      if (e.name === "AbortError") return;            // user Stop / unmount aborted — not an error
      setMsg(e.message || "Voice test failed");
      cleanup();
    }
  };

  const stop = () => cleanup(); // stop playback + abort + back to idle

  // Stop audio and free the URL if the user navigates away mid-playback.
  useEffect(() => () => cleanup(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const playing = phase === "playing";

  return <Card title="Test voice" sub="hear the draft settings in the browser"
    foot={
      // Footer mirrors FormSaveBar: button + hint on the left, error right-aligned
      // where the save bar shows its "Unsaved" badge.
      <div className="z-foot">
        {/* One button, three phases: idle → Synthesizing (disabled) → Stop (danger style). */}
        <button
          className={playing ? "z-btn d" : "z-btn p"}
          disabled={playing ? false : phase === "synth" || !text.trim()}
          onClick={playing ? stop : test}
        >
          {playing ? "Stop" : phase === "synth" ? "Synthesizing…" : "Test voice"}
        </button>
        {/* marginTop: 0 cancels .z-fh's 3px top margin so the text centers against the button. */}
        <span className="z-fh" style={{ marginTop: 0 }}>Speak the phrase with the current (unsaved) settings.</span>
        <span style={{ flex: 1 }} />
        {msg && <span className="z-fh" style={{ marginTop: 0, color: "#b91c1c" }}>{msg}</span>}
      </div>
    }>
    {/* .z-f gives the card body the standard field rhythm (vertical padding + gap). */}
    <div className="z-f">
      <div className="z-inp">
        <input
          value={text}
          placeholder="Phrase to synthesize"
          onChange={(e) => onText(e.target.value)}
        />
      </div>
    </div>
  </Card>;
}

// Read-only "preferred voice" marker built from the CURRENT (possibly unsaved)
// TTS draft: the user copies it and pastes it into a system-prompt profile to
// pin this voice. Mirrors the VoiceTestCard styling; the marker string is built
// by buildVoiceMarker (same secret-exclusion / whitespace rules as the server).
function VoiceMarkerCard({ provider, settings }) {
  const marker = buildVoiceMarker(provider, settings);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef(null);
  const [msg, setMsg] = useState(null); // last error text (success shows on the button)
  // Guard the transient "Copied" timer so it cannot setState after unmount.
  const mountedRef = useRef(true);
  const timerRef = useRef(null);
  useEffect(() => () => {
    mountedRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const copy = async () => {
    setMsg(null);
    try {
      await navigator.clipboard.writeText(marker);
    } catch {
      // Clipboard API unavailable (insecure context / older browser): fall back
      // to selecting the input and the legacy execCommand copy.
      try {
        const el = inputRef.current;
        if (el) {
          el.focus();
          el.select();
          document.execCommand("copy");
        } else {
          throw new Error("no input");
        }
      } catch {
        setMsg("Copy failed — select the field and copy manually");
        return;
      }
    }
    setCopied(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (mountedRef.current) setCopied(false);
    }, 1500);
  };

  return <Card title="Prompt voice marker" sub="paste into a prompt profile to pin this voice"
    foot={
      <div className="z-foot">
        <button className="z-btn p" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
        <span className="z-fh" style={{ marginTop: 0 }}>Copy and paste into a prompt profile to pin this voice.</span>
        <span style={{ flex: 1 }} />
        {msg && <span className="z-fh" style={{ marginTop: 0, color: "#b91c1c" }}>{msg}</span>}
      </div>
    }>
    <div className="z-f">
      <div className="z-inp">
        <input
          ref={inputRef}
          value={marker}
          readOnly
          onFocus={(e) => e.target.select()}
        />
      </div>
    </div>
  </Card>;
}

// Inline catalog editor for the offline-NLU (simple-nlu) provider. Instead of
// hand-editing the raw `aliases`/`actions` textareas, it discovers every enum slot
// live from the MCP tool catalog and offers one input per unique enum VALUE. Each
// slot is classified as an ENTITY (device/scene id → Russian phrases in `aliases`)
// or an ACTION (state/command verb → Russian verbs in `actions`); a per-slot toggle
// lets the operator reclassify an ambiguous slot. The source of truth stays
// `draft.aliases`/`draft.actions`: every keystroke re-serializes the whole field,
// so this editor and the SchemaForm textareas stay in sync.
export function NluCatalogEditor({ draft, onChange }) {
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null); // soft hint, never thrown
  // `overrides` is transient (local, not persisted): keyed by "<tool>.<slot>" →
  // "entity"|"action". The slot-name / ⊆actionNames heuristic in classifySlotKind
  // classifies correctly by default for this catalog (state/action → action;
  // device_id/scene → entity), so the toggle is only needed to reclassify an
  // ambiguous slot. It resets on remount — acceptable for a manual override.
  const [overrides, setOverrides] = useState({});

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await getTools();
        if (alive) setSources((data && data.sources) || []);
      } catch (e) {
        if (alive) setError(e.message || "Не удалось загрузить инструменты");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  // Derive everything from props each render — no duplicate local copy of the
  // text, so the textareas in SchemaForm remain the single source of truth.
  const { map: nameToVerbs, extraLines: actionExtras } = parseActions(draft.actions || "");
  const actionNames = Object.keys(nameToVerbs);
  const { map: aliasMap, extraLines: aliasExtras } = parseAliasText(draft.aliases || "");
  const slots = allEnumSlots(sources);
  const kindOf = (s) => classifySlotKind(s.slot, s.values, actionNames, overrides[`${s.tool}.${s.slot}`]);
  const entitySlots = slots.filter((s) => kindOf(s) === "entity");
  const actionSlots = slots.filter((s) => kindOf(s) === "action");
  // ALL catalog values — used to keep an action-slot value out of `aliases`.
  const catalogValues = new Set(slots.flatMap((s) => s.values));

  // Apply an entity value's edit: rebuild the whole aliases field, preserving the
  // extra/advanced lines and never re-emitting a value owned by an action slot.
  const setAliasPhrase = (value, text) => {
    onChange("aliases", serializeAliasText(entitySlots, { ...aliasMap, [value]: text }, aliasExtras, catalogValues));
  };

  // Apply an action value's edit: rebuild the whole actions field from the live
  // verb map + this value's new verbs, preserving the extra lines.
  const setActionVerbs = (value, text) => {
    onChange("actions", serializeActions(actionSlots, { ...nameToVerbs, [value]: text }, actionExtras));
  };

  // Flip a slot's classification (entity <-> action) via a transient override.
  const toggleKind = (s) => {
    const key = `${s.tool}.${s.slot}`;
    const cur = kindOf(s);
    const next = cur === "entity" ? "action" : "entity";
    // Strip this slot's values from the field of the OLD kind so a value never
    // lives in both `aliases` and `actions` after a reclassification.
    if (cur === "entity") {
      const cleaned = { ...aliasMap };
      let changed = false;
      for (const v of s.values) { if (v in cleaned) { delete cleaned[v]; changed = true; } }
      if (changed) onChange("aliases", serializeAliasText(entitySlots, cleaned, aliasExtras, catalogValues));
    } else {
      const cleaned = { ...nameToVerbs };
      let changed = false;
      for (const v of s.values) { if (v in cleaned) { delete cleaned[v]; changed = true; } }
      if (changed) onChange("actions", serializeActions(actionSlots, cleaned, actionExtras));
    }
    setOverrides((prev) => ({ ...prev, [key]: next }));
  };

  // One row per unique enum VALUE across ALL slots; group rows under their owning
  // tool header. Each slot carries a small kind toggle; each value's input follows
  // the OWNING slot's classification (entity → aliases, action → actions).
  const rows = [];
  let lastTool = null;
  let lastSlot = null;
  const seen = new Set();
  for (const s of slots) {
    if (s.tool !== lastTool) {
      // Tool header doubles as a section divider: the first tool sits flush, every
      // later tool gets a top rule so the value groups read as distinct sections.
      const firstTool = rows.length === 0;
      rows.push(
        <div
          key={`h-${s.tool}`}
          style={{
            fontSize: 12.5,
            fontWeight: 600,
            color: "var(--ink2)",
            ...(firstTool
              ? { marginTop: 0 }
              : { marginTop: 18, paddingTop: 14, borderTop: "1px solid var(--line2)" }),
          }}
        >
          {s.tool}
        </div>
      );
      lastTool = s.tool;
      lastSlot = null;
    }
    const kind = kindOf(s);
    // Per-slot kind toggle, rendered once per slot (above its value inputs).
    if (`${s.tool}.${s.slot}` !== lastSlot) {
      lastSlot = `${s.tool}.${s.slot}`;
      // The server treats a slot as a state slot only if EVERY enum value is an
      // action name (set(enum) <= action_names). A partially-filled action slot
      // (some verbs present, some empty) silently breaks the command — warn.
      const anyFilled = s.values.some((v) => String(nameToVerbs[v] || "").trim() !== "");
      const anyEmpty = s.values.some((v) => String(nameToVerbs[v] || "").trim() === "");
      const partial = kind === "action" && anyFilled && anyEmpty;
      // Slot header: mono slot label + compact Seg kind toggle + (if partial) a warn pill.
      rows.push(
        <div key={`k-${s.tool}.${s.slot}`} style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 12 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--mut2)" }}>{s.slot}</span>
          <Seg
            options={["Устройство", "Действие"]}
            value={kind === "entity" ? "Устройство" : "Действие"}
            onChange={(v) => { const want = v === "Действие" ? "action" : "entity"; if (want !== kind) toggleKind(s); }}
          />
          {partial && <span className="z-pill warn">не все значения заполнены</span>}
        </div>
      );
    }
    for (const value of s.values) {
      if (seen.has(value)) continue; // one input per unique value
      seen.add(value);
      const isEntity = kind === "entity";
      // Value row: bare enum value chip (the slot name already shows in the header) + a
      // compact input following the owning slot's kind (entity → aliases, action → actions).
      rows.push(
        <div key={value} style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 10, alignItems: "center", padding: "3px 0" }}>
          <code style={{ fontSize: 11.5, color: "var(--mut)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value}</code>
          <div className="z-inp sm">
            <input
              value={isEntity ? (aliasMap[value] || "") : (nameToVerbs[value] || "")}
              placeholder={isEntity ? "русские названия через запятую" : "глаголы-команды через запятую"}
              onChange={(e) => (isEntity ? setAliasPhrase(value, e.target.value) : setActionVerbs(value, e.target.value))}
            />
          </div>
        </div>
      );
    }
  }

  return <Card title="Сущности и команды (из MCP)" sub="устройства/сцены → русские названия; состояния (on/off, lock/unlock) → глаголы; переключатель меняет тип слота">
    <div className="z-f">
      {loading ? (
        <div className="z-fh" style={{ marginTop: 0 }}>Загрузка…</div>
      ) : error ? (
        <div className="z-fh" style={{ marginTop: 0, color: "#b91c1c" }}>{error}</div>
      ) : slots.length === 0 ? (
        <div className="z-fh" style={{ marginTop: 0 }}>
          Инструменты из MCP не обнаружены — проверь, что сервер умного дома доступен и включён.
        </div>
      ) : (
        rows
      )}
      <div className="z-fh" style={{ marginTop: 12, opacity: 0.8 }}>
        Числовые слоты (яркость, температура) задаются числом в самой команде — алиасы для них не нужны.
      </div>
    </div>
  </Card>;
}

// ── Generic provider stage (STT / LLM / TTS) ──────────────────────────────
function ProviderStage({ cat, title, crumb, desc }) {
  const { catalog, patch, system } = useAppData();
  // This stage's backend is being hot-reloaded (model load in flight) per the heartbeat.
  const isReloading = (system?.reloading || []).includes(cat);
  const category = catalog.categories.find((c) => c.id === cat);
  const selected = category.selected;
  const prov = category.providers.find((p) => p.id === selected) || category.providers[0];

  const buildPatch = (draft) => ({ [cat]: { selected, instances: { [selected]: draft } } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(prov.values, buildPatch, patch);

  const switchProvider = async (newId) => {
    if (newId === selected) return;
    try { await patch({ [cat]: { selected: newId } }); } catch { /* surfaced elsewhere */ }
  };

  // `q` (optional) is the server-side search string for remote-search fields;
  // forwarded only when present so plain fields keep the legacy call shape.
  const optionsFor = async (field, q) => {
    const r = await (q ? getOptions(cat, selected, field, q) : getOptions(cat, selected, field));
    return r.options;
  };

  return <div className="z-page narrow">
    <PageHeader title={title} crumb={crumb} desc={desc} />
    {isReloading && (
      <div className="z-banner warn">
        <span className="z-spin" />
        <span><b>Applying…</b> loading the {prov.label} backend — downloading the model on first use can take a while.</span>
      </div>
    )}
    <Selector
      label="Provider"
      options={category.providers.map((p) => p.id)}
      value={selected}
      onChange={switchProvider}
      caption={prov.label}
    />
    <div className="z-grid">
      <Card title={prov.label} foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
        {/* key={selected}: dynamic selects fetch their option list once on mount, and
            providers often share an identical field set, so without the key a provider
            switch would be reconciled in place and keep the PREVIOUS provider's options
            (e.g. OpenRouter's model catalog shown under Groq). The key forces a remount
            so every DynamicSelect refetches for the newly selected provider. */}
        <SchemaForm key={selected} schema={prov.schema} values={draft} onChange={onChange} optionsFor={optionsFor} />
      </Card>
      {/* Offline-NLU only: per-value entity/action inputs sourced live from the MCP catalog. */}
      {cat === "llm" && selected === "simple-nlu" && <NluCatalogEditor draft={draft} onChange={onChange} />}
      {/* TTS only: test the CURRENT (unsaved) draft settings with an in-browser playback. */}
      {cat === "tts" && <VoiceTestCard provider={selected} settings={draft} />}
      {/* TTS only: ready-to-copy prompt marker built from the same live draft. */}
      {cat === "tts" && <VoiceMarkerCard provider={selected} settings={draft} />}
    </div>
  </div>;
}

export function Wakeword() {
  return <ProviderStage cat="wakeword" title="Wakeword · Verify" crumb="Pipeline / Stage 02"
    desc="Server-side wake-word verifier. Re-checks the captured phrase actually started with the wake word and rejects false triggers before STT." />;
}
export function STT() {
  return <ProviderStage cat="stt" title="STT · Speech to text" crumb="Pipeline / Stage 03"
    desc="Recognize the captured phrase. Cloud Whisper or offline Vosk." />;
}
export function LLM() {
  return <ProviderStage cat="llm" title="LLM · Reasoning & tools" crumb="Pipeline / Stage 04"
    desc="Generates the reply and calls smart-home tools over MCP. OpenAI-compatible chat completions." />;
}
export function Accents() {
  return <ProviderStage cat="stress" title="Accents · Stress placement" crumb="Pipeline / Stage 05"
    desc="Place Russian word stress on the reply text so TTS pronounces it correctly." />;
}
export function TTS() {
  return <ProviderStage cat="tts" title="TTS · Text to speech" crumb="Pipeline / Stage 06"
    desc="Synthesize the reply to audio served to the speakers." />;
}

// ── VAD (catalog category + core sub-sections) ────────────────────────────

// Embeddable provider card for one catalog category: selector + schema form +
// save bar — same mechanics as ProviderStage, minus the PageHeader. Render it
// only when the category exists in the catalog.
function ProviderCard({ cat, sub }) {
  const { catalog, patch } = useAppData();
  const category = catalog.categories.find((c) => c.id === cat);
  const selected = category.selected;
  const prov = category.providers.find((p) => p.id === selected) || category.providers[0];

  const buildPatch = (draft) => ({ [cat]: { selected, instances: { [selected]: draft } } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(prov.values, buildPatch, patch);

  const switchProvider = async (newId) => {
    if (newId === selected) return;
    try { await patch({ [cat]: { selected: newId } }); } catch { /* surfaced elsewhere */ }
  };

  // `q` (optional) is the server-side search string for remote-search fields;
  // forwarded only when present so plain fields keep the legacy call shape.
  const optionsFor = async (field, q) => {
    const r = await (q ? getOptions(cat, selected, field, q) : getOptions(cat, selected, field));
    return r.options;
  };

  return <Card title={prov.label} sub={sub} foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
    <Selector
      label="Provider"
      options={category.providers.map((p) => p.id)}
      value={selected}
      onChange={switchProvider}
      caption={prov.label}
    />
    {/* key={selected}: see ProviderStage — remount the form on provider switch so
        DynamicSelect fields (which fetch options on mount) refetch for the new
        provider instead of keeping the previous provider's option list. */}
    <SchemaForm key={selected} schema={prov.schema} values={draft} onChange={onChange} optionsFor={optionsFor} />
  </Card>;
}

export function VAD() {
  const { catalog, patch } = useAppData();
  // VAD is a catalog category since R2 (webrtc provider: aggressiveness).
  // Tolerate an older backend without it — render the core.vad cards only.
  const vadCat = catalog.categories.find((c) => c.id === "vad");
  const coreSchema = catalog.core.schema;
  // vad prop is a $ref to $defs.VadConfig — resolve to the object schema.
  const vadSchema = coreSchema.$defs ? coreSchema.$defs.VadConfig : null;
  const rawVad = catalog.core.values.vad || {};
  // mic_channel / mic_normalize / mic_highpass live in VadConfig but get their own card
  // below. Split them out: the Advanced card edits everything EXCEPT the mic fields, the
  // mic card edits ONLY the mic fields. Both patch core.vad and apply() deep-merges the
  // partial patch, so the two disjoint subsets never clobber each other.
  const { mic_channel, mic_normalize, mic_highpass, ...vadValues } = rawVad;

  const buildPatch = (draft) => ({ core: { vad: draft } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(vadValues, buildPatch, patch);

  // Independent form for the server-side end-of-phrase confirmation chime
  const ackSchema = coreSchema.$defs ? coreSchema.$defs.AckConfig : null;
  const ackValues = catalog.core.values.ack || {};
  const buildAckPatch = (draft) => ({ core: { ack: draft } });
  const {
    draft: ackDraft, onChange: ackOnChange, dirty: ackDirty,
    saving: ackSaving, err: ackErr, save: ackSave,
  } = useStageForm(ackValues, buildAckPatch, patch);

  // Chime selector options for core.ack.sound_path: bundled files from assets/chimes,
  // shown by bare filename, plus an empty-valued "synthesized" default at the top.
  const ackOptionsFor = async () => {
    const { options } = await getChimes();
    return [
      { value: "", label: "Synthesized chime (default)" },
      ...options.map((p) => ({ value: p, label: p.split("/").pop() })),
    ];
  };

  // Mic input card: which device mic channel feeds the pipeline + the pre-STT
  // conditioning toggles. The fields live in VadConfig (core.vad.mic_channel /
  // mic_normalize / mic_highpass) — a LIVE reconfig, applied on the next utterance.
  // Build a mini-schema with just those props so they render in their own card.
  const micSchema = vadSchema && vadSchema.properties ? {
    type: "object",
    properties: {
      mic_channel: vadSchema.properties.mic_channel,
      mic_normalize: vadSchema.properties.mic_normalize,
      mic_highpass: vadSchema.properties.mic_highpass,
    },
  } : null;
  const micValues = { mic_channel, mic_normalize, mic_highpass };
  const buildMicPatch = (draft) => ({ core: { vad: draft } });
  const {
    draft: micDraft, onChange: micOnChange, dirty: micDirty,
    saving: micSaving, err: micErr, save: micSave,
  } = useStageForm(micValues, buildMicPatch, patch);

  const chime = useChimePreview();

  if (!vadSchema) return <div className="z-page"><div className="z-card"><div className="z-empty"><b>VAD</b>Schema unavailable.</div></div></div>;

  return <div className="z-page">
    <PageHeader title="VAD · Voice capture" crumb="Pipeline / Stage 01"
      desc="The speaker streams audio continuously and never signals end-of-phrase — we detect it server-side with the selected VAD. Tune sensitivity to pauses here." />
    {/* Two equal columns: VAD provider + end-pointing on the left, mic + chime cards stacked on the right */}
    <div className="z-cols even">
      <div className="z-grid">
        {vadCat && <ProviderCard cat="vad" sub="speech/no-speech classifier" />}
        <Card title="End-pointing thresholds" sub="when a phrase is considered started / finished"
          foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
          <SchemaForm schema={{ ...vadSchema, $defs: coreSchema.$defs }} root={{ ...vadSchema, $defs: coreSchema.$defs }} values={draft} onChange={onChange} skip={["mic_channel", "mic_normalize", "mic_highpass"]} />
        </Card>
      </div>
      {(micSchema || ackSchema) && <div className="z-grid">
        {micSchema && <Card title="Microphone input & conditioning" sub="which device mic stream feeds the pipeline · pre-STT conditioning"
          foot={<FormSaveBar dirty={micDirty} saving={micSaving} onSave={micSave} errors={errorLines(micErr)} />}>
          <SchemaForm schema={{ ...micSchema, $defs: coreSchema.$defs }} root={{ ...micSchema, $defs: coreSchema.$defs }} values={micDraft} onChange={micOnChange} skip={["mic_normalize", "mic_highpass"]} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>
            <SchemaForm schema={{ ...micSchema, $defs: coreSchema.$defs }} root={{ ...micSchema, $defs: coreSchema.$defs }} values={micDraft} onChange={micOnChange} skip={["mic_channel"]} />
          </div>
        </Card>}
        {ackSchema && <Card title="End-of-phrase chime" sub="confirmation played when your phrase ends"
          foot={<FormSaveBar dirty={ackDirty} saving={ackSaving} onSave={ackSave} errors={errorLines(ackErr)} />}>
          <SchemaForm schema={{ ...ackSchema, $defs: coreSchema.$defs }} root={{ ...ackSchema, $defs: coreSchema.$defs }} values={ackDraft} onChange={ackOnChange} optionsFor={ackOptionsFor}
            itemActionFor={(field) => (field === "sound_path" ? chime.play : null)} itemActionBusy={chime.busy} />
          {/* Per-item play buttons live in the dropdown; only surface preview errors here. */}
          {chime.msg && <div className="z-fh" style={{ marginTop: 6, ...(chime.msg.tone === "err" ? { color: "#b91c1c" } : {}) }}>{chime.msg.text}</div>}
        </Card>}
      </div>}
    </div>
  </div>;
}
