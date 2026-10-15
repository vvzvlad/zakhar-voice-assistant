import React, { useState, useEffect, useRef } from "react";
import { Selector, PageHeader, FormSaveBar, Toggle } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getOptions, getChimes, playChime, streamTtsVoice } from "../api.js";
import { playAudioResponse } from "../streamAudio.js";
import { buildVoiceMarker } from "../voiceMarker.js";
import { pendingTab, clearTab } from "../navStore.js";
import { chainsOf, chainsUsing, moveItem, settingsPageFor, COMMANDS_PROVIDER } from "../chains.js";
import NluCatalogEditor from "../components/NluCatalogEditor.jsx";

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

// `q` (optional) is the server-side search string for remote-search fields;
// forwarded only when present so plain fields keep the legacy call shape.
const optionsLoader = (cat, providerId) => async (field, q) => {
  const r = await (q ? getOptions(cat, providerId, field, q) : getOptions(cat, providerId, field));
  return r.options;
};

function ReloadingBanner({ cat, label }) {
  // This stage's backend is being hot-reloaded (model load in flight) per the heartbeat.
  const { system } = useAppData();
  if (!(system?.reloading || []).includes(cat)) return null;
  return <div className="z-banner warn">
    <span className="z-spin" />
    <span><b>Applying…</b> loading the {label} backend — downloading the model on first use can take a while.</span>
  </div>;
}

// ── Generic provider stage with a selected provider (Wakeword / Accents) ──
function ProviderStage({ cat, title, crumb, desc }) {
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

  return <div className="z-page narrow">
    <PageHeader title={title} crumb={crumb} desc={desc} />
    <ReloadingBanner cat={cat} label={prov.label} />
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
            switch would be reconciled in place and keep the PREVIOUS provider's options.
            The key forces a remount so every DynamicSelect refetches. */}
        <SchemaForm key={selected} schema={prov.schema} values={draft} onChange={onChange}
          optionsFor={optionsLoader(cat, selected)} />
      </Card>
    </div>
  </div>;
}

// ── Provider settings without a selected provider (STT / LLM / TTS / Commands) ──
// Which provider runs is decided elsewhere (the Chains page, the TTS variant
// list), so these pages show one provider's settings at a time behind tabs.

// The tab to show: a provider requested by a "configure" link from another page,
// else `preferred` (the provider in use), else the first one.
function useProviderTab(cat, providers, preferred) {
  const ids = providers.map((p) => p.id);
  const [tab, setTab] = useState(() => {
    const requested = pendingTab(cat);
    if (ids.includes(requested)) return requested;
    return ids.includes(preferred) ? preferred : ids[0];
  });
  useEffect(() => { clearTab(cat); }, [cat]);
  // Fall back when the catalog no longer has the tab's provider.
  return [ids.includes(tab) ? tab : ids[0], setTab];
}

// Switching a tab only changes which provider's settings are shown; it never PATCHes.
function ProviderTabs({ providers, tab, onTab }) {
  const prov = providers.find((p) => p.id === tab);
  return <Selector label="Provider settings" options={providers.map((p) => p.id)} value={tab}
    onChange={onTab} caption={prov?.label} />;
}

// One provider's settings form. Saving patches only that provider's instance and
// never touches routing. Render it with key={provider id}: the remount drops the
// previous tab's draft and makes every DynamicSelect refetch for the new provider.
// `usage` is a line shown above the form; `children(draft, onChange)` renders
// extra cards bound to the same live draft.
function ProviderSettings({ cat, prov, usage, skip = [], children }) {
  const { patch } = useAppData();
  const buildPatch = (draft) => ({ [cat]: { instances: { [prov.id]: draft } } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(prov.values, buildPatch, patch);
  return <>
    <Card title={prov.label} foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
      {usage && <div className="z-fh" style={{ margin: 0, padding: "11px 0 2px" }}>{usage}</div>}
      <SchemaForm schema={prov.schema} values={draft} onChange={onChange}
        optionsFor={optionsLoader(cat, prov.id)} skip={skip} />
    </Card>
    {children && children(draft, onChange)}
  </>;
}

function usageLine(chains, field, id) {
  const names = chainsUsing(chains, field, id);
  return names.length ? `Used in chains: ${names.join(", ")}` : "Not used in any chain";
}

// STT and LLM: provider tabs; `field` is the chain field that names this
// category's provider. The offline command engine is excluded from the LLM tabs
// (it has its own Commands page).
function ChainProviderStage({ cat, field, title, crumb, desc }) {
  const { catalog, config } = useAppData();
  const chains = chainsOf(config);
  const category = catalog.categories.find((c) => c.id === cat);
  const providers = category.providers.filter((p) => settingsPageFor(cat, p.id) === cat);
  const [tab, setTab] = useProviderTab(cat, providers, chains.find((c) => c.enabled)?.[field]);
  const prov = providers.find((p) => p.id === tab);

  return <div className="z-page narrow">
    <PageHeader title={title} crumb={crumb} desc={desc} />
    {prov && <ReloadingBanner cat={cat} label={prov.label} />}
    {prov
      ? <>
        <ProviderTabs providers={providers} tab={tab} onTab={setTab} />
        <div className="z-grid">
          <ProviderSettings key={tab} cat={cat} prov={prov} usage={usageLine(chains, field, tab)} />
        </div>
      </>
      : <div className="z-card"><div className="z-empty"><b>No providers</b>The backend reports no providers for this stage.</div></div>}
  </div>;
}

// TTS variants: the ordered fallback list (tts.variants) — the next enabled
// variant takes over when a provider fails before the speaker gets any audio.
// Edited as one draft and saved as ONE patch of the whole array (PATCH replaces
// lists). "Configure" only switches the settings tab below.
function TtsVariantsCard({ variants, providers, onConfigure }) {
  const { patch } = useAppData();
  const buildPatch = (d) => ({ tts: { variants: d.variants } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm({ variants }, buildPatch, patch);
  const list = draft.variants || [];
  const set = (next) => onChange("variants", next);
  const unlisted = providers.filter((p) => !list.some((v) => v.provider === p.id));
  const labelOf = (id) => providers.find((p) => p.id === id)?.label;

  return <Card title="Variants" sub="tried in order until one starts speaking"
    foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
    {list.length === 0 && <div className="z-fh" style={{ padding: "11px 0" }}>Add at least one enabled variant — the server rejects a list without one.</div>}
    {list.map((v, i) => <div key={v.provider} className="z-f row" data-testid="tts-variant" style={{ opacity: v.enabled ? 1 : 0.55 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, minWidth: 0 }}>
        <span className="mono" style={{ color: "var(--mut2)", fontSize: 11 }}>{i + 1}</span>
        <b className="mono" style={{ fontSize: 12.5 }}>{v.provider}</b>
        {labelOf(v.provider) && <span className="z-fh" style={{ marginTop: 0 }}>{labelOf(v.provider)}</span>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <Toggle sm on={!!v.enabled} onChange={(on) => set(list.map((x, j) => (j === i ? { ...x, enabled: on } : x)))} />
        <button type="button" className="z-mini" aria-label="Move up" disabled={i === 0} onClick={() => set(moveItem(list, i, -1))}>↑</button>
        <button type="button" className="z-mini" aria-label="Move down" disabled={i === list.length - 1} onClick={() => set(moveItem(list, i, 1))}>↓</button>
        <button type="button" className="z-mini" onClick={() => onConfigure(v.provider)}>Configure</button>
        <button type="button" className="z-mini" onClick={() => set(list.filter((_, j) => j !== i))}>Remove</button>
      </div>
    </div>)}
    {unlisted.length > 0 && <div className="z-f row">
      <span className="z-fh" style={{ marginTop: 0 }}>Add a variant</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, justifyContent: "flex-end" }}>
        {unlisted.map((p) => <button type="button" key={p.id} className="z-mini"
          onClick={() => set([...list, { provider: p.id, enabled: true }])}>+ {p.id}</button>)}
      </div>
    </div>}
  </Card>;
}

export function Wakeword() {
  return <ProviderStage cat="wakeword" title="Wakeword · Verify" crumb="Pipeline / Stage 02"
    desc="Server-side wake-word verifier. Re-checks the captured phrase actually started with the wake word and rejects false triggers before STT." />;
}
export function STT() {
  return <ChainProviderStage cat="stt" field="stt" title="STT · Speech to text" crumb="Providers"
    desc="Settings of each speech recognizer: cloud Whisper or offline Vosk. Which one runs is set on the Chains page." />;
}
export function LLM() {
  return <ChainProviderStage cat="llm" field="handler" title="LLM · Reasoning & tools" crumb="Providers"
    desc="Settings of each chat model that generates the reply and calls smart-home tools over MCP. Which one runs is set on the Chains page." />;
}
export function Commands() {
  const { catalog, config } = useAppData();
  const llm = catalog.categories.find((c) => c.id === "llm");
  const prov = llm?.providers.find((p) => p.id === COMMANDS_PROVIDER);
  return <div className="z-page narrow">
    <PageHeader title="Commands · Offline command engine" crumb="Providers"
      desc="The offline command engine answers smart-home commands from its own vocabulary without an LLM. Spoken names and verbs come from the MCP tool catalog." />
    {prov
      ? <div className="z-grid">
        {/* `aliases`/`actions` are edited through the NluCatalogEditor below, so
            their raw textareas are hidden; every other field stays. */}
        <ProviderSettings cat="llm" prov={prov} usage={usageLine(chainsOf(config), "handler", prov.id)}
          skip={["aliases", "actions"]}>
          {(draft, onChange) => <NluCatalogEditor draft={draft} onChange={onChange} />}
        </ProviderSettings>
      </div>
      : <div className="z-card"><div className="z-empty"><b>Unavailable</b>The backend has no offline command engine.</div></div>}
  </div>;
}
export function Accents() {
  return <ProviderStage cat="stress" title="Accents · Stress placement" crumb="Pipeline / Stage 04"
    desc="Place Russian word stress on the reply text so TTS pronounces it correctly." />;
}
export function TTS() {
  const { catalog } = useAppData();
  const category = catalog.categories.find((c) => c.id === "tts");
  const providers = category.providers;
  const variants = category.variants || [];
  const [tab, setTab] = useProviderTab("tts", providers, variants.find((v) => v.enabled)?.provider);
  const prov = providers.find((p) => p.id === tab);

  return <div className="z-page narrow">
    <PageHeader title="TTS · Text to speech" crumb="Pipeline / Stage 05"
      desc="Synthesize the reply to audio served to the speakers. Variants are tried in order; settings are per provider." />
    {prov && <ReloadingBanner cat="tts" label={prov.label} />}
    <div className="z-grid">
      <TtsVariantsCard variants={variants} providers={providers} onConfigure={setTab} />
      {prov && <div>
        <ProviderTabs providers={providers} tab={tab} onTab={setTab} />
        <div className="z-grid">
          <ProviderSettings key={tab} cat="tts" prov={prov}>
            {(draft) => <>
              {/* Test the CURRENT (unsaved) draft settings with an in-browser playback. */}
              <VoiceTestCard provider={tab} settings={draft} />
              {/* Ready-to-copy prompt marker built from the same live draft. */}
              <VoiceMarkerCard provider={tab} settings={draft} />
            </>}
          </ProviderSettings>
        </div>
      </div>}
    </div>
  </div>;
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
    <SchemaForm key={selected} schema={prov.schema} values={draft} onChange={onChange} optionsFor={optionsLoader(cat, selected)} />
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
