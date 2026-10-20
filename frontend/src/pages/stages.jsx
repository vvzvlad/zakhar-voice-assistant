// Providers pages: every provider and engine setting. Which provider runs and
// which stages are on is chosen on the Pipeline pages (pipeline.jsx, chains.jsx).
// Each page has ONE Save, in the bar at the bottom (FormPage), that sends every
// dirty card and tab as one PATCH (usePageForm); switching a provider tab never
// PATCHes.
import React, { useState, useEffect, useRef } from "react";
import { Selector, Card, FormPage, PageHeader } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { usePageForm } from "../usePageForm.js";
import { getOptions, getChimes, playChime, streamTtsVoice } from "../api.js";
import { playAudioResponse } from "../streamAudio.js";
import { buildVoiceMarker } from "../voiceMarker.js";
import { pendingTab, clearTab } from "../navStore.js";
import { chainsOf, chainsUsing, usedByChains, byVariantOrder, settingsPageFor, COMMANDS_PROVIDER } from "../chains.js";
import NluCatalogEditor from "../components/NluCatalogEditor.jsx";

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
      // Footer: the button on the left, the synthesis error right-aligned.
      <div className="z-foot">
        {/* One button, three phases: idle → Synthesizing (disabled) → Stop (danger style). */}
        <button
          className={playing ? "z-btn d" : "z-btn p"}
          disabled={playing ? false : phase === "synth" || !text.trim()}
          onClick={playing ? stop : test}
        >
          {playing ? "Stop" : phase === "synth" ? "Synthesizing…" : "Test voice"}
        </button>
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

// Page-form parts, one per provider of `cat`: its settings, saved into its instance.
const providerParts = (cat, providers) => providers.map((p) => ({
  key: `${cat}:${p.id}`,
  values: p.values,
  toPatch: (d) => ({ [cat]: { instances: { [p.id]: d } } }),
}));

// The selected provider of a stage switched on and off on a Pipeline page
// (wakeword, stress, followup) and its page-form part: its settings without `enabled`.
function selectedSettings(category) {
  const prov = category.providers.find((p) => p.id === category.selected) || category.providers[0];
  const { enabled: _enabled, ...values } = prov.values;
  return {
    prov,
    part: { key: `${category.id}:${prov.id}`, values, toPatch: (d) => ({ [category.id]: { instances: { [prov.id]: d } } }) },
  };
}

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

// Switching a tab only changes which provider's settings are shown; it never
// PATCHes. Tabs are named by the providers' labels; a tab holding unsaved edits
// carries a dot, a provider the pipeline runs (`inUse`, ids) an "in use" marker.
// One provider gets no tabs.
function ProviderTabs({ cat, providers, tab, onTab, form, inUse }) {
  if (providers.length < 2) return null;
  return <Selector label="Provider settings" options={providers.map((p) => ({ value: p.id, label: p.label }))} value={tab}
    onChange={onTab} inUse={inUse}
    marked={providers.filter((p) => form.isDirty(`${cat}:${p.id}`)).map((p) => p.id)} />;
}

// One provider's settings form, bound to its page-form part. Render it with
// key={provider id}: the remount makes every DynamicSelect refetch for the new
// provider, while the draft lives in the page form and survives it.
// `usage` is a line shown above the form; `children(draft, onChange)` renders
// extra cards bound to the same live draft.
function ProviderSettings({ cat, prov, form, usage, children }) {
  const key = `${cat}:${prov.id}`;
  const draft = form.draftOf(key);
  const onChange = form.onChangeOf(key);
  return <>
    <Card title={prov.label}>
      {usage && <div className="z-fh" style={{ margin: 0, padding: "11px 0 2px" }}>{usage}</div>}
      <SchemaForm schema={prov.schema} values={draft} onChange={onChange} optionsFor={optionsLoader(cat, prov.id)} />
    </Card>
    {children && children(draft, onChange)}
  </>;
}

function usageLine(chains, field, id) {
  const names = chainsUsing(chains, field, id);
  return names.length ? `Used in chains: ${names.join(", ")}` : "Not used in any chain";
}
// Where a VAD engine and a TTS provider are put to use: on Capture and Speech.
const vadUsage = (selected, id) => (selected === id ? "Selected on Capture" : "Not selected");
function ttsUsage(variants, id) {
  const i = variants.findIndex((v) => v.provider === id);
  return i < 0 ? "Not in variants" : `Variant ${i + 1} on Speech${variants[i].enabled ? "" : " (off)"}`;
}

// STT and LLM: provider tabs; `field` is the chain field that names this
// category's provider. The offline command engine is excluded from the LLM tabs
// (it has its own Commands page).
function ChainProviderStage({ cat, field, title, desc }) {
  const { catalog, config, patch } = useAppData();
  const chains = chainsOf(config);
  const category = catalog.categories.find((c) => c.id === cat);
  const providers = category.providers.filter((p) => settingsPageFor(cat, p.id) === cat);
  const [tab, setTab] = useProviderTab(cat, providers, chains.find((c) => c.enabled)?.[field]);
  const form = usePageForm(providerParts(cat, providers), patch);
  const prov = providers.find((p) => p.id === tab);

  return <FormPage form={form} narrow>
    <PageHeader title={title} desc={desc} />
    {prov && <ReloadingBanner cat={cat} label={prov.label} />}
    {prov
      ? <>
        <ProviderTabs cat={cat} providers={providers} tab={tab} onTab={setTab} form={form} inUse={usedByChains(chains, field)} />
        <div className="z-grid">
          <ProviderSettings key={tab} cat={cat} prov={prov} form={form} usage={usageLine(chains, field, tab)} />
        </div>
      </>
      : <div className="z-card"><div className="z-empty"><b>No providers</b>The backend reports no providers for this stage.</div></div>}
  </FormPage>;
}

export function STT() {
  return <ChainProviderStage cat="stt" field="stt" title="STT · Speech to text"
    desc="Settings of each speech recognizer: cloud Whisper or offline Vosk. Which one runs is set on the Logic page." />;
}
export function LLM() {
  return <ChainProviderStage cat="llm" field="handler" title="LLM · Reasoning & tools"
    desc="Settings of each chat model that generates the reply and calls smart-home tools over MCP. Which one runs is set on the Logic page." />;
}
export function Commands() {
  const { catalog, config, patch } = useAppData();
  const llm = catalog.categories.find((c) => c.id === "llm");
  const prov = llm?.providers.find((p) => p.id === COMMANDS_PROVIDER);
  // The command catalog is the core.commands section; the engine's own settings are
  // its llm instance. Both save with the page's one Save.
  const { $defs } = catalog.core.schema;
  const commandsSchema = { ...$defs.CommandsConfig, $defs };
  const form = usePageForm([
    { key: "commands", values: catalog.core.values.commands, toPatch: (d) => ({ core: { commands: d } }) },
    ...(prov ? providerParts("llm", [prov]) : []),
  ], patch);
  // The engine settings start collapsed in the same disclosure as the forms' "Advanced", and
  // mount only while open.
  const [open, setOpen] = useState(false);
  const key = `llm:${COMMANDS_PROVIDER}`;
  return <FormPage form={form} narrow>
    <PageHeader title="Commands · Offline command engine"
      desc="The offline command engine answers smart-home commands from its own vocabulary without an LLM. Spoken names and verbs come from the MCP tool catalog." />
    {prov
      ? <div className="z-grid">
        <NluCatalogEditor draft={form.draftOf("commands")} onChange={form.onChangeOf("commands")} />
        <div className="z-card"><div className="z-card-b">
          <details onToggle={(e) => setOpen(e.currentTarget.open)}>
            <summary className="z-disclosure">Engine settings</summary>
            {open && <>
              <div className="z-fh" style={{ margin: 0, padding: "0 0 2px" }}>{usageLine(chainsOf(config), "handler", prov.id)}</div>
              {/* `aliases`/`actions` are edited through the vocabulary editor above,
                  so their raw textareas are hidden; the rest of the catalog stays. */}
              <SchemaForm schema={commandsSchema} values={form.draftOf("commands")} onChange={form.onChangeOf("commands")}
                skip={["aliases", "actions"]} />
              <SchemaForm schema={prov.schema} values={form.draftOf(key)} onChange={form.onChangeOf(key)}
                optionsFor={optionsLoader("llm", prov.id)} />
            </>}
          </details>
        </div></div>
      </div>
      : <div className="z-card"><div className="z-empty"><b>Unavailable</b>The backend has no offline command engine.</div></div>}
  </FormPage>;
}
export function TTS() {
  const { catalog, patch } = useAppData();
  const category = catalog.categories.find((c) => c.id === "tts");
  const variants = category.variants || [];
  // Tabs follow the Speech page's variant order.
  const providers = byVariantOrder(category.providers, variants);
  const stress = selectedSettings(catalog.categories.find((c) => c.id === "stress"));
  const followup = selectedSettings(catalog.categories.find((c) => c.id === "followup"));
  const [tab, setTab] = useProviderTab("tts", providers, variants.find((v) => v.enabled)?.provider);
  const form = usePageForm([...providerParts("tts", providers), stress.part, followup.part], patch);
  const prov = providers.find((p) => p.id === tab);

  return <FormPage form={form} narrow>
    <PageHeader title="TTS · Text to speech"
      desc="Settings of each speech synthesizer, of stress placement on the reply text and of listening for a follow-up after it. Which synthesizers run, and in what order, is set on the Speech page." />
    {prov && <ReloadingBanner cat="tts" label={prov.label} />}
    <ReloadingBanner cat="stress" label={stress.prov.label} />
    <ReloadingBanner cat="followup" label={followup.prov.label} />
    <div className="z-grid">
      {prov && <div id="card-tts">
        <ProviderTabs cat="tts" providers={providers} tab={tab} onTab={setTab} form={form}
          inUse={variants.filter((v) => v.enabled).map((v) => v.provider)} />
        <div className="z-grid">
          <ProviderSettings key={tab} cat="tts" prov={prov} form={form} usage={ttsUsage(variants, tab)}>
            {(draft) => <>
              {/* Test the CURRENT (unsaved) draft settings with an in-browser playback. */}
              <VoiceTestCard provider={tab} settings={draft} />
              {/* Ready-to-copy prompt marker built from the same live draft. */}
              <VoiceMarkerCard provider={tab} settings={draft} />
            </>}
          </ProviderSettings>
        </div>
      </div>}
      <Card id="card-stress" title="Stress placement" sub={stress.prov.label}>
        <SchemaForm schema={stress.prov.schema} values={form.draftOf(stress.part.key)} onChange={form.onChangeOf(stress.part.key)}
          optionsFor={optionsLoader("stress", stress.prov.id)} skip={["enabled"]} />
      </Card>
      <Card id="card-followup" title="Follow-up" sub={followup.prov.label}>
        <SchemaForm schema={followup.prov.schema} values={form.draftOf(followup.part.key)} onChange={form.onChangeOf(followup.part.key)}
          optionsFor={optionsLoader("followup", followup.prov.id)} skip={["enabled"]} />
      </Card>
    </div>
  </FormPage>;
}

// ── VAD (catalog category + core sub-sections + wake-word verify) ─────────

export function VAD() {
  const { catalog, patch } = useAppData();
  // VAD is a catalog category since R2 (webrtc provider: aggressiveness).
  // Tolerate an older backend without it — render the core.vad cards only.
  const vadCat = catalog.categories.find((c) => c.id === "vad");
  const engines = vadCat ? vadCat.providers : [];
  const wake = selectedSettings(catalog.categories.find((c) => c.id === "wakeword"));
  const coreSchema = catalog.core.schema;
  // vad prop is a $ref to $defs.VadConfig — resolve to the object schema.
  const vadSchema = coreSchema.$defs ? coreSchema.$defs.VadConfig : null;
  const ackSchema = coreSchema.$defs ? coreSchema.$defs.AckConfig : null;
  // mic_channel / mic_normalize / mic_highpass live in VadConfig but get their own card,
  // so core.vad is split into two disjoint parts; both patch core.vad and the merge
  // of the two keeps them apart.
  const { mic_channel, mic_normalize, mic_highpass, ...vadValues } = catalog.core.values.vad || {};
  // The chime's `enabled` is switched on the Capture page.
  const { enabled: _ackEnabled, ...ackValues } = catalog.core.values.ack || {};
  const [tab, setTab] = useProviderTab("vad", engines, vadCat?.selected);
  const form = usePageForm([
    { key: "endpointing", values: vadValues, toPatch: (d) => ({ core: { vad: d } }) },
    { key: "mic", values: { mic_channel, mic_normalize, mic_highpass }, toPatch: (d) => ({ core: { vad: d } }) },
    { key: "ack", values: ackValues, toPatch: (d) => ({ core: { ack: d } }) },
    ...providerParts("vad", engines),
    wake.part,
  ], patch);
  const engine = engines.find((p) => p.id === tab);

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

  const chime = useChimePreview();

  if (!vadSchema) return <div className="z-page"><div className="z-card"><div className="z-empty"><b>VAD</b>Schema unavailable.</div></div></div>;

  return <FormPage form={form}>
    <PageHeader title="VAD · Voice capture"
      desc="The speaker streams audio continuously and never signals end-of-phrase — we detect it server-side with the VAD engine chosen on the Capture page. Tune sensitivity to pauses here." />
    <ReloadingBanner cat="wakeword" label={wake.prov.label} />
    {/* Two equal columns: VAD engine + end-pointing on the left, mic + chime + wake-word cards stacked on the right */}
    <div className="z-cols even">
      <div className="z-grid">
        {engine && <div id="card-vad">
          <ProviderTabs cat="vad" providers={engines} tab={tab} onTab={setTab} form={form} inUse={[vadCat.selected]} />
          <ProviderSettings key={tab} cat="vad" prov={engine} form={form} usage={vadUsage(vadCat.selected, tab)} />
        </div>}
        <Card title="End-pointing thresholds" sub="when a phrase is considered started / finished">
          <SchemaForm schema={{ ...vadSchema, $defs: coreSchema.$defs }} root={{ ...vadSchema, $defs: coreSchema.$defs }}
            values={form.draftOf("endpointing")} onChange={form.onChangeOf("endpointing")} skip={["mic_channel", "mic_normalize", "mic_highpass"]} />
        </Card>
      </div>
      <div className="z-grid">
        {micSchema && <Card title="Microphone input & conditioning" sub="which device mic stream feeds the pipeline · pre-STT conditioning">
          {/* The conditioning toggles are advanced: they fold full-width under the channel selector. */}
          <SchemaForm schema={{ ...micSchema, $defs: coreSchema.$defs }} root={{ ...micSchema, $defs: coreSchema.$defs }}
            values={form.draftOf("mic")} onChange={form.onChangeOf("mic")} />
        </Card>}
        {ackSchema && <Card id="card-ack" title="End-of-phrase chime" sub="confirmation played when your phrase ends">
          <SchemaForm schema={{ ...ackSchema, $defs: coreSchema.$defs }} root={{ ...ackSchema, $defs: coreSchema.$defs }}
            values={form.draftOf("ack")} onChange={form.onChangeOf("ack")} optionsFor={ackOptionsFor} skip={["enabled"]}
            itemActionFor={(field) => (field === "sound_path" ? chime.play : null)} itemActionBusy={chime.busy} />
          {/* Per-item play buttons live in the dropdown; only surface preview errors here. */}
          {chime.msg && <div className="z-fh" style={{ marginTop: 6, ...(chime.msg.tone === "err" ? { color: "#b91c1c" } : {}) }}>{chime.msg.text}</div>}
        </Card>}
        <Card id="card-wakeword" title="Wake-word verify" sub={wake.prov.label}>
          <SchemaForm schema={wake.prov.schema} values={form.draftOf(wake.part.key)} onChange={form.onChangeOf(wake.part.key)}
            optionsFor={optionsLoader("wakeword", wake.prov.id)} skip={["enabled"]} />
        </Card>
      </div>
    </div>
  </FormPage>;
}
