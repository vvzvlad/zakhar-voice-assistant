import React, { useState, useEffect, useRef } from "react";
import { Selector, PageHeader, FormSaveBar } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getOptions, getChimes, playChime, testTtsVoice } from "../api.js";

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
  const audioRef = useRef(null); // currently playing Audio element, if any
  const urlRef = useRef(null);   // object URL backing it, revoked on cleanup

  const onText = (v) => {
    setText(v);
    try { localStorage.setItem(TTS_TEST_TEXT_KEY, v); } catch { /* privacy mode: keep in memory */ }
  };

  // Stop playback (if any), free the object URL and return to idle. Idempotent
  // (null refs make a repeat call a no-op), and the "ended"/"error" listeners in
  // test() only invoke it while their own Audio still owns the refs, so a stale
  // event from a superseded clip can never touch a newer playback.
  const cleanup = () => {
    if (audioRef.current) {
      try { audioRef.current.pause(); } catch { /* already unloaded — ignore */ }
      audioRef.current = null;
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
    setPhase(null);
  };

  const test = async () => {
    setPhase("synth");
    setMsg(null);
    try {
      const blob = await testTtsVoice(provider, settings, text);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      urlRef.current = url;
      // Guard by instance: a late event from a superseded clip must not touch the
      // refs, which by then belong to a newer Audio (Stop + quick re-Test race).
      audio.addEventListener("ended", () => { if (audioRef.current === audio) cleanup(); });
      audio.addEventListener("error", () => {
        if (audioRef.current !== audio) return;
        setMsg("Audio playback failed");
        cleanup();
      });
      await audio.play();
      // Playback started: the same button now acts as Stop until "ended" fires.
      setPhase("playing");
    } catch (e) {
      setMsg(e.message || "Voice test failed");
      cleanup();
    }
  };

  const stop = () => cleanup(); // pause + revoke + back to idle

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

// ── Generic provider stage (STT / LLM / TTS) ──────────────────────────────
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

  // `q` (optional) is the server-side search string for remote-search fields;
  // forwarded only when present so plain fields keep the legacy call shape.
  const optionsFor = async (field, q) => {
    const r = await (q ? getOptions(cat, selected, field, q) : getOptions(cat, selected, field));
    return r.options;
  };

  return <div className="z-page narrow">
    <PageHeader title={title} crumb={crumb} desc={desc} />
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
      {/* TTS only: test the CURRENT (unsaved) draft settings with an in-browser playback. */}
      {cat === "tts" && <VoiceTestCard provider={selected} settings={draft} />}
    </div>
  </div>;
}

export function STT() {
  return <ProviderStage cat="stt" title="STT · Speech to text" crumb="Pipeline / Stage 02"
    desc="Recognize the captured phrase. Cloud Whisper or offline Vosk." />;
}
export function LLM() {
  return <ProviderStage cat="llm" title="LLM · Reasoning & tools" crumb="Pipeline / Stage 03"
    desc="Generates the reply and calls smart-home tools over MCP. OpenAI-compatible chat completions." />;
}
export function TTS() {
  return <ProviderStage cat="tts" title="TTS · Text to speech" crumb="Pipeline / Stage 04"
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
  // VAD is a catalog category since R2 (webrtc provider: aggressiveness, auto_gain).
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
