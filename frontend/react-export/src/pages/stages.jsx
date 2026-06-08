import React from "react";
import { nav } from "../navStore.js";
import { Selector, PageHeader, FormSaveBar, Seg, Field } from "../components/primitives.jsx";
import SchemaForm, { schemaNeedsRestart } from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getOptions } from "../api.js";
import { STAGES } from "../stageMeta.js";

function Card({ title, sub, children, foot }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
    <div className="z-card-b">{children}</div>
    {foot}
  </div>;
}

// Sub-nav across pipeline stages (status dots are not wired to live health yet).
function StageNav({ active }) {
  return <div className="z-snav">
    {STAGES.map((s, i) => <React.Fragment key={s.key}>
      <div className={"z-snav-i" + (s.key === active ? " on" : "")} onClick={() => nav(s.key)}>
        <span className="z-dot ok" />{s.name}
      </div>
      {i < STAGES.length - 1 && <span className="z-snav-sep">›</span>}
    </React.Fragment>)}
  </div>;
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

  const optionsFor = async (field) => {
    const r = await getOptions(cat, selected, field);
    return r.options;
  };

  const restart = schemaNeedsRestart(prov.schema);

  return <div className="z-page">
    <PageHeader title={title} crumb={crumb} desc={desc} />
    <StageNav active={cat} />
    <Selector
      label="Provider"
      options={category.providers.map((p) => p.id)}
      value={selected}
      onChange={switchProvider}
      caption={prov.label}
    />
    <Card title={prov.label} foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} restart={restart} errors={errorLines(err)} />}>
      <SchemaForm schema={prov.schema} values={draft} onChange={onChange} optionsFor={optionsFor} />
    </Card>
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

// ── VAD (core sub-section) ────────────────────────────────────────────────
// Frontend-only presets that map to silence_ms / min_speech_ms macros. These are
// NOT a backend field — just a convenience that nudges two real fields together.
const VAD_PRESETS = {
  Fast: { silence_ms: 500, min_speech_ms: 150 },
  Balanced: { silence_ms: 800, min_speech_ms: 200 },
  Patient: { silence_ms: 1200, min_speech_ms: 300 },
};

function matchPreset(v) {
  for (const [name, p] of Object.entries(VAD_PRESETS)) {
    if (p.silence_ms === v.silence_ms && p.min_speech_ms === v.min_speech_ms) return name;
  }
  return null;
}

export function VAD() {
  const { catalog, patch } = useAppData();
  const coreSchema = catalog.core.schema;
  // vad prop is a $ref to $defs.VadConfig — resolve to the object schema.
  const vadSchema = coreSchema.$defs ? coreSchema.$defs.VadConfig : null;
  const vadValues = catalog.core.values.vad || {};

  const buildPatch = (draft) => ({ core: { vad: draft } });
  const { draft, onChange, dirty, saving, err, save } = useStageForm(vadValues, buildPatch, patch);

  const preset = matchPreset(draft);
  const applyPreset = (name) => {
    const p = VAD_PRESETS[name];
    onChange("silence_ms", p.silence_ms);
    onChange("min_speech_ms", p.min_speech_ms);
  };

  if (!vadSchema) return <div className="z-page"><div className="z-card"><div className="z-empty"><b>VAD</b>Схема недоступна.</div></div></div>;

  return <div className="z-page">
    <PageHeader title="VAD · Voice capture" crumb="Pipeline / Stage 01"
      desc="The speaker streams audio continuously and never signals end-of-phrase — we detect it with WebRTC VAD. Tune sensitivity to pauses here." />
    <StageNav active="vad" />
    <Card title="Sensitivity preset" sub="quick tuning · frontend macro">
      <Field label="Pause sensitivity" hint="Presets nudge trailing-silence and min-speech together. Fine-tune below.">
        <Seg full options={Object.keys(VAD_PRESETS)} value={preset || ""} onChange={applyPreset} />
      </Field>
    </Card>
    <div style={{ height: 16 }} />
    <Card title="Advanced parameters" foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} restart errors={errorLines(err)} />}>
      <SchemaForm schema={{ ...vadSchema, $defs: coreSchema.$defs }} root={{ ...vadSchema, $defs: coreSchema.$defs }} values={draft} onChange={onChange} />
    </Card>
  </div>;
}
