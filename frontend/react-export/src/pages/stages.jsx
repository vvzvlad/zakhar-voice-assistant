import React from "react";
import { Selector, PageHeader, FormSaveBar } from "../components/primitives.jsx";
import SchemaForm from "../components/SchemaForm.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm, errorLines } from "../useStageForm.js";
import { getOptions, getChimes } from "../api.js";

function Card({ title, sub, children, foot }) {
  return <div className="z-card">
    {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
    <div className="z-card-b">{children}</div>
    {foot}
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

  return <div className="z-page">
    <PageHeader title={title} crumb={crumb} desc={desc} />
    <Selector
      label="Provider"
      options={category.providers.map((p) => p.id)}
      value={selected}
      onChange={switchProvider}
      caption={prov.label}
    />
    <Card title={prov.label} foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
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

export function VAD() {
  const { catalog, patch } = useAppData();
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

  if (!vadSchema) return <div className="z-page"><div className="z-card"><div className="z-empty"><b>VAD</b>Schema unavailable.</div></div></div>;

  return <div className="z-page">
    <PageHeader title="VAD · Voice capture" crumb="Pipeline / Stage 01"
      desc="The speaker streams audio continuously and never signals end-of-phrase — we detect it with WebRTC VAD. Tune sensitivity to pauses here." />
    <Card title="Advanced parameters" foot={<FormSaveBar dirty={dirty} saving={saving} onSave={save} errors={errorLines(err)} />}>
      <SchemaForm schema={{ ...vadSchema, $defs: coreSchema.$defs }} root={{ ...vadSchema, $defs: coreSchema.$defs }} values={draft} onChange={onChange} skip={["mic_channel", "mic_normalize", "mic_highpass"]} />
    </Card>
    {micSchema && <>
      <div style={{ height: 16 }} />
      <Card title="Microphone channel & conditioning" sub="which device mic stream feeds the pipeline · pre-STT conditioning">
        <SchemaForm schema={{ ...micSchema, $defs: coreSchema.$defs }} root={{ ...micSchema, $defs: coreSchema.$defs }} values={micDraft} onChange={micOnChange} skip={["mic_normalize", "mic_highpass"]} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>
          <SchemaForm schema={{ ...micSchema, $defs: coreSchema.$defs }} root={{ ...micSchema, $defs: coreSchema.$defs }} values={micDraft} onChange={micOnChange} skip={["mic_channel"]} />
        </div>
        <FormSaveBar dirty={micDirty} saving={micSaving} onSave={micSave} errors={errorLines(micErr)} />
      </Card>
    </>}
    {ackSchema && <>
      <div style={{ height: 16 }} />
      <Card title="End-of-phrase chime" sub="confirmation played when your phrase ends">
        <SchemaForm schema={{ ...ackSchema, $defs: coreSchema.$defs }} root={{ ...ackSchema, $defs: coreSchema.$defs }} values={ackDraft} onChange={ackOnChange} optionsFor={ackOptionsFor} />
        <FormSaveBar dirty={ackDirty} saving={ackSaving} onSave={ackSave} errors={errorLines(ackErr)} />
      </Card>
    </>}
  </div>;
}
