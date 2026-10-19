// Pipeline pages Capture and Speech (Logic is chains.jsx): they only choose
// what runs and switch stages on and off; every setting lives on the Providers
// page an item's "configure →" link opens. One Save per page sends the whole
// page as one PATCH (usePageForm, FormPage).
import React from "react";
import { Card, FormPage, PageHeader, Seg, Toggle } from "../components/primitives.jsx";
import { useAppData } from "../appData.jsx";
import { usePageForm } from "../usePageForm.js";
import { nav, navTab } from "../navStore.js";
import { moveItem, voiceOf } from "../chains.js";

// A stage's on/off switch: `enabled` of its selected provider (wakeword, stress, followup).
function enabledPart(category) {
  const prov = category.providers.find((p) => p.id === category.selected) || category.providers[0];
  return {
    prov,
    part: {
      key: category.id,
      values: { enabled: prov.values.enabled },
      toPatch: (d) => ({ [category.id]: { instances: { [prov.id]: d } } }),
    },
  };
}

const descriptionOf = (schema, field) => schema?.properties?.[field]?.description;

// One pipeline item: its name with a "configure →" link, a hint and the control
// (beside the name with `row`, under it otherwise).
function Item({ label, hint, onConfigure, row, children }) {
  return <div className={"z-f" + (row ? " row" : "")}>
    <div className={row ? "z-fmeta" : ""}>
      <div className="z-fl" style={{ justifyContent: "flex-start" }}>
        <b>{label}</b>
        <button type="button" className="z-mini" onClick={onConfigure}>configure →</button>
      </div>
      {hint && <div className="z-fh">{hint}</div>}
    </div>
    {/* flex: an inline <span> toggle has no box of its own outside a flex container. */}
    <div className={row ? "z-fctl" : ""} style={row ? { display: "flex", alignItems: "center" } : {}}>{children}</div>
  </div>;
}

export function Capture() {
  const { catalog, patch } = useAppData();
  const vad = catalog.categories.find((c) => c.id === "vad");
  const wake = enabledPart(catalog.categories.find((c) => c.id === "wakeword"));
  const ackSchema = catalog.core.schema.$defs?.AckConfig;
  const form = usePageForm([
    { key: "vad", values: { selected: vad.selected }, toPatch: (d) => ({ vad: { selected: d.selected } }) },
    wake.part,
    { key: "ack", values: { enabled: catalog.core.values.ack?.enabled }, toPatch: (d) => ({ core: { ack: d } }) },
  ], patch);
  const engine = form.draftOf("vad").selected;
  const engineLabel = vad.providers.find((p) => p.id === engine)?.label;
  const choose = vad.providers.length > 1;

  return <FormPage form={form} narrow>
    <PageHeader title="Capture"
      desc="What runs on the speaker's audio before recognition: the VAD engine that detects where your phrase ends, the wake-word check and the confirmation chime. Their settings are on Providers → VAD." />
    <Card>
      {/* Leaving drops an unsaved engine choice, so the link opens the saved engine. */}
      <Item label="VAD engine" hint={choose ? engineLabel : "Detects where your phrase ends."} row={!choose}
        onConfigure={() => navTab("vad", vad.selected)}>
        {choose
          ? <Seg full options={vad.providers.map((p) => p.id)} value={engine} onChange={(id) => form.onChangeOf("vad")("selected", id)} />
          : <span className="z-fh" style={{ marginTop: 0 }}>{engineLabel}</span>}
      </Item>
      <Item row label="Wake-word verify" hint={descriptionOf(wake.prov.schema, "enabled")} onConfigure={() => nav("vad")}>
        <Toggle on={!!form.draftOf(wake.part.key).enabled} onChange={(on) => form.onChangeOf(wake.part.key)("enabled", on)} />
      </Item>
      <Item row label="End-of-phrase chime" hint={descriptionOf(ackSchema, "enabled")} onConfigure={() => nav("vad")}>
        <Toggle on={!!form.draftOf("ack").enabled} onChange={(on) => form.onChangeOf("ack")("enabled", on)} />
      </Item>
    </Card>
  </FormPage>;
}

// TTS variants: the ordered fallback list (tts.variants) — the next enabled
// variant takes over when a provider fails before the speaker gets any audio.
// Each row names the voice the provider speaks with; "configure →" opens its
// tab on Providers → TTS.
function TtsVariantsCard({ list, providers, onChange }) {
  const unlisted = providers.filter((p) => !list.some((v) => v.provider === p.id));

  return <Card title="Variants" sub="tried in order until one starts speaking">
    {list.length === 0 && <div className="z-fh" style={{ padding: "11px 0" }}>Add at least one enabled variant — the server rejects a list without one.</div>}
    {list.map((v, i) => {
      const prov = providers.find((p) => p.id === v.provider);
      const about = [prov?.label, prov && voiceOf(prov.values)].filter(Boolean).join(" · ");
      // A switched-off variant dims its number, name and voice; its toggle and buttons stay as is.
      return <div key={v.provider} className="z-f row" data-testid="tts-variant">
        <div className={v.enabled ? "" : "z-dim"} style={{ display: "flex", alignItems: "baseline", gap: 10, minWidth: 0 }}>
          <span className="mono" style={{ color: "var(--mut2)", fontSize: 11 }}>{i + 1}</span>
          <b className="mono" style={{ fontSize: 12.5 }}>{v.provider}</b>
          {about && <span className="z-fh" style={{ marginTop: 0 }}>{about}</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Toggle sm on={!!v.enabled} onChange={(on) => onChange(list.map((x, j) => (j === i ? { ...x, enabled: on } : x)))} />
          <button type="button" className="z-mini" aria-label="Move up" disabled={i === 0} onClick={() => onChange(moveItem(list, i, -1))}>↑</button>
          <button type="button" className="z-mini" aria-label="Move down" disabled={i === list.length - 1} onClick={() => onChange(moveItem(list, i, 1))}>↓</button>
          <button type="button" className="z-mini" onClick={() => navTab("tts", v.provider)}>configure →</button>
          <button type="button" className="z-mini" onClick={() => onChange(list.filter((_, j) => j !== i))}>Remove</button>
        </div>
      </div>;
    })}
    {unlisted.length > 0 && <div className="z-f row">
      <span className="z-fh" style={{ marginTop: 0 }}>Add a variant</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, justifyContent: "flex-end" }}>
        {unlisted.map((p) => <button type="button" key={p.id} className="z-mini"
          onClick={() => onChange([...list, { provider: p.id, enabled: true }])}>+ {p.id}</button>)}
      </div>
    </div>}
  </Card>;
}

export function Speech() {
  const { catalog, patch } = useAppData();
  const tts = catalog.categories.find((c) => c.id === "tts");
  const stress = enabledPart(catalog.categories.find((c) => c.id === "stress"));
  const followup = enabledPart(catalog.categories.find((c) => c.id === "followup"));
  // PATCH replaces lists, so the variants go as the whole array.
  const form = usePageForm([
    { key: "variants", values: { variants: tts.variants || [] }, toPatch: (d) => ({ tts: { variants: d.variants } }) },
    stress.part,
    followup.part,
  ], patch);

  return <FormPage form={form} narrow>
    <PageHeader title="Speech"
      desc="How the reply is spoken: the first TTS variant that starts speaking voices it, after stress placement marks Russian word stress on the text. Voices and synthesizer settings are on Providers → TTS." />
    <div className="z-grid">
      <TtsVariantsCard list={form.draftOf("variants").variants} providers={tts.providers}
        onChange={(next) => form.onChangeOf("variants")("variants", next)} />
      <Card>
        <Item row label="Stress placement" hint={descriptionOf(stress.prov.schema, "enabled")} onConfigure={() => nav("tts")}>
          <Toggle on={!!form.draftOf(stress.part.key).enabled} onChange={(on) => form.onChangeOf(stress.part.key)("enabled", on)} />
        </Item>
        <Item row label="Follow-up" hint={descriptionOf(followup.prov.schema, "enabled")} onConfigure={() => nav("tts")}>
          <Toggle on={!!form.draftOf(followup.part.key).enabled} onChange={(on) => form.onChangeOf(followup.part.key)("enabled", on)} />
        </Item>
      </Card>
    </div>
  </FormPage>;
}
