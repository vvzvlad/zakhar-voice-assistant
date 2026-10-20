// Logic page — the ordered list of recognition chains (core.chains). A chain
// pairs an STT provider with a handler (an llm-category provider: an LLM or the
// offline command engine). After the input stages the chains are tried in order
// on the same audio until one answers. The whole list is one draft saved as ONE
// patch of the full array, because PATCH /api/config replaces lists.
import React from "react";
import { PageHeader, FormPage, Select, Toggle, Field } from "../components/primitives.jsx";
import { Ic } from "../components/icons.jsx";
import { useAppData } from "../appData.jsx";
import { useStageForm } from "../useStageForm.js";
import { nav, navTab } from "../navStore.js";
import {
  chainsOf, providerOptions, canNarrow, startsFirst, lastEnabledIndex, handlerHasVocabulary,
  moveItem, newChain, settingsPageFor,
} from "../chains.js";

// Open the page that edits provider `id` of category `cat`, on that provider's tab.
function configure(cat, id) {
  const page = settingsPageFor(cat, id);
  if (page === cat) navTab(page, id);
  else nav(page);
}

function ProviderPick({ label, cat, value, options, onChange }) {
  return <div className="z-f">
    <div className="z-fl">
      <b>{label}</b>
      <button type="button" className="z-btn g sm" onClick={() => configure(cat, value)}>configure →</button>
    </div>
    <Select value={value} options={options} onChange={onChange} />
  </div>;
}

function ChainCard({ chain, index, count, catalog, chains, isLastEnabled, onUpdate, onMove, onDelete }) {
  const openEnded = isLastEnabled && handlerHasVocabulary(catalog, chain.handler);
  // A switched-off chain is not dimmed: everything on the card is a working control,
  // and the toggle shows the state.
  return <div className="z-card" data-testid="chain-card">
    {/* On a narrow screen the controls drop under the name. */}
    <div className="z-card-h" style={{ flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "1 1 180px", minWidth: 0 }}>
        <span className="mono" style={{ color: "var(--mut2)", fontSize: 12 }}>#{index + 1}</span>
        <div className="z-inp sm" style={{ flex: 1, maxWidth: 280 }}>
          <input aria-label="Chain name" value={chain.name} onChange={(e) => onUpdate({ name: e.target.value })} />
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <Toggle sm on={!!chain.enabled} onChange={(on) => onUpdate({ enabled: on })} />
        <button type="button" className="z-btn g sm" aria-label="Move up" disabled={index === 0} onClick={() => onMove(-1)}>↑</button>
        <button type="button" className="z-btn g sm" aria-label="Move down" disabled={index === count - 1} onClick={() => onMove(1)}>↓</button>
        <button type="button" className="z-btn d sm" onClick={onDelete}>Delete</button>
      </div>
    </div>
    <div className="z-card-b">
      <div className="z-row2">
        <ProviderPick label="STT" cat="stt" value={chain.stt} options={providerOptions(catalog, "stt")}
          onChange={(v) => onUpdate({ stt: v })} />
        <ProviderPick label="Handler" cat="llm" value={chain.handler} options={providerOptions(catalog, "llm")}
          onChange={(v) => onUpdate({ handler: v })} />
      </div>
      {canNarrow(catalog, chain) && <Field row label="Narrow STT to handler vocabulary" hint="Recognize only words the handler knows.">
        <Toggle on={!!chain.narrow} onChange={(on) => onUpdate({ narrow: on })} />
      </Field>}
      {/* The first enabled chain always starts at speech; later ones may start with it. */}
      {!startsFirst(chains, index) && <Field row label="Recognize in parallel"
        hint="Start this chain's STT together with the first chain. Handlers still run in order.">
        <Toggle on={!!chain.parallel} onChange={(on) => onUpdate({ parallel: on })} />
      </Field>}
      {openEnded && <div className="z-banner warn" style={{ margin: "12px 0" }}>
        <Ic n="restart" w={15} />
        <span>This is the last enabled chain and its handler only knows its vocabulary: phrases outside its vocabulary get «не понял».</span>
      </div>}
    </div>
  </div>;
}

export default function Chains() {
  const { catalog, config, patch } = useAppData();
  const buildPatch = (d) => ({ core: { chains: d.chains } });
  const form = useStageForm({ chains: chainsOf(config) }, buildPatch, patch);
  const list = form.draft.chains || [];
  const setList = (next) => form.onChange("chains", next);
  const last = lastEnabledIndex(list);
  // `narrow` stays as it is while its switch is hidden: the backend ignores it
  // for a pair that cannot narrow.
  const update = (i, fields) => setList(list.map((c, j) => (j === i ? { ...c, ...fields } : c)));

  return <FormPage form={form} narrow>
    <PageHeader title="Logic · Recognition chains"
      desc="Each chain is an STT plus a handler. Chains are tried in order on the same audio: the next one takes over when the handler declines, the STT returns nothing or fails, or the handler fails before running a tool."
      actions={<button className="z-btn p" onClick={() => setList([...list, newChain(list, catalog)])}><Ic n="add" w={14} />Add chain</button>} />
    <div className="z-grid">
      {list.length === 0 && <div className="z-card"><div className="z-empty"><b>No chains</b>Add a chain so recognized speech gets a handler.</div></div>}
      {list.map((c, i) => <ChainCard key={i} chain={c} index={i} count={list.length} catalog={catalog} chains={list}
        isLastEnabled={i === last}
        onUpdate={(fields) => update(i, fields)}
        onMove={(delta) => setList(moveItem(list, i, delta))}
        onDelete={() => setList(list.filter((_, j) => j !== i))} />)}
    </div>
  </FormPage>;
}
