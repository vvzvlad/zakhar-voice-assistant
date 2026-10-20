// Pure helpers for recognition chains (config.core.chains) and TTS variants
// (the tts catalog category's `variants`), shared by the Logic, Speech, STT/LLM,
// Commands and Dashboard pages and kept here so they can be unit-tested.
//
// A chain is {name, enabled, stt, handler, narrow, parallel}: `stt` is an stt
// provider id, `handler` an llm-category provider id. Chains run in list order
// on the same audio until one answers.
import { voiceField, voiceName } from "./voiceMarker.js";

export const chainsOf = (config) => config?.core?.chains || [];

const providersOf = (catalog, cat) =>
  catalog?.categories?.find((c) => c.id === cat)?.providers || [];

export const providerIds = (catalog, cat) => providersOf(catalog, cat).map((p) => p.id);

// Dropdown options for a category's providers: the id is stored, the catalog label shown.
export const providerOptions = (catalog, cat) =>
  providersOf(catalog, cat).map((p) => ({ value: p.id, label: p.label }));

// Capability flags the catalog carries on each provider entry.
const sttSupportsVocabulary = (catalog, id) =>
  !!providersOf(catalog, "stt").find((p) => p.id === id)?.supports_vocabulary;
export const handlerHasVocabulary = (catalog, id) =>
  !!providersOf(catalog, "llm").find((p) => p.id === id)?.has_vocabulary;

// STT can be narrowed only to the vocabulary of a handler that has one, and only
// when that STT supports a vocabulary; the backend ignores `narrow` otherwise.
export const canNarrow = (catalog, chain) =>
  sttSupportsVocabulary(catalog, chain.stt) && handlerHasVocabulary(catalog, chain.handler);

// The first enabled chain always starts its STT at speech, so `parallel` means
// nothing for a chain with no enabled chain before it.
export const startsFirst = (chains, i) => !chains.slice(0, i).some((c) => c.enabled);

export function lastEnabledIndex(chains) {
  for (let i = chains.length - 1; i >= 0; i--) if (chains[i].enabled) return i;
  return -1;
}

// Names of the chains whose `field` ("stt" | "handler") points at provider `id`;
// a disabled chain is marked "(off)".
export const chainsUsing = (chains, field, id) =>
  chains.filter((c) => c[field] === id).map((c) => (c.enabled ? c.name : `${c.name} (off)`));

// Provider ids the enabled chains run in `field` ("stt" | "handler").
export const usedByChains = (chains, field) => chains.filter((c) => c.enabled).map((c) => c[field]);

// Swap item i with its neighbour i+delta; out-of-range moves return the list as is.
export function moveItem(list, i, delta) {
  const j = i + delta;
  if (j < 0 || j >= list.length) return list;
  const next = list.slice();
  [next[i], next[j]] = [next[j], next[i]];
  return next;
}

// A new enabled chain with a name not yet in the list, the first provider of
// each category and the backend's defaults (narrow when the pair can).
export function newChain(chains, catalog) {
  const names = new Set(chains.map((c) => c.name));
  let n = chains.length + 1;
  while (names.has(`chain-${n}`)) n += 1;
  return {
    name: `chain-${n}`,
    enabled: true,
    stt: providerIds(catalog, "stt")[0] || "",
    handler: providerIds(catalog, "llm")[0] || "",
    narrow: true,
    parallel: false,
  };
}

// The offline command engine is an llm-category provider, but its settings and
// vocabulary live on the Commands page; every other provider on its category's
// Providers page.
export const COMMANDS_PROVIDER = "simple-nlu";
export const settingsPageFor = (cat, id) =>
  (cat === "llm" && id === COMMANDS_PROVIDER ? "commands" : cat);

export const ttsVariantsOf = (catalog) =>
  catalog?.categories?.find((c) => c.id === "tts")?.variants || [];

// TTS providers in the order of the variants (the fallback order), then the
// providers no variant names, in catalog order.
export function byVariantOrder(providers, variants) {
  const listed = variants.map((v) => providers.find((p) => p.id === v.provider)).filter(Boolean);
  return [...listed, ...providers.filter((p) => !listed.includes(p))];
}

// The voice a TTS provider speaks with, by name, read from its settings the way
// a prompt voice marker names it; "" when the settings name no voice.
export function voiceOf(values = {}) {
  const voice = { fields: values };
  return voiceField(voice) ? voiceName(voice, values) : "";
}

// Dashboard rows, one per enabled chain: its share of the served requests
// (served / sum of served over every chain in per_chain; null when nothing was
// served) and its average time.
export function chainRows(chains, perChain) {
  const pc = perChain || {};
  const sum = Object.values(pc).reduce((a, m) => a + ((m && m.served) || 0), 0);
  return chains.filter((c) => c.enabled).map((c) => {
    const m = pc[c.name] || {};
    return {
      name: c.name,
      stt: c.stt,
      handler: c.handler,
      share: sum > 0 ? (m.served || 0) / sum : null,
      avg_ms: m.avg_ms ?? null,
    };
  });
}
