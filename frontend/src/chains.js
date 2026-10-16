// Pure helpers for recognition chains (config.core.chains) and TTS variants
// (the tts catalog category's `variants`), shared by the Chains, STT/LLM/TTS,
// Commands and Dashboard pages and kept here so they can be unit-tested.
//
// A chain is {name, enabled, stt, handler, narrow, parallel}: `stt` is an stt
// provider id, `handler` an llm-category provider id. Chains run in list order
// on the same audio until one answers.

export const chainsOf = (config) => config?.core?.chains || [];

const providersOf = (catalog, cat) =>
  catalog?.categories?.find((c) => c.id === cat)?.providers || [];

export const providerIds = (catalog, cat) => providersOf(catalog, cat).map((p) => p.id);

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

// The backend turns on strict mode for a vocabulary-bound handler by itself when
// an enabled chain follows it (it declines instead of guessing); not a setting.
export const isStrict = (catalog, chains, i) =>
  handlerHasVocabulary(catalog, chains[i].handler) && chains.slice(i + 1).some((c) => c.enabled);

export function lastEnabledIndex(chains) {
  for (let i = chains.length - 1; i >= 0; i--) if (chains[i].enabled) return i;
  return -1;
}

// Names of the chains whose `field` ("stt" | "handler") points at provider `id`.
export const chainsUsing = (chains, field, id) =>
  chains.filter((c) => c[field] === id).map((c) => c.name);

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
// vocabulary live on the Commands page; every other provider on its category page.
export const COMMANDS_PROVIDER = "simple-nlu";
export const settingsPageFor = (cat, id) =>
  (cat === "llm" && id === COMMANDS_PROVIDER ? "commands" : cat);

export const ttsVariantsOf = (catalog) =>
  catalog?.categories?.find((c) => c.id === "tts")?.variants || [];

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
