// Pure helpers for the offline-NLU (simple-nlu) alias editor. These translate
// between the provider's two textarea fields (`aliases`, `actions`) and the live
// MCP tool catalog, so the panel can offer one alias input per discovered entity
// instead of asking the operator to hand-edit the raw textarea. No React here.

// Parse the `actions` textarea into a name→verbs map plus the lines we must
// preserve verbatim. Each line `name = verbs` maps `name.trim().toLowerCase()` to
// the RIGHT side trimmed verbatim (we keep the operator's exact verb string — case
// and spacing — since it is re-emitted as-is). Blank lines, `#` comments, and
// lines without `=` are pushed into `extraLines` verbatim so manual edits survive.
// Duplicate names: last wins (matching parseAliasText). NOTE: this is the editor's
// view of the field; the backend's parse_actions lowercases verbs and splits on
// commas — that divergence is intentional, the editor only round-trips text.
export function parseActions(actionsText) {
  const map = {};
  const extraLines = [];
  for (const rawLine of String(actionsText || "").split("\n")) {
    const line = rawLine;
    const trimmed = line.trim();
    const eq = trimmed.indexOf("=");
    if (!trimmed || trimmed.startsWith("#") || eq < 0) {
      extraLines.push(line);
      continue;
    }
    const name = trimmed.slice(0, eq).trim().toLowerCase();
    const verbs = trimmed.slice(eq + 1).trim();
    if (!name) {
      extraLines.push(line);
      continue;
    }
    map[name] = verbs; // duplicate name keys: last wins
  }
  return { map, extraLines };
}

// Extract the action NAMES (the token before `=`) from the `actions` textarea.
// Thin wrapper over parseActions: the keys of its map are the action names,
// lowercased + trimmed, in encounter order. Kept exported for callers (and tests)
// that only need membership.
export function parseActionNames(actionsText) {
  return Object.keys(parseActions(actionsText).map);
}

// Discover the enum slots that are alias targets, from the live tool catalog.
//
// `sources` is the panel's /api/tools shape: each source has `tools: [{name,
// description, parameters}]`. For every tool property that carries a non-empty
// `enum`, emit `{tool, slot, type, values}` — EXCEPT "state slots", where every
// enum value is also an action name (case-insensitive). Those are driven by the
// action verbs, not by aliases, so they are skipped. Slots without an enum
// (e.g. brightness/temperature numbers) are not alias targets and are omitted.
// Results are returned in stable encounter order (source → tool → property).
export function entitySlotsFromTools(sources, actionNames) {
  const actionSet = new Set((actionNames || []).map((n) => String(n).toLowerCase()));
  const out = [];
  for (const source of sources || []) {
    for (const tool of (source && source.tools) || []) {
      const props = (tool && tool.parameters && tool.parameters.properties) || {};
      for (const [slot, prop] of Object.entries(props)) {
        const values = prop && Array.isArray(prop.enum) ? prop.enum : [];
        if (values.length === 0) continue; // no enum -> not an alias target
        // A state slot is one whose every enum value is an action name; verbs
        // already fill it, so aliases would be redundant.
        const isStateSlot = values.every((v) => actionSet.has(String(v).toLowerCase()));
        if (isStateSlot) continue;
        out.push({ tool: tool.name, slot, type: prop.type, values });
      }
    }
  }
  return out;
}

// Discover EVERY enum slot in the live tool catalog, with NO action-name
// filtering. For every tool property that carries a non-empty `enum`, emit
// `{tool, slot, type, values}`. Missing `parameters`/`properties`/`enum` are
// tolerated. Results are returned in stable encounter order (source → tool →
// property). The caller classifies each slot as entity vs action via
// classifySlotKind; this is the raw, unfiltered superset of entitySlotsFromTools.
export function allEnumSlots(sources) {
  const out = [];
  for (const source of sources || []) {
    for (const tool of (source && source.tools) || []) {
      const props = (tool && tool.parameters && tool.parameters.properties) || {};
      for (const [slot, prop] of Object.entries(props)) {
        const values = prop && Array.isArray(prop.enum) ? prop.enum : [];
        if (values.length === 0) continue; // no enum -> not a mappable slot
        out.push({ tool: tool.name, slot, type: prop.type, values });
      }
    }
  }
  return out;
}

// Discover EVERY enum slot in the live tool catalog, carrying the OWNING source's
// id alongside the tool/slot. Same shape + traversal as allEnumSlots but adds
// `server` (the source id) so the catalog editor can group/filter by MCP source.
// For every tool property with a non-empty `enum`, emit
// `{server, tool, slot, type, values}` in stable encounter order.
export function enumSlotsWithSource(sources) {
  const out = [];
  for (const source of sources || []) {
    for (const tool of (source && source.tools) || []) {
      const props = (tool && tool.parameters && tool.parameters.properties) || {};
      for (const [slot, prop] of Object.entries(props)) {
        const values = prop && Array.isArray(prop.enum) ? prop.enum : [];
        if (values.length === 0) continue; // no enum -> not a mappable slot
        out.push({ server: source && source.id, tool: tool.name, slot, type: prop.type, values });
      }
    }
  }
  return out;
}

// Discover the NUMERIC slots: every REQUIRED tool property that has NO enum and a
// type in ("integer", "number", "string"). These are filled by a number spoken in
// the command itself (brightness/temperature) — there is no value dictionary, so
// they are shown as read-only info cards. A `string` type signals an off-capable
// number ("set X to 30" or "off"). Only properties listed in the tool's `required`
// array are emitted: an optional free-form arg is not a slot the operator must map.
// Emits `{server, tool, slot, type}` in stable encounter order.
const NUMBER_SLOT_TYPES = new Set(["integer", "number", "string"]);
export function numberSlotsFromTools(sources) {
  const out = [];
  for (const source of sources || []) {
    for (const tool of (source && source.tools) || []) {
      const params = (tool && tool.parameters) || {};
      const props = params.properties || {};
      const required = new Set(Array.isArray(params.required) ? params.required : []);
      for (const [slot, prop] of Object.entries(props)) {
        if (!required.has(slot)) continue; // only required args are mandatory slots
        const hasEnum = prop && Array.isArray(prop.enum) && prop.enum.length > 0;
        if (hasEnum) continue; // enum -> a dictionary slot, handled elsewhere
        if (!NUMBER_SLOT_TYPES.has(prop && prop.type)) continue;
        out.push({ server: source && source.id, tool: tool.name, slot, type: prop.type });
      }
    }
  }
  return out;
}

// Split a comma-separated phrase string into an array of trimmed, de-duplicated,
// non-empty words. The inverse of joinWords. Used by the chip input to turn the
// stored `aliases`/`actions` RHS text into removable tags. Each part is sanitized
// so it can never break the line-based `phrases = value` / `name = verbs` format on
// round-trip: any `=`, `\r`, or `\n` (which would mis-split the line or inject a new
// one) is replaced with a space, runs of whitespace are collapsed to one, and the
// result is trimmed — then empties are dropped and the existing de-dup is kept.
export function splitWords(s) {
  const out = [];
  const seen = new Set();
  for (const part of String(s || "").split(",")) {
    const w = part.replace(/[=\r\n]+/g, " ").replace(/\s+/g, " ").trim();
    if (!w || seen.has(w)) continue;
    seen.add(w);
    out.push(w);
  }
  return out;
}

// Join an array of words back into the comma-separated RHS text the field stores.
export function joinWords(arr) {
  return (arr || []).join(", ");
}

// Group raw enum slots (from enumSlotsWithSource) into the catalog editor's display
// groups, classifying each slot via `kindOf`. Two grouping rules:
//   - ACTION-kind slots merge by their VALUE SET: `groupKey = sorted(values).join(" ")`.
//     Identical action sets across tools collapse into one group; `tools`/`servers`
//     accumulate the union (rendered as "used in: a, b"). This dedups e.g. set_light
//     and set_switch both exposing {on, off}.
//   - ENTITY-kind slots never merge: `groupKey = "${tool}.${slot}"` (one group per
//     tool's slot), even if two tools share a value set.
// `kindOf(slot)` receives the raw slot and returns "entity" | "action"; the caller
// keys overrides by GROUP key, so the resolver must already factor that in.
// Emits `[{key, kind, slotName, tools:[], servers:[], values:[]}]` in first-seen order.
export function groupSlots(slots, kindOf) {
  const groups = [];
  const byKey = new Map();
  for (const s of slots || []) {
    const kind = kindOf(s);
    const key =
      kind === "action"
        ? [...s.values].sort().join(" ")
        : `${s.tool}.${s.slot}`;
    let g = byKey.get(key);
    if (!g) {
      g = { key, kind, slotName: s.slot, tools: [], servers: [], values: [] };
      byKey.set(key, g);
      groups.push(g);
    }
    if (!g.tools.includes(s.tool)) g.tools.push(s.tool);
    if (s.server != null && !g.servers.includes(s.server)) g.servers.push(s.server);
    // Union the values (action groups merge identical sets; this also keeps any
    // out-of-order duplicates from re-adding a value).
    for (const v of s.values) if (!g.values.includes(v)) g.values.push(v);
  }
  return groups;
}

// Action slot names: these conventionally hold a state/command verb (their enum
// values are actions, not entity ids), so they map to verbs in `actions` rather
// than phrases in `aliases`. Used as a name-based hint by classifySlotKind.
const ACTION_SLOT_NAMES = new Set([
  "state", "action", "mode", "command", "operation", "switch", "power", "toggle",
]);

// Decide whether an enum slot maps to an ENTITY (device/scene id → Russian
// phrases in `aliases`) or an ACTION (state/command verb → Russian verbs in
// `actions`). Priority:
//   1. an explicit operator `override` ("entity"|"action") always wins;
//   2. a slot whose NAME is a known action slot name (state/action/...) → action;
//   3. else a slot whose every enum value is also an action name → action
//      (the same ⊆actionNames heuristic the backend's _train uses for state slots);
//   4. otherwise → entity.
export function classifySlotKind(slotName, slotValues, actionNames, override) {
  if (override === "entity" || override === "action") return override;
  if (ACTION_SLOT_NAMES.has(String(slotName).toLowerCase())) return "action";
  const actionSet = (actionNames || []).map((a) => String(a).toLowerCase());
  if (
    slotValues.length &&
    slotValues.every((v) => actionSet.includes(String(v).toLowerCase()))
  ) {
    return "action";
  }
  return "entity";
}

// Split the `aliases` textarea into a value→phrases map plus the lines we must
// preserve verbatim. We only manage the BARE form `phrases = value`, i.e. NOT the
// explicit `tool.slot:value` form (the part before any `:` has no dot, or the part
// after `:` is empty — mirroring the server's parse_aliases). Every other line —
// blank, `#` comment, explicit `tool.slot:value`, or anything unparseable — is
// pushed into `extraLines` so manual/advanced edits survive a round-trip.
export function parseAliasText(aliasesText) {
  const map = {};
  const extraLines = [];
  for (const rawLine of String(aliasesText || "").split("\n")) {
    const line = rawLine;
    const trimmed = line.trim();
    const eq = trimmed.indexOf("=");
    // Keep blanks, comments, and lines without `=` as-is.
    if (!trimmed || trimmed.startsWith("#") || eq < 0) {
      extraLines.push(line);
      continue;
    }
    const phrases = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim();
    // Explicit `tool.slot:value` form: a colon whose left part contains a dot and
    // whose right part is non-empty (mirrors the server's parse_aliases). Preserve
    // it verbatim. A stray colon without a dot is a bare value (server-consistent).
    const colon = value.indexOf(":");
    const isExplicit = colon > 0 && value.slice(0, colon).includes(".") && value.slice(colon + 1).trim() !== "";
    if (!value || isExplicit) {
      extraLines.push(line);
      continue;
    }
    map[value] = phrases; // duplicate value keys: last wins
  }
  return { map, extraLines };
}

// Rebuild the `aliases` textarea from the discovered entities, the current
// value→phrases map, and the preserved extra lines. For each unique enum VALUE
// across `entities` (in discovered order), emit `${phrases} = ${value}` when a
// non-empty trimmed phrase exists for it. Managed lines come first, then the
// preserved `extraLines` verbatim. Joined with newlines.
//
// `knownCatalogValues` (optional Set) lists ALL enum values in the live catalog,
// across entity AND action slots. In the safety-net loop that re-emits map entries
// not covered by `entities`, a value present in this set is SKIPPED: such a value
// is a real catalog value that currently belongs to an ACTION slot (so its verbs
// live in `actions`, not here) — re-emitting it would duplicate it into `aliases`.
// Only genuinely out-of-catalog manual aliases are preserved. Omitting the arg (or
// passing an empty set) keeps the original behavior — nothing is suppressed.
export function serializeAliasText(entities, valueToPhrases, extraLines, knownCatalogValues) {
  const known = knownCatalogValues || new Set();
  const lines = [];
  const seen = new Set();
  for (const entity of entities || []) {
    for (const value of (entity && entity.values) || []) {
      if (seen.has(value)) continue; // one line per unique value
      seen.add(value);
      const phrases = (valueToPhrases && valueToPhrases[value]) || "";
      if (phrases.trim() === "") continue; // skip values the user left blank
      lines.push(`${phrases} = ${value}`);
    }
  }
  // Preserve managed bare aliases whose value is NOT in the discovered entity list
  // (MCP offline at load, or a value the server resolves but the schema doesn't
  // enumerate). Without this they'd be silently dropped on the next edit. But a
  // value that IS a catalog value (just classified into an action slot) must NOT
  // be re-emitted here — only out-of-catalog manual aliases are re-emitted.
  for (const [value, phrases] of Object.entries(valueToPhrases || {})) {
    if (seen.has(value)) continue;
    seen.add(value);
    if (known.has(value)) continue; // in-catalog action-slot value -> not an alias
    if (String(phrases).trim() === "") continue;
    lines.push(`${phrases} = ${value}`);
  }
  for (const line of extraLines || []) lines.push(line);
  return lines.join("\n");
}

// Rebuild the `actions` textarea from the discovered action slots, the current
// name→verbs map, and the preserved extra lines. Symmetric to serializeAliasText:
// for each unique action enum VALUE across `actionSlots` (in discovered order),
// emit `${value} = ${verbs}` when a non-empty trimmed verb string exists for it.
// Then append any remaining `nameToVerbs` entries not yet emitted (custom action
// names the operator added by hand, or actions whose slot is currently hidden),
// then the preserved `extraLines` verbatim. Joined with newlines.
export function serializeActions(actionSlots, nameToVerbs, extraLines) {
  const lines = [];
  const seen = new Set();
  for (const slot of actionSlots || []) {
    for (const value of (slot && slot.values) || []) {
      if (seen.has(value)) continue; // one line per unique value
      seen.add(value);
      const verbs = (nameToVerbs && nameToVerbs[value]) || "";
      if (verbs.trim() === "") continue; // skip values the user left blank
      lines.push(`${value} = ${verbs}`);
    }
  }
  // Preserve custom action names not present in any discovered action slot.
  for (const [name, verbs] of Object.entries(nameToVerbs || {})) {
    if (seen.has(name)) continue;
    seen.add(name);
    if (String(verbs).trim() === "") continue;
    lines.push(`${name} = ${verbs}`);
  }
  for (const line of extraLines || []) lines.push(line);
  return lines.join("\n");
}
