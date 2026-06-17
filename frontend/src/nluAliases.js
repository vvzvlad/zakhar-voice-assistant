// Pure helpers for the offline-NLU (simple-nlu) alias editor. These translate
// between the provider's two textarea fields (`aliases`, `actions`) and the live
// MCP tool catalog, so the panel can offer one alias input per discovered entity
// instead of asking the operator to hand-edit the raw textarea. No React here.

// Extract the action NAMES (the token before `=`) from the `actions` textarea.
// Names are lowercased + trimmed. Blank lines, `#` comments, and lines without
// `=` are skipped. Returns the names in encounter order (duplicates kept as-is —
// callers only test membership, so dedup is unnecessary).
export function parseActionNames(actionsText) {
  const names = [];
  for (const rawLine of String(actionsText || "").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const name = line.slice(0, eq).trim().toLowerCase();
    if (name) names.push(name);
  }
  return names;
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
export function serializeAliasText(entities, valueToPhrases, extraLines) {
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
  // Preserve managed bare aliases whose value is NOT in the discovered catalog
  // (MCP offline at load, or a value the server resolves but the schema doesn't
  // enumerate). Without this they'd be silently dropped on the next edit.
  for (const [value, phrases] of Object.entries(valueToPhrases || {})) {
    if (seen.has(value)) continue;
    seen.add(value);
    if (String(phrases).trim() === "") continue;
    lines.push(`${phrases} = ${value}`);
  }
  for (const line of extraLines || []) lines.push(line);
  return lines.join("\n");
}
