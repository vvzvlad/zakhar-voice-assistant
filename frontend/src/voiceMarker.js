// Mirror of src/voice_marker.py: parse the single-line preferred-voice marker
// "<<<<<VOICE provider=<id> field=value ...>>>>>" from profile text. Returns
// { provider, fields } or null. Display-only; the server re-parses on activation.
const VOICE_RE = /<<<<<VOICE\b([^\n]*?)>>>>>/;
export function parseVoiceMarker(text) {
  if (!text) return null;
  const m = VOICE_RE.exec(text);
  if (!m) return null;
  const fields = {};
  for (const tok of m[1].trim().split(/\s+/)) {
    if (!tok.includes("=")) continue;
    const i = tok.indexOf("=");
    const k = tok.slice(0, i).trim();
    if (k) fields[k] = tok.slice(i + 1).trim();
  }
  const provider = fields.provider; delete fields.provider;
  return provider ? { provider, fields } : null;
}
// Short human label, e.g. "yandex/zahar" or "fishaudio/s2-pro/3b5c9f" or "piper".
export function voiceLabel(parsed) {
  if (!parsed) return "";
  const vals = Object.values(parsed.fields);
  return [parsed.provider, ...vals].join("/");
}

// Mirror of src/voice_marker.py SECRET_FIELDS: credentials are never written
// into a prompt marker.
export const SECRET_FIELDS = new Set(["api_key", "token", "password"]);

// Build the single-line prompt marker from a TTS provider id + its current
// settings, e.g. "<<<<<VOICE provider=yandex voice=zahar>>>>>". Secrets are
// excluded; empty/nullish values are skipped; values containing whitespace or a
// ">" are skipped because the space-separated single-line format (closed by
// ">>>>>") cannot round-trip them. Provider-only (no emittable fields) yields
// the bare form "<<<<<VOICE provider=piper>>>>>".
export function buildVoiceMarker(provider, settings = {}) {
  if (!provider) return "";
  const parts = [`provider=${provider}`];
  for (const [k, v] of Object.entries(settings || {})) {
    if (SECRET_FIELDS.has(k)) continue;
    if (k.endsWith("_label")) continue; // hidden display-label companion, not voice identity
    if (v === null || v === undefined) continue;
    const s = String(v);
    // Skip values that cannot survive the single-line, space-separated, ">>>>>"-
    // terminated format: empty, whitespace-bearing, or containing ">".
    if (s === "" || /\s/.test(s) || s.includes(">")) continue;
    parts.push(`${k}=${s}`);
  }
  return `<<<<<VOICE ${parts.join(" ")}>>>>>`;
}
