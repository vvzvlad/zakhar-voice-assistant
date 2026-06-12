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
// Short human label, e.g. "yandex/zahar" or "fishaudio/s2-pro/3b5c9f" or "teratts".
export function voiceLabel(parsed) {
  if (!parsed) return "";
  const vals = Object.values(parsed.fields);
  return [parsed.provider, ...vals].join("/");
}
