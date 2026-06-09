// VAD sensitivity presets, extracted from pages/stages.jsx (R-FE-1,
// behaviour-preserving) so the matcher can be unit-tested.
//
// Frontend-only presets that map to silence_ms / min_speech_ms macros. These are
// NOT a backend field — just a convenience that nudges two real fields together.
export const VAD_PRESETS = {
  Fast: { silence_ms: 500, min_speech_ms: 150 },
  Balanced: { silence_ms: 800, min_speech_ms: 200 },
  Patient: { silence_ms: 1200, min_speech_ms: 300 },
};

// Return the preset name whose silence_ms+min_speech_ms both equal the draft's,
// else null. Extra fields on `v` are ignored; only those two must match exactly.
export function matchPreset(v) {
  for (const [name, p] of Object.entries(VAD_PRESETS)) {
    if (p.silence_ms === v.silence_ms && p.min_speech_ms === v.min_speech_ms) return name;
  }
  return null;
}
