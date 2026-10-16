// Static stage presentation metadata (colors / order / labels / icons).
export const STAGE_COLOR = {
  vad: "#64748b",
  wakeword: "#d97706",
  stt: "#0891b2",
  llm: "#4f46e5",
  stress: "#9333ea",
  tts: "#0d9488",
};
export const STAGE_ORDER = ["vad", "wakeword", "stt", "llm", "stress", "tts"];

// Accent for the "early filler" marker (an announcement spoken mid-run, not a
// pipeline stage). Distinct from the STAGE_COLOR palette so it reads as an event.
export const FILLER_COLOR = "#f59e0b";

// Pipeline stages by catalog category (`cat`); the top bar names a category whose
// backend is reloading by its stage `name`.
export const STAGES = [
  { key: "vad", name: "VAD", cat: "vad" },
  { key: "wakeword", name: "Wakeword", cat: "wakeword" },
  { key: "stt", name: "STT", cat: "stt" },
  { key: "llm", name: "LLM", cat: "llm" },
  { key: "stress", name: "Accents", cat: "stress" },
  { key: "tts", name: "TTS", cat: "tts" },
];
