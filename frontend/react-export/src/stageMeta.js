// Static stage presentation metadata (colors / order / labels / icons).
// RUAccent intentionally removed — the backend stage does not exist.
export const STAGE_COLOR = {
  vad: "#64748b",
  stt: "#0891b2",
  llm: "#4f46e5",
  tts: "#0d9488",
};
export const STAGE_ORDER = ["vad", "stt", "llm", "tts"];

// Pipeline stages shown in the dashboard service-map and the stage page sub-nav.
// `cat` links a stage to a catalog category (vad has none — it's a core section).
export const STAGES = [
  { key: "vad", name: "VAD", role: "Voice capture", cat: null },
  { key: "stt", name: "STT", role: "Speech → text", cat: "stt" },
  { key: "llm", name: "LLM", role: "Reasoning + tools", cat: "llm" },
  { key: "tts", name: "TTS", role: "Text → speech", cat: "tts" },
];
