// Static stage presentation metadata (colors / order / labels / icons).
export const STAGE_COLOR = {
  vad: "#64748b",
  stt: "#0891b2",
  llm: "#4f46e5",
  stress: "#9333ea",
  tts: "#0d9488",
};
export const STAGE_ORDER = ["vad", "stt", "llm", "stress", "tts"];

// Accent for the "early filler" marker (an announcement spoken mid-run, not a
// pipeline stage). Distinct from the STAGE_COLOR palette so it reads as an event.
export const FILLER_COLOR = "#f59e0b";

// Pipeline stages shown in the dashboard service-map and the stage page sub-nav.
// `cat` links a stage to a catalog category. All stages are catalog
// categories now; the VAD page additionally edits core.vad sections
// (end-pointing thresholds, mic conditioning).
export const STAGES = [
  { key: "vad", name: "VAD", role: "Voice capture", cat: "vad" },
  { key: "stt", name: "STT", role: "Speech → text", cat: "stt" },
  { key: "llm", name: "LLM", role: "Reasoning + tools", cat: "llm" },
  { key: "stress", name: "Accents", role: "Stress placement", cat: "stress" },
  { key: "tts", name: "TTS", role: "Text → speech", cat: "tts" },
];
