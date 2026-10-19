// Navigation config + a tiny module-level navigate store.
// `nav(id)` is callable from any module; App registers the real navigate fn.
// Item tuple: [id, label, children?] where children is an optional array of
// [id, label] tuples rendered as always-visible sub-items under the parent.
export const NAV = [
  { grp: "Overview", items: [["dashboard", "Dashboard"]] },
  // Pipeline follows the request path: the devices that feed it, then the pages
  // that choose what runs and switch stages on and off; every provider and engine
  // setting lives on the Providers pages.
  { grp: "Pipeline", items: [["devices", "Devices"], ["capture", "Capture"], ["logic", "Logic"], ["speech", "Speech"]] },
  { grp: "Providers", items: [["vad", "VAD"], ["stt", "STT"], ["llm", "LLM", [["prompt", "System Prompt"]]], ["commands", "Commands"], ["tts", "TTS"], ["tools", "Tools"]] },
  { grp: "Operations", items: [["log", "Request Log"], ["devlog", "Device Log"], ["system", "System"]] },
];
// Flatten children into TITLES so child ids stay valid routes/page titles.
export const TITLES = Object.fromEntries(
  NAV.flatMap((g) => g.items.flatMap(([id, label, kids]) => [[id, label], ...(kids || [])])),
);
