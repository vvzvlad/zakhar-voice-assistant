// Navigation config + a tiny module-level navigate store.
// `nav(id)` is callable from any module; App registers the real navigate fn.
// Item tuple: [id, label, children?] where children is an optional array of
// [id, label] tuples rendered as always-visible sub-items under the parent.
export const NAV = [
  { grp: "Overview", items: [["dashboard", "Dashboard"]] },
  { grp: "Pipeline", items: [["devices", "Devices"], ["vad", "VAD / Capture"], ["stt", "STT"], ["llm", "LLM", [["mcp", "MCP Servers"], ["prompt", "System Prompt"]]], ["tts", "TTS"]] },
  { grp: "Operations", items: [["log", "Request Log"], ["system", "System"]] },
];
// Flatten children into TITLES so child ids stay valid routes/page titles.
export const TITLES = Object.fromEntries(
  NAV.flatMap((g) => g.items.flatMap(([id, label, kids]) => [[id, label], ...(kids || [])])),
);
