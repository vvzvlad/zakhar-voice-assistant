// Navigation config + a tiny module-level navigate store.
// `nav(id)` is callable from any module; App registers the real navigate fn.
export const NAV = [
  { grp: "Overview", items: [["dashboard", "Dashboard"]] },
  { grp: "Pipeline", items: [["vad", "VAD / Capture"], ["stt", "STT"], ["llm", "LLM"], ["tts", "TTS"]] },
  { grp: "Integrations", items: [["mcp", "MCP Servers"], ["prompt", "System Prompt"], ["devices", "Devices"]] },
  { grp: "Operations", items: [["log", "Request Log"], ["system", "System"]] },
];
export const TITLES = Object.fromEntries(NAV.flatMap((g) => g.items));
