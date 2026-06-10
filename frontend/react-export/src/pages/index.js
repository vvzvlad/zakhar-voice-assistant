import Dashboard from "./dashboard.jsx";
import { VAD, STT, LLM, TTS } from "./stages.jsx";
import Log from "./log.jsx";
import { MCP, Prompt, Devices } from "./integrations.jsx";
import Device from "./device.jsx";

// RUAccent intentionally omitted — the backend stage does not exist.
export const PAGES = {
  dashboard: Dashboard,
  vad: VAD, stt: STT, llm: LLM, tts: TTS,
  mcp: MCP, prompt: Prompt, devices: Devices,
  log: Log, device: Device,
};
