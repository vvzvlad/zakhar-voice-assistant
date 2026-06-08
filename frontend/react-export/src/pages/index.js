import Dashboard from "./dashboard.jsx";
import { VAD, STT, LLM, RUAccent, TTS } from "./stages.jsx";
import Log from "./log.jsx";
import { MCP, Prompt, Context, Devices } from "./integrations.jsx";
import Network from "./network.jsx";

export const PAGES = {
  dashboard: Dashboard,
  vad: VAD, stt: STT, llm: LLM, ruaccent: RUAccent, tts: TTS,
  mcp: MCP, prompt: Prompt, context: Context, devices: Devices,
  log: Log, network: Network,
};
