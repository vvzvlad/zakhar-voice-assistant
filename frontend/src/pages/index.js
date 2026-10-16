import Dashboard from "./dashboard.jsx";
import { VAD, Wakeword, STT, LLM, Commands, Accents, TTS } from "./stages.jsx";
import Chains from "./chains.jsx";
import Log from "./log.jsx";
import { MCP, Prompt, Devices } from "./integrations.jsx";
import SystemPage from "./system.jsx";

export const PAGES = {
  dashboard: Dashboard,
  vad: VAD, wakeword: Wakeword, chains: Chains, stress: Accents, tts: TTS,
  stt: STT, llm: LLM, commands: Commands,
  mcp: MCP, prompt: Prompt, devices: Devices,
  log: Log, system: SystemPage,
};
