import Dashboard from "./dashboard.jsx";
import { VAD, STT, LLM, Commands, TTS } from "./stages.jsx";
import { Capture, Speech } from "./pipeline.jsx";
import Chains from "./chains.jsx";
import Log from "./log.jsx";
import { MCP } from "./tools.jsx";
import { Prompt, SystemPrompt } from "./prompt.jsx";
import { Devices } from "./devices.jsx";
import SystemPage from "./system.jsx";

export const PAGES = {
  dashboard: Dashboard,
  capture: Capture, logic: Chains, speech: Speech,
  vad: VAD, stt: STT, llm: LLM, "system-prompt": SystemPrompt, prompt: Prompt, commands: Commands, tts: TTS, tools: MCP,
  devices: Devices, log: Log, system: SystemPage,
};
