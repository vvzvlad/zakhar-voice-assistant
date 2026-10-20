import Dashboard from "./dashboard.jsx";
import { VAD, STT, LLM, Commands, TTS } from "./stages.jsx";
import { Capture, Speech } from "./pipeline.jsx";
import Chains from "./chains.jsx";
import Log from "./log.jsx";
import { MCP, Prompt, Devices } from "./integrations.jsx";
import SystemPage from "./system.jsx";

export const PAGES = {
  dashboard: Dashboard,
  capture: Capture, logic: Chains, speech: Speech,
  vad: VAD, stt: STT, llm: LLM, prompt: Prompt, commands: Commands, tts: TTS, tools: MCP,
  devices: Devices, log: Log, system: SystemPage,
};
