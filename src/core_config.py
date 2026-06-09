"""Core (non-provider) configuration sections.

These have no provider choice, no runtime stage factory and no multiplicity, so
they are plain pydantic models (not providers). They still feed the panel the same
way — via model_json_schema() — but stay out of the REGISTRY.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ContextConfig(BaseModel):
    max_turns: int = Field(5, ge=1)
    ttl_seconds: int = Field(300, ge=0)


class AudioConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8200
    ttl: int = 300
    public_base_url: str = ""


class VadConfig(BaseModel):
    aggressiveness: int = Field(2, ge=0, le=3)
    silence_ms: int = 800
    min_speech_ms: int = 200
    max_utterance_ms: int = 15000
    no_speech_timeout_ms: int = 8000
    trim_start_ms: int = Field(
        200,
        ge=0,
        description="Drop this many ms off the start of the captured sample before STT (cuts the wake-word tail / button-press click). 0 disables.",
    )


class MicConfig(BaseModel):
    # Which Voice PE mic channel feeds the whole server pipeline (capture + VAD + STT).
    # 0 = processed (XMOS AGC output); 1 = less-processed (XMOS noise-suppression
    # output) — cleaner but quieter. Read live, so panel changes apply next utterance.
    channel: Literal[0, 1] = Field(0, json_schema_extra={"widget": "select"})
    # Linear input gain applied to the selected channel's PCM before VAD/STT. Boosts
    # the quieter less-processed channel (and helps the VAD detect speech).
    gain: float = Field(1.0, ge=1.0, le=8.0, json_schema_extra={"widget": "slider"})


class NetworkConfig(BaseModel):
    external_proxy: str = ""


class OpenWeatherMapConfig(BaseModel):
    api_key: str = ""
    city: str = "Moscow"
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", json_schema_extra={"widget": "textarea"})


class McpServerConfig(BaseModel):
    # One external MCP server entry. `name` is unique and also the ToolHub source id
    # (so it matches the /api/tools source id). `prompt` is appended to the system
    # prompt to describe this server's tools to the model.
    name: str
    url: str = ""
    token: str = ""
    transport: Literal["auto", "streamable_http", "sse"] = "auto"
    prompt: str = ""


class CalendarConfig(BaseModel):
    # CalDAV account for the built-in calendar MCP. `calendar` selects which calendar
    # to use by name; empty means the first calendar on the account.
    url: str = ""
    username: str = ""
    password: str = ""
    calendar: str = ""
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", json_schema_extra={"widget": "textarea"})


class DeviceConfig(BaseModel):
    name: str
    host: str
    psk: str


class EsphomeConfig(BaseModel):
    port: int = 6053


class PromptConfig(BaseModel):
    system_prompt_path: str = "data/system_prompt.md"


class RunsConfig(BaseModel):
    # Observability: persist every pipeline run to SQLite (data/runs.db). Disable
    # to skip recording entirely; retention_days=0 keeps rows forever.
    enabled: bool = True
    retention_days: int = Field(30, ge=0)
    # Rolling window of recent utterance audio (the exact PCM sent to STT), kept in
    # runs.db so operators can download/play it from the log to diagnose mis-triggers.
    store_audio: bool = True
    audio_keep: int = Field(100, ge=1)  # how many most-recent utterances keep their audio


class RemindersConfig(BaseModel):
    # One-shot voice reminders (data/reminders.db). Disable to drop the subsystem.
    enabled: bool = True


class CaptureConfig(BaseModel):
    # Raw audio capture: when enabled, every finalized utterance's post-frontend PCM
    # (what the Voice PE streams over the Native API) is saved as a 16 kHz / mono /
    # 16-bit WAV under `dir`. Off by default; toggle it on for a capture session.
    enabled: bool = False
    dir: str = "data/captures"


class AckConfig(BaseModel):
    # Server-side "end-of-phrase" confirmation chime ("блям"). The wake-word chime was
    # moved OFF the device (it overlapped the user's command now that the firmware
    # streams continuously with pre-roll); instead the server plays this short clip to
    # the speaker the instant VAD end-points the utterance — immediate "got it"
    # feedback, BEFORE STT/LLM/TTS run. The full spoken reply still plays later.
    enabled: bool = True
    # Optional override for the chime audio file (mp3 or wav). When set to an existing
    # path the operator's exact device «блям» clip is used; when empty (or missing) a
    # short two-tone sine chime is synthesized once and cached. Read live per run.
    sound_path: str = ""


class CoreConfig(BaseModel):
    context: ContextConfig = ContextConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    mic: MicConfig = MicConfig()
    network: NetworkConfig = NetworkConfig()
    openweathermap: OpenWeatherMapConfig = OpenWeatherMapConfig()
    mcp_servers: list[McpServerConfig] = []
    calendar: CalendarConfig = CalendarConfig()
    esphome: EsphomeConfig = EsphomeConfig()
    prompt: PromptConfig = PromptConfig()
    panel: PanelConfig = PanelConfig()
    runs: RunsConfig = RunsConfig()
    reminders: RemindersConfig = RemindersConfig()
    capture: CaptureConfig = CaptureConfig()
    ack: AckConfig = AckConfig()
    devices: list[DeviceConfig] = []
    tts_timeout: int = 30
    log_level: str = "INFO"
