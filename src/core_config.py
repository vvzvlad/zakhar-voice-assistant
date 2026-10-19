"""Core (non-provider) configuration sections.

These have no provider choice, no runtime stage factory and no multiplicity, so
they are plain pydantic models (not providers). They still feed the panel the same
way — via model_json_schema() — but stay out of the REGISTRY.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Marks a genuine credential field: the panel renders it as a masked reveal input.
# Imported from src.plugins.base (no circular import — base imports nothing from here).
from src.plugins.base import SECRET_FIELD_EXTRA

# Panel descriptions shared by the built-in tool sources.
SOURCE_ENABLED_DESCRIPTION = "Off removes this source's tools and its prompt block."
SOURCE_PROMPT_DESCRIPTION = "Appended to the system prompt to describe this source's tools to the model."


class ContextConfig(BaseModel):
    max_turns: int = Field(
        5, ge=1,
        title="Context depth",
        description="How many recent question-answer pairs are sent to the model.",
    )
    ttl_seconds: int = Field(
        300, ge=0,
        title="Dialog timeout (s)",
        description="Idle time after which the next request starts a new dialog; 0 = every request starts fresh.",
    )


class AudioConfig(BaseModel):
    # Bind address of the device server (devices connect to ws://<server>:<port>/device).
    host: str = Field("0.0.0.0", title="Bind host")
    port: int = Field(8200, title="Port")
    advertise_host: str = Field(
        "",
        title="Advertised address",
        description="LAN address the devices find this server at by mDNS. Empty: the bind host, "
        "or this machine's LAN address when it binds all; in a Docker bridge network that is the "
        "container's own address, so set the host's LAN address there.",
    )


class AgentMcpConfig(BaseModel):
    """Agent-facing MCP server (external agents drive the assistant over /mcp).

    Served by the admin panel on the panel port — there is no separate bind, so
    `enabled` is the only knob (read live per /mcp request). Old docs may still
    carry host/port keys: pydantic ignores extras on parse and the stale keys
    drop on the next save — no migration needed."""
    enabled: bool = Field(True, title="Enabled")


class VadConfig(BaseModel):
    """Engine-independent voice-capture policy: end-pointing thresholds plus the
    mic channel selection and conditioning toggles (pre-STT highpass/normalize and
    the decision-only VAD auto gain, applied by the pipeline for every VAD engine).
    The speech classifier itself (e.g. WebRTC aggressiveness) is the swappable
    vad stage plugin (src/plugins/vad/*), configured under the `vad` slot."""

    silence_ms: int = Field(
        800,
        title="End-of-phrase pause",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Trailing silence after speech before the utterance is considered finished. Higher tolerates longer mid-phrase pauses but adds the same delay before the reply starts.",
    )
    wake_pause_ms: int = Field(
        2000,
        title="Pause after the wake word",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Silence allowed between the wake word and the start of the command. Until the command starts, the phrase ends only after this pause, not after the end-of-phrase pause; a command said in one breath with the wake word is not delayed. At or below the end-of-phrase pause it has no effect.",
    )
    min_speech_ms: int = Field(
        200,
        title="Minimum utterance length",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Minimum total detected speech for the audio to count as a real utterance; shorter blips are treated as noise and dropped.",
    )
    max_utterance_ms: int = Field(
        15000,
        title="Maximum utterance length",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Hard cap on a single utterance — it is force-finalized once it reaches this length even if no trailing silence was seen.",
    )
    no_speech_timeout_ms: int = Field(
        8000,
        title="Give up after silence",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="If no speech at all is detected after capture starts, give up and end the run after this long.",
    )
    trim_start_ms: int = Field(
        200,
        ge=0,
        title="Trim from start",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Drop this many ms off the start of the captured sample before STT: the oldest part of the 2000 ms pre-roll, before the wake word. 0 disables.",
    )
    mic_channel: Literal[0, 1] = Field(
        0,
        title="Mic processing level",
        # Rendered as a segment control with pole captions (see SchemaForm.ScaleSeg).
        # Lower = more processed / louder; higher = rawer / cleaner.
        json_schema_extra={"poles": ["more processed / louder", "raw / cleaner"]},
        description="Selects which mic stream feeds the pipeline. Lower is more processed and louder; higher is rawer and cleaner. Falls back to the most-processed stream if the device does not provide the selected one. Applies on the next utterance.",
    )
    mic_normalize: bool = Field(
        False,
        title="Volume normalization",
        description="Peak-normalize each captured utterance before STT so its loudest sample reaches a target level (~-3 dBFS). Adapts the quieter less-processed channel to a consistent loudness without clipping; near-silent clips are left untouched. Off by default; applies on the next utterance.",
    )
    mic_highpass: bool = Field(
        False,
        title="Low-frequency filter",
        description="Apply a ~80 Hz high-pass filter to the whole utterance before STT to strip DC offset and low-frequency rumble (table thumps, HVAC) that hurt recognition. Off by default; applies on the next utterance.",
    )
    mic_auto_gain: bool = Field(
        False,
        title="VAD auto gain",
        description="Boost quiet audio for the speech/no-speech decision only, before it reaches the VAD engine (the recorded/STT audio is untouched). Lets any VAD engine end-point the quiet, less-processed mic channel. Read live per chunk.",
    )


class NetworkConfig(BaseModel):
    external_proxy: str = Field(
        "",
        title="External proxy",
        description="Proxy URL for cloud API requests, e.g. socks5://host:1080; empty = direct.",
    )


class OpenWeatherMapConfig(BaseModel):
    # Master switch: a disabled source is removed from the ToolHub and its prompt block is dropped.
    enabled: bool = Field(True, title="Enabled", description=SOURCE_ENABLED_DESCRIPTION)
    api_key: str = Field("", title="API key", json_schema_extra=SECRET_FIELD_EXTRA)
    city: str = Field("Moscow", title="City", description="City the weather is reported for.")
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", title="Tools prompt", description=SOURCE_PROMPT_DESCRIPTION,
                        json_schema_extra={"widget": "textarea"})


class McpServerConfig(BaseModel):
    # One external MCP server entry. `name` is unique and also the ToolHub source id
    # (so it matches the /api/tools source id). `prompt` is appended to the system
    # prompt to describe this server's tools to the model.
    # Master switch: a disabled source is removed from the ToolHub and its prompt block is dropped.
    enabled: bool = True
    name: str
    url: str = ""
    token: str = Field("", json_schema_extra=SECRET_FIELD_EXTRA)
    transport: Literal["auto", "streamable_http", "sse"] = "auto"
    prompt: str = ""
    slow: bool = Field(
        False,
        title="Slow tools",
        description="Mark this server's tools as slow (web search, long lookups). The assistant speaks a short filler line before calling a slow tool so the user is not left waiting in silence.",
    )


class CalendarConfig(BaseModel):
    # CalDAV account for the built-in calendar MCP. `calendar` selects which calendar
    # to use by name; empty means the first calendar on the account.
    # Master switch: a disabled source is removed from the ToolHub and its prompt block is dropped.
    enabled: bool = Field(True, title="Enabled", description=SOURCE_ENABLED_DESCRIPTION)
    url: str = Field("", title="CalDAV URL")
    username: str = Field("", title="Username")
    password: str = Field("", title="Password", json_schema_extra=SECRET_FIELD_EXTRA)
    calendar: str = Field("", title="Calendar", description="Calendar name on the account; empty = the first one.")
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", title="Tools prompt", description=SOURCE_PROMPT_DESCRIPTION,
                        json_schema_extra={"widget": "textarea"})


class AssistantTasksConfig(BaseModel):
    # Jobs handed off to the ZeroClaw assistant through its webhook channel;
    # ZeroClaw posts each reply to POST /api/assistant-tasks/reply, which speaks it.
    # Master switch: a disabled source is removed from the ToolHub and its prompt block is dropped.
    enabled: bool = Field(True, title="Enabled", description=SOURCE_ENABLED_DESCRIPTION)
    url: str = Field("", title="Webhook URL",
                     description="ZeroClaw webhook channel the tasks are posted to. Its replies go to "
                     "POST /api/assistant-tasks/reply on this panel.")
    secret: str = Field("", title="Webhook secret", json_schema_extra=SECRET_FIELD_EXTRA)
    task_suffix: str = Field(
        "Ответ прозвучит на колонке, когда будет готов: начни с того, о чём была задача, "
        "ответь коротко, одним-тремя предложениями, без разметки, списков и ссылок.",
        title="Task suffix",
        description="Appended to every task after a blank line.",
    )
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", title="Tools prompt", description=SOURCE_PROMPT_DESCRIPTION,
                        json_schema_extra={"widget": "textarea"})


class DeviceConfig(BaseModel):
    # The device's hello id.
    name: str
    # Shown in the admin panel instead of the id; empty shows the id. Display only:
    # everything server-side (context, reminders, run log, agent MCP) keys on `name`.
    label: str = ""
    # When False the server refuses the device's connection; flipping it is
    # hot-applied via DeviceManager.reconfigure().
    enabled: bool = True
    # Sent to the device in `config`, percent: its wake-word probability cutoff and
    # its on-device VAD gate.
    wake_cutoff: int = Field(80, ge=0, le=100)
    vad_cutoff: int = Field(30, ge=0, le=100)
    # The .tflite models sent to the device: its wake-word model and the VAD model that gates
    # it. Empty: the model the release ships (device_server.RELEASE_WAKE_MODEL /
    # RELEASE_VAD_MODEL). wake_model is otherwise a model name from GET /api/wake-models
    # (zakhar/zakhar_v28); either field also takes the path of another file, e.g. on the
    # models volume.
    # Empty rather than a path by default: device lists are written back to config.json with
    # every field (Device.set_control and _hello dump them), so a default path would be frozen
    # into the stored config and outlive its file in a later release.
    wake_model: str = ""
    vad_model: str = ""


class PromptConfig(BaseModel):
    system_prompt_path: str = "data/system_prompt.md"


class RunsConfig(BaseModel):
    # Observability: persist every pipeline run to SQLite (data/runs.db). Disable
    # to skip recording entirely; retention_days=0 keeps rows forever.
    enabled: bool = Field(True, title="Record runs")
    retention_days: int = Field(30, ge=0, title="Retention (days)", description="0 keeps runs forever.")
    # Rolling window of recent utterance audio, kept in runs.db so operators can
    # download/play it from the log to diagnose mis-triggers. Stereo when the device
    # streams both mic channels: left = the pipeline/STT channel exactly as STT
    # received it, right = the other raw mic channel; mono when only one stream is sent.
    store_audio: bool = Field(True, title="Record audio")
    audio_keep: int = Field(
        100, ge=1,
        title="Stored recordings",
        description="How many of the most recent utterance recordings to keep for "
        "playback in the Request Log. Older recordings are dropped as new ones "
        "arrive (applied on the next utterance). The audio is stored in runs.db, "
        "so a higher number uses more disk.",
    )  # rolling ring-buffer cap; applied per-utterance via RunsStore.put_audio()


class RemindersConfig(BaseModel):
    # One-shot voice reminders (data/reminders.db). Disable to drop the subsystem.
    enabled: bool = Field(True, title="Enabled", description="Off removes the reminder tools and pauses announcing reminders.")


class CaptureConfig(BaseModel):
    # Raw audio capture: when enabled, every finalized utterance's post-frontend PCM
    # (what the device streams) is saved as a 16 kHz / mono /
    # 16-bit WAV under `dir`. Off by default; toggle it on for a capture session.
    enabled: bool = False
    dir: str = "data/captures"


class AckConfig(BaseModel):
    # Server-side "end-of-phrase" confirmation chime ("блям"). The wake-word chime was
    # moved OFF the device (it overlapped the user's command now that the firmware
    # streams continuously with pre-roll); instead the server plays this short clip to
    # the speaker the instant VAD end-points the utterance — immediate "got it"
    # feedback, BEFORE STT/LLM/TTS run. The full spoken reply still plays later.
    enabled: bool = Field(
        True,
        title="End-of-phrase chime",
        description="Play a short confirmation chime to the speaker the instant your phrase is end-pointed — immediate \"got it\" feedback before STT/LLM/TTS run. The full spoken reply still plays afterwards.",
    )
    # The dropdown is fed by GET /api/chimes (bundled assets/chimes clips); an empty
    # value keeps the synthesized chime, which is built once and cached.
    sound_path: str = Field(
        "",
        title="Chime sound",
        json_schema_extra={"options": "dynamic"},
        description="Chime audio file (WAV) played on end-of-phrase. Pick a bundled clip to use that exact «блям»; the synthesized default (empty value) plays a short two-tone sine chime. Applies per run.",
    )


class ChainConfig(BaseModel):
    """One recognition chain: an STT provider and the handler (llm provider) that
    takes its text. Each utterance goes through the enabled chains in list order
    until one answers."""
    name: str
    enabled: bool = True
    stt: str          # stt provider id (its settings live in stt.instances.<id>)
    handler: str      # llm provider id (its settings live in llm.instances.<id>)
    # Restrict the STT to the handler's closed vocabulary when the STT supports it.
    narrow: bool = True
    # Start this chain's STT at speech, alongside the first chain's, instead of
    # only when the chain is reached.
    parallel: bool = False


class CoreConfig(BaseModel):
    context: ContextConfig = ContextConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    network: NetworkConfig = NetworkConfig()
    openweathermap: OpenWeatherMapConfig = OpenWeatherMapConfig()
    mcp_servers: list[McpServerConfig] = []
    calendar: CalendarConfig = CalendarConfig()
    assistant_tasks: AssistantTasksConfig = AssistantTasksConfig()
    agent_mcp: AgentMcpConfig = AgentMcpConfig()
    prompt: PromptConfig = PromptConfig()
    runs: RunsConfig = RunsConfig()
    reminders: RemindersConfig = RemindersConfig()
    capture: CaptureConfig = CaptureConfig()
    ack: AckConfig = AckConfig()
    devices: list[DeviceConfig] = []
    chains: list[ChainConfig] = []
    tts_timeout: int = 30
    log_level: str = "INFO"
