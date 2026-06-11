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
    aggressiveness: int = Field(
        2, ge=0, le=3,
        title="Speech detection strictness",
        # Rendered as a labeled segment control (see SchemaForm.ScaleSeg): each level
        # gets a word label, the extremes get pole captions, and the numeric value is
        # shown small for debugging. The "stricter = cuts sooner" explanation lives in
        # those labels, so the description stays short.
        json_schema_extra={
            "choices": [
                {"value": 0, "label": "Lenient"},
                {"value": 1, "label": "Balanced"},
                {"value": 2, "label": "Strict"},
                {"value": 3, "label": "Strictest"},
            ],
            "poles": ["waits longest", "cuts off soonest"],
            "readout": True,
        },
        description="How strictly WebRTC VAD decides that speech has ended.",
    )
    silence_ms: int = Field(
        800,
        title="End-of-phrase pause",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Trailing silence after speech before the utterance is considered finished. Higher tolerates longer mid-phrase pauses but adds the same delay before the reply starts.",
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
        description="Drop this many ms off the start of the captured sample before STT (cuts the wake-word tail / button-press click). 0 disables.",
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
    # Rolling window of recent utterance audio, kept in runs.db so operators can
    # download/play it from the log to diagnose mis-triggers. Stereo when the device
    # streams both mic channels: left = the pipeline/STT channel exactly as STT
    # received it, right = the other raw mic channel; mono when only one stream is sent.
    store_audio: bool = True
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
    enabled: bool = Field(
        True,
        description="Play a short confirmation chime to the speaker the instant your phrase is end-pointed — immediate \"got it\" feedback before STT/LLM/TTS run. The full spoken reply still plays afterwards.",
    )
    # The dropdown is fed by GET /api/chimes (bundled assets/chimes clips); an empty
    # value keeps the synthesized chime, which is built once and cached.
    sound_path: str = Field(
        "",
        title="Chime sound",
        json_schema_extra={"options": "dynamic"},
        description="Chime audio file (mp3 or wav) played on end-of-phrase. Pick a bundled clip to use that exact «блям»; the synthesized default (empty value) plays a short two-tone sine chime. Applies per run.",
    )


class CoreConfig(BaseModel):
    context: ContextConfig = ContextConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    network: NetworkConfig = NetworkConfig()
    openweathermap: OpenWeatherMapConfig = OpenWeatherMapConfig()
    mcp_servers: list[McpServerConfig] = []
    calendar: CalendarConfig = CalendarConfig()
    esphome: EsphomeConfig = EsphomeConfig()
    prompt: PromptConfig = PromptConfig()
    runs: RunsConfig = RunsConfig()
    reminders: RemindersConfig = RemindersConfig()
    capture: CaptureConfig = CaptureConfig()
    ack: AckConfig = AckConfig()
    devices: list[DeviceConfig] = []
    tts_timeout: int = 30
    log_level: str = "INFO"
