"""Core (non-provider) configuration sections.

These have no provider choice, no runtime stage factory and no multiplicity, so
they are plain pydantic models (not providers). They still feed the panel the same
way — via model_json_schema() — but stay out of the REGISTRY.
"""

from typing import Literal

from pydantic import BaseModel, Field

# SECRET_FIELD_EXTRA marks a genuine credential field (the panel renders it as a masked
# reveal input), ADVANCED_FIELD_EXTRA a rarely changed one (folded under "Advanced").
# Imported from src.plugins.base (no circular import — base imports nothing from here).
from src.plugins.base import ADVANCED_FIELD_EXTRA, PHRASE_FIELD_EXTRA, SECRET_FIELD_EXTRA

# Panel descriptions shared by the built-in tool sources.
SOURCE_ENABLED_DESCRIPTION = "Off removes this source's tools and its prompt block."
SOURCE_PROMPT_DESCRIPTION = "Appended to the system prompt to describe this source's tools to the model."


class ContextConfig(BaseModel):
    max_turns: int = Field(
        5, ge=1,
        title="Context depth",
        description="How many recent question-answer pairs are sent to the model.",
        json_schema_extra={"unit": "turns"},
    )
    ttl_seconds: int = Field(
        300, ge=0,
        title="Dialog timeout",
        json_schema_extra={"unit": "s"},
        description="Idle time after which a new dialog starts; 0 starts every request fresh.",
    )


class AudioConfig(BaseModel):
    # Bind address of the device server (devices connect to ws://<server>:<port>/device).
    host: str = Field("0.0.0.0", title="Bind host")
    port: int = Field(8200, title="Port")
    # Announced by mDNS. Empty: the bind host, or this machine's LAN address when it binds
    # all; in a Docker bridge network that is the container's own address, which the devices
    # cannot reach, so the host's LAN address goes here.
    advertise_host: str = Field(
        "",
        title="Advertised address",
        description="LAN address devices find this server at; set the host's address when in Docker.",
    )


class AgentMcpConfig(BaseModel):
    """Agent-facing MCP server (external agents drive the assistant over /mcp).

    Served by the admin panel on the panel port — there is no separate bind, so
    `enabled` is the only knob (read live per /mcp request)."""
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
        description="Silence after speech that ends the phrase; more allows longer pauses but delays the reply.",
    )
    min_speech_ms: int = Field(
        200,
        title="Minimum utterance length",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="Less speech than this is taken for noise and dropped.",
    )
    max_utterance_ms: int = Field(
        15000,
        title="Maximum utterance length",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="A phrase is cut off at this length even without a pause.",
    )
    # The wake word itself does not count as speech. A run that hears nothing by then ends
    # silently and sends nothing to STT.
    no_speech_timeout_ms: int = Field(
        4000,
        title="Wait for the command",
        json_schema_extra={"unit": "ms", "multipleOf": 100},
        description="How long to wait for the command after the wake word before giving up.",
    )
    # The oldest part of the device's 2000 ms pre-roll, cut off the captured sample before STT.
    trim_start_ms: int = Field(
        200,
        ge=0,
        title="Trim from start",
        json_schema_extra={**ADVANCED_FIELD_EXTRA, "unit": "ms", "multipleOf": 100},
        description="Audio dropped from the start, before the wake word, ahead of STT; 0 drops nothing.",
    )
    # Falls back to the most-processed stream if the device does not send the selected one.
    # Applies on the next utterance.
    mic_channel: Literal[0, 1] = Field(
        1,
        title="Mic processing level",
        # Rendered as a segment control with pole captions (see SchemaForm.ScaleSeg).
        # Lower = more processed / louder; higher = rawer / cleaner.
        json_schema_extra={"poles": ["more processed / louder", "raw / cleaner"]},
        description="Which microphone stream feeds the pipeline: lower is louder, higher is cleaner.",
    )
    # Peak-normalizes each captured utterance so its loudest sample reaches ~-3 dBFS: the
    # quieter, less-processed channel gets a steady loudness without clipping; near-silent
    # clips are left as they are. Applies on the next utterance.
    mic_normalize: bool = Field(
        False,
        title="Volume normalization",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
        description="Raises each phrase to a steady loudness before STT.",
    )
    # A ~80 Hz high-pass over the whole utterance strips the DC offset and the rumble (table
    # thumps, HVAC) that hurt recognition. Applies on the next utterance.
    mic_highpass: bool = Field(
        False,
        title="Low-frequency filter",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
        description="Cuts hum and rumble below about 80 Hz before STT.",
    )
    # Lets any VAD engine end-point the quiet, less-processed mic channel. Read live per chunk.
    mic_auto_gain: bool = Field(
        False,
        title="VAD auto gain",
        description="Boosts quiet audio for the VAD decision only; the recording stays as it is.",
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
    city: str = Field(
        "Moscow", title="City", description="City the weather is reported for.",
        json_schema_extra=PHRASE_FIELD_EXTRA,
    )
    # Appended to the system prompt to describe this built-in MCP's tools to the model.
    prompt: str = Field("", title="Tools prompt", description=SOURCE_PROMPT_DESCRIPTION,
                        json_schema_extra={"widget": "textarea"})


class McpServerConfig(BaseModel):
    # One external MCP server entry. `name` is unique and also the ToolHub source id
    # (so it matches the /api/tools source id). `prompt` is appended to the system
    # prompt to describe this server's tools to the model.
    # Master switch: a disabled source is removed from the ToolHub and its prompt block is dropped.
    enabled: bool = Field(True, title="Enabled", description=SOURCE_ENABLED_DESCRIPTION)
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
                     description="ZeroClaw webhook channel the tasks are posted to.")
    secret: str = Field("", title="Webhook secret", json_schema_extra=SECRET_FIELD_EXTRA)
    task_suffix: str = Field(
        "Ответ прозвучит на колонке, когда будет готов: начни с того, о чём была задача, "
        "ответь коротко, одним-тремя предложениями, без разметки, списков и ссылок.",
        title="Task suffix",
        description="Appended to every task after a blank line.",
        json_schema_extra=PHRASE_FIELD_EXTRA,
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
    # Sent to the device in `config`: its cancellation of its own playback in the
    # microphone its stop word hears while it plays (the Voice PE switches its XMOS channel 0
    # to the raw microphone for it). 0 off, the stop word hears the wake word's input; 1..3
    # on, with the residual echo suppression from mild to strongest (the full-duplex modes
    # only; in the others any of 1..3 just means on). The mode: 0 SR low cost, 1 SR high
    # performance, 2 FD low cost, 3 FD high performance; the filter length 1..8. The device
    # re-creates its canceller when the mode or the filter length changes.
    echo_cancel: int = Field(1, ge=0, le=3)
    echo_cancel_mode: int = Field(2, ge=0, le=3)
    echo_cancel_filter: int = Field(4, ge=1, le=8)
    # Sent to the device in `config`, ms: the audio its speaker holds before it starts a
    # stream, and again after the stream runs dry. More: fewer stutters while the TTS pauses,
    # the answer starts later.
    play_prime_ms: int = Field(700, ge=0, le=2000)
    # Auto volume. auto_volume goes to the device in `config`: while it is on, the speaker
    # measures the room's noise on its raw microphone every 15 s of quiet and sends `ambient`
    # (dBFS of its 32-bit microphone slot). The threshold and the gain stay on the server: the
    # noise (the median of the last minute's `ambient`) adds (noise - threshold) x gain percent to
    # the speaker's own volume, sent as `volume_boost`: above the threshold the reply is louder
    # than the dial, below it quieter. The gain is percent per dB: the firmware's volume is ~0.39
    # dB per percent (audio_core volume_dac_byte: 0.45/100 x 175 steps of 0.5 dB), so 2.5 %/dB
    # moves the reply 1 dB per 1 dB of noise.
    auto_volume: bool = False
    auto_volume_threshold: int = Field(-60, ge=-120, le=0)
    auto_volume_gain: float = Field(2.0, ge=0, le=10)
    # The server runs everything it plays on the device through src/reply_compressor.py, read
    # at the start of each stream: quiet parts up to 20 dB louder, peaks held at -11.5 dBFS, so
    # speech is denser. Louder at the same dial for a TTS whose peaks sit below full scale (about
    # +2-3 dB with Fish Audio); one that peaks at full scale, as Piper does, plays up to ~3 dB
    # quieter. -11.5 dBFS because the device's DAC adds up to +10.5 dB at the top of the dial:
    # the peaks stay 1 dB under full scale.
    compress_replies: bool = False
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
    # Sent to the device in `log_stream` after every hello and whenever it changes.
    # The server keeps it core.device_log_days days under data/device_logs/; off, the device
    # sends nothing.
    log: bool = Field(
        False,
        title="Device log",
        description="The device streams its whole log to the server.",
    )


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
    # Older recordings are dropped as new ones arrive, from the next utterance on.
    store_audio: bool = Field(True, title="Record audio")
    audio_keep: int = Field(
        100, ge=1,
        title="Stored recordings",
        description="How many recent phrase recordings the Request Log keeps; more uses more disk.",
    )  # rolling ring-buffer cap; applied per-utterance via RunsStore.put_audio()


class RemindersConfig(BaseModel):
    # One-shot voice reminders (data/reminders.db). Disable to drop the subsystem.
    enabled: bool = Field(True, title="Enabled", description="Off removes the reminder tools and pauses announcing reminders.")


class AckConfig(BaseModel):
    # Server-side "end-of-phrase" confirmation chime ("блям"). The wake-word chime was
    # moved OFF the device (it overlapped the user's command now that the firmware
    # streams continuously with pre-roll); instead the server plays this short clip to
    # the speaker the instant VAD end-points the utterance — immediate "got it"
    # feedback, BEFORE STT/LLM/TTS run. The full spoken reply still plays later.
    enabled: bool = Field(
        True,
        title="End-of-phrase chime",
        description="Plays a short chime on the device the moment your phrase ends.",
    )
    # The dropdown is fed by GET /api/chimes (bundled assets/chimes clips); an empty
    # value keeps the synthesized chime (a short two-tone sine), which is built once and
    # cached. Applies per run.
    sound_path: str = Field(
        "",
        title="Chime sound",
        json_schema_extra={"options": "dynamic"},
        description="A bundled clip, or the default synthesized two-tone chime.",
    )


# Default verbs are real (not commented) so the command handlers work out of the box.
_DEFAULT_ACTIONS = (
    "on = включи, включай, зажги, зажги свет, вруби, врубай, запусти\n"
    "off = выключи, выключай, погаси, выруби, отключи, останови"
)

_DEFAULT_FILLER_WORDS = "захар, пожалуйста, ну, а, давай, ка, мне, спасибо"

# A self-documenting, all-commented example so the aliases field matches nothing
# until the operator configures it for their actual entity/scene ids.
_DEFAULT_ALIASES = (
    "# One mapping per line. LHS: comma-separated Russian synonyms; RHS: the exact\n"
    "# ARG VALUE the tool expects (an entity/scene id), auto-routed to the tool+param\n"
    "# whose enum contains it. Use 'tool.slot:value' to target explicitly when the\n"
    "# MCP schema does not enumerate the value. Lines starting with # are ignored.\n"
    "# свет в зале, люстра = bright_room_light\n"
    "# ночь, спать = night\n"
    "# яркость, свет.brightness:80 = set_light.brightness:80"
)


class CommandsConfig(BaseModel):
    """The command catalog: the household's spoken names of the devices and scenes,
    the action verbs, the filler words and the values an on/off verb sets on a level
    slot. Read by the command handlers (simple-nlu builds its model from it, jev reads
    it per request; src/commands.py); edited on the panel's Commands page."""

    aliases: str = Field(
        _DEFAULT_ALIASES,
        title="Aliases (Russian phrase = value)",
        description=(
            "Map Russian phrases to the exact tool arg value. One per line: "
            "'phrase1, phrase2 = value' (auto-route by enum) or "
            "'phrase = tool.slot:value' (explicit target). '#' lines are ignored."
        ),
        json_schema_extra={"widget": "textarea"},
    )
    actions: str = Field(
        _DEFAULT_ACTIONS,
        title="Action verbs",
        description=(
            "Map a canonical action name to its Russian trigger verbs, one per "
            "line: 'action = verb1, verb2, ...'. When the action name equals an "
            "enum value of a tool's state-like param, it fills that slot. A verb "
            "counts in any imperative or infinitive form («включи», «включить»)."
        ),
        json_schema_extra={"widget": "textarea"},
    )
    # In strict mode, when a later recognition chain takes what this one declines, a direct
    # match runs only if every word is an action verb, a word of the matched alias phrase, a
    # number that set a value, or one of these; the first word may also be one letter off one
    # of these of four letters or more (a misheard «захар»).
    filler_words: str = Field(
        _DEFAULT_FILLER_WORDS,
        title="Filler words (strict mode)",
        description="Comma-separated extra words a command may contain in strict mode.",
        json_schema_extra=PHRASE_FIELD_EXTRA,
    )
    # A numeric level slot with no on/off enum, e.g. dimmer brightness.
    level_on: int = Field(
        100,
        ge=0,
        title="On level",
        description="Level an on-verb sets on a numeric level slot; an off-verb sets 0.",
    )
    # A required string level slot with no enum.
    off_token: str = Field(
        "off",
        title="Off token (string level slot)",
        description="Value an off-verb sets on a text level slot, e.g. a climate temperature.",
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
    ack: AckConfig = AckConfig()
    commands: CommandsConfig = CommandsConfig()
    devices: list[DeviceConfig] = []
    chains: list[ChainConfig] = []
    tts_timeout: int = 30
    log_level: str = "INFO"
    # How many days of device log files (core.devices[].log) are kept.
    device_log_days: int = 7
