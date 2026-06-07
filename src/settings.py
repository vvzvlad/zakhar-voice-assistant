"""Single source of configuration via pydantic-settings."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_errors import load_settings_or_exit


@dataclass(frozen=True)
class DeviceConfig:
    """One ESPHome speaker the assistant connects to (as the API client)."""

    name: str
    host: str
    psk: str


class Settings(BaseSettings):
    # --- Credentials (no default) ---
    # Missing in the environment → Settings() raises at startup. Never defaulted.
    groq_api_key: str
    weather_api_key: str

    # --- Our own services (no default) ---
    mcp_smarthome_url: str      # external smart-home MCP server endpoint (Node-RED), e.g. http://10.0.0.5:8001/mcp
    tts_base_url: str           # home_assistant_tts service base, e.g. http://10.0.0.5:8124
    public_base_url: str        # base URL the speakers use to fetch audio, e.g. http://10.0.0.10:8200

    # --- Devices list, flat string, parsed in @property (no default) ---
    esphome_devices: str        # "name|host|psk;name2|host2|psk2"

    # --- Non-secret config (defaults OK) ---
    mcp_smarthome_token: str = ""  # Optional bearer token for the MCP server; empty = no auth header
    esphome_port: int = 6053
    groq_model: str = "openai/gpt-oss-120b"
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_proxy: str = ""        # SOCKS/HTTP proxy for Groq + weather; empty = direct
    weather_city: str = "Moscow"
    stt_backend: str = "groq"   # groq | vosk
    vosk_model_path: str = "models/vosk-model-small-ru-0.22"
    tts_backend: str = "teratts"  # teratts | piper
    piper_voice_path: str = "models/ru_RU-irina-medium.onnx"  # config json expected at <path>.json
    tts_timeout: int = 30
    audio_host: str = "0.0.0.0"
    audio_port: int = 8200
    audio_ttl: int = 300        # seconds an MP3 stays in the audio-server cache
    log_level: str = "INFO"
    system_prompt_path: str = "data/system_prompt.md"
    context_dir: str = "data"   # per-device context files live here as context_<name>.txt

    # --- Server-side VAD end-pointing (defaults OK) ---
    # The speaker streams mic PCM continuously and never signals end-of-speech,
    # so we detect the end of an utterance ourselves with WebRTC VAD.
    # webrtcvad accepts only 0..3 and raises on anything else; constrain here so a
    # bad value fails fast with a clear validation error instead of an obscure
    # webrtcvad traceback at Pipeline construction time.
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)  # higher = more aggressive non-speech filtering
    vad_silence_ms: int = 800            # trailing silence (after speech) that ends the utterance
    vad_min_speech_ms: int = 200         # minimum speech before end-detection arms
    vad_max_utterance_ms: int = 15000    # hard cap: finalize even if no trailing silence
    vad_no_speech_timeout_ms: int = 8000  # if no speech detected at all after start, finalize (STT-empty -> RUN_END)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def devices(self) -> list[DeviceConfig]:
        """Parse ESPHOME_DEVICES ("name|host|psk;name2|host2|psk2") into DeviceConfig list.
        Devices are separated by ';', fields by '|'. Whitespace around tokens is stripped.
        Empty entries (e.g. trailing ';') are skipped. A malformed entry (not exactly 3
        non-empty fields) raises ValueError so a typo fails loudly at startup."""
        result: list[DeviceConfig] = []
        for entry in self.esphome_devices.split(";"):
            entry = entry.strip()
            if not entry:
                continue  # skip empty entries (e.g. trailing ';')
            # Base64 PSKs never contain ';' or '|', so these separators are safe.
            parts = [p.strip() for p in entry.split("|")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"Invalid ESPHOME_DEVICES entry: {entry!r}")
            name, host, psk = parts
            result.append(DeviceConfig(name=name, host=host, psk=psk))
        return result


settings = load_settings_or_exit(Settings)
