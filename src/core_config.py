"""Core (non-provider) configuration sections.

These have no provider choice, no runtime stage factory and no multiplicity, so
they are plain pydantic models (not providers). They still feed the panel the same
way — via model_json_schema() — but stay out of the REGISTRY.
"""

from pydantic import BaseModel, Field


class ContextConfig(BaseModel):
    max_turns: int = Field(5, ge=1)
    ttl_seconds: int = Field(300, ge=0)
    dir: str = "data"


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


class NetworkConfig(BaseModel):
    external_proxy: str = ""


class WeatherConfig(BaseModel):
    api_key: str = ""
    city: str = "Moscow"


class McpConfig(BaseModel):
    # Single smart-home MCP server for now (multi-server is a later feature).
    url: str = ""
    token: str = ""


class DeviceConfig(BaseModel):
    name: str
    host: str
    psk: str


class EsphomeConfig(BaseModel):
    port: int = 6053


class PromptConfig(BaseModel):
    system_prompt_path: str = "data/system_prompt.md"


class CoreConfig(BaseModel):
    context: ContextConfig = ContextConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    network: NetworkConfig = NetworkConfig()
    weather: WeatherConfig = WeatherConfig()
    mcp: McpConfig = McpConfig()
    esphome: EsphomeConfig = EsphomeConfig()
    prompt: PromptConfig = PromptConfig()
    devices: list[DeviceConfig] = []
    tts_timeout: int = 30
    log_level: str = "INFO"
