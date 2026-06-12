"""Provider discovery: importing this package runs every provider's @register.

Explicit imports (no entry points / dynamic discovery) — all providers are in-repo
and known at build time, per the config-system design.
"""

from src.plugins import base  # noqa: F401
from src.plugins.vad import silero, ten, webrtc  # noqa: F401
from src.plugins.tts import teratts, piper, yandex, fishaudio  # noqa: F401
from src.plugins.stt import groq as stt_groq, openrouter as stt_openrouter, vosk  # noqa: F401
from src.plugins.llm import openrouter, groq as llm_groq  # noqa: F401
from src.plugins.stress import ruaccent as ruaccent_provider  # noqa: F401
