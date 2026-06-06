import os

# Provide the required env vars BEFORE any test module imports src.settings
# (Settings() is instantiated at import time and would otherwise fail). In CI the
# same variables are injected via the workflow's `env:` block.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("WEATHER_API_KEY", "test-weather-key")
os.environ.setdefault("SMARTHOME_URL", "http://smarthome.test/voice_command")
os.environ.setdefault("TTS_BASE_URL", "http://tts.test:8124")
os.environ.setdefault("PUBLIC_BASE_URL", "http://10.0.0.10:8200")
os.environ.setdefault("ESPHOME_DEVICES", "living|10.0.0.5|dGVzdHBzaw==")
