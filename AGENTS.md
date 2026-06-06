# Agent Instructions — zakhar-voice-assistant

Client-side voice assistant for a "HA Voice PE" ESP32 speaker. It connects to the
speaker over the ESPHome Native API (as the client), then runs the pipeline:
STT (Groq Whisper) → LLM (Groq chat) → smart-home commands (Node-RED) →
TTS (local TeraTTS HTTP service), and serves the resulting audio back to N speakers.
It replaces Home Assistant in this loop.

## Project structure
- `src/` — application code (`settings.py` is the single config entry point)
- `tests/` — pytest
- `data/` — runtime state: per-device context files, cached prompt (gitignored, docker volume)
- `templates/` — static assets that ship inside the image (`default_prompt.md`)
- `main.py` — thin entry point over `src/`

## Setup
All routine actions go through the `Makefile` — run `make help` to list targets.
```bash
make install           # create .venv and install dev/test deps
cp .env.example .env   # then fill in the values  (shortcut: make env)
```

## Running tests
```bash
make test              # runs .venv/bin/pytest
```

## Running the app
```bash
make run               # runs .venv/bin/python main.py
```

## Environment

Required (no default — missing → fail at startup):
- `GROQ_API_KEY`, `WEATHER_API_KEY` — credentials.
- `SMARTHOME_URL` — Node-RED endpoint for smart-home commands.
- `TTS_BASE_URL` — local TeraTTS service base.
- `PUBLIC_BASE_URL` — base URL the speakers use to fetch audio.
- `ESPHOME_DEVICES` — `name|host|psk;name2|host2|psk2`.

Optional (sensible defaults in code): `ESPHOME_PORT`, `GROQ_MODEL`,
`GROQ_STT_MODEL`, `GROQ_PROXY`, `WEATHER_CITY`, `TTS_BACKEND`, `TTS_TIMEOUT`,
`AUDIO_HOST`, `AUDIO_PORT`, `AUDIO_TTL`, `LOG_LEVEL`, `SYSTEM_PROMPT_PATH`,
`CONTEXT_DIR`. See `.env.example`.

## Conventions
- All mutable state goes under `data/`.
- All config comes from ENV / `.env` (see `.env.example`).
- Credentials / addresses of our own services that the user provides go ONLY into
  `.env` (never into code, never via inline env vars); read them through `Settings`.
- No default/example credentials in code; missing ENV var → fail at startup.
- A default address is allowed ONLY for public third-party APIs (Groq, OpenWeatherMap).
  Addresses of self-hosted services have no default.
- Code comments are in English.
- All repeated actions (env setup, tests, run) go through `make` targets — add or
  extend a target instead of running ad-hoc commands.
- Python always runs inside a local `.venv`, created automatically by `make` on
  first use (`make test` / `make run` bootstrap it) — never the system Python.
- Tests are required for new code; in CI `build` depends on `test`.
- No `EXPOSE` in the Dockerfile — the audio port is published to the LAN via compose `ports`.
