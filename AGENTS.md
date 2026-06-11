# Agent Instructions — zakhar-voice-assistant

Client-side voice assistant for a "HA Voice PE" ESP32 speaker. It connects to the
speaker over the ESPHome Native API (as the client), then runs the pipeline:
STT (cloud Whisper, default Groq) → LLM (default Claude Haiku 4.5 via OpenRouter) →
smart-home tools via an external MCP server →
TTS (local TeraTTS HTTP service), and serves the resulting audio back to N speakers.
It replaces Home Assistant in this loop. Smart-home control is an MCP integration:
the app is an MCP client (running an agentic tool-calling loop) that connects to an
external smart-home MCP server hosted in Node-RED (node-red-contrib-mcp-server)
configured under `core.mcp_servers` in `data/config.json`.

## Project structure
- `src/` — application code. Config core: `config_store.py` (atomic JSON load/save) +
  `config_service.py` (`ConfigService`/`ConfigDoc`) + `core_config.py` (`CoreConfig`).
  Stage providers live under `src/plugins/<stage>/<id>.py`, each with its own pydantic
  `ConfigModel` + `create()`; `src/plugins/base.py` has the `Provider`/`REGISTRY`.
  `src/reconfig.py` (+ `src/runtime.py`) own how config changes are applied at runtime:
  `reconfig.action_for` classifies each changed path and the `Reconfigurator` hot-reconfigures
  the running process (backends/tools/audio/devices/reminders) without a restart.
- `tests/` — pytest
- `data/` — runtime state: `config.json`, per-device context files, `prompts.db` (SQLite store of named system-prompt profiles; seeded once from the legacy `system_prompt.md`) (gitignored, docker volume)
- `templates/` — committed reference assets seeded into `data/` on first boot (`default_prompt.md`, `default_config.json`)
- `main.py` — thin entry point over `src/`

## Setup
All routine actions go through the `Makefile` — run `make help` to list targets.
```bash
make install           # create .venv and install dev/test deps
make config            # seed data/config.json from the template, then edit it
```

## Running tests
```bash
make test              # runs .venv/bin/pytest
```

## Running the app
```bash
make run               # runs .venv/bin/python main.py
```

## Configuration

All config lives in `data/config.json` (created on first boot from
`templates/default_config.json`). There is no `.env` / environment-based config.

- Per stage (`vad`/`stt`/`llm`/`tts`) the doc holds `{selected, instances}`; each provider
  defines its own settings via a pydantic `ConfigModel` in `src/plugins/<stage>/<id>.py`.
  The config core (`ConfigService`/`config_store`) is provider-agnostic — it never names a
  provider's fields. Adding a provider = one new plugin file; zero changes to the core.
- `core.*` holds non-provider settings (context, audio, vad — end-pointing thresholds +
  mic conditioning, network, openweathermap, mcp_servers,
  calendar, esphome, prompt, panel, runs, reminders, devices, tts_timeout, log_level) as plain
  `CoreConfig` pydantic sections.
- API keys are plain string fields in the JSON (trusted-LAN service, no masking). `data/`
  is gitignored so `config.json` never gets committed.

## Conventions
- All mutable state goes under `data/`.
- All config is in `data/config.json`, read via `ConfigService`; never hardcode config in code.
- A new provider is a self-contained plugin (`ConfigModel` + `create()`) registered with
  `@register`; do not teach the config core about specific provider fields.
- A default address is allowed ONLY for public third-party APIs (Groq, OpenRouter, OpenWeatherMap).
- Code comments are in English.
- All repeated actions (env setup, tests, run) go through `make` targets — add or
  extend a target instead of running ad-hoc commands.
- Python always runs inside a local `.venv`, created automatically by `make` on
  first use (`make test` / `make run` bootstrap it) — never the system Python.
- Tests are required for new code; in CI `build` depends on `test`.
- No `EXPOSE` in the Dockerfile — the audio port is published to the LAN via compose `ports`.
