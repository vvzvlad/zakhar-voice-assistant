# Agent Instructions — zakhar-voice-assistant

Server-side voice assistant for ESP32 voice devices (the "HA Voice PE" speaker, the wall
panel). Devices connect to it over the WebSocket device protocol (`docs/device-protocol.md`,
`src/device_server.py` on `core.audio.port`), then it runs the pipeline:
STT (cloud Whisper, default Groq) → LLM (default Claude Haiku 4.5 via OpenRouter) →
smart-home tools via an external MCP server → RuAccent (Russian stress placement) →
TTS (offline Piper / cloud Yandex SpeechKit / Fish Audio), and plays the resulting audio back on the devices.
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
  `src/agent_mcp.py` — the agent-facing MCP SERVER (FastMCP over streamable HTTP,
  served by the admin panel at `/mcp` on the panel port): external agents read the
  run log, view/patch the config, speak text, set a speaker's volume and send full
  text requests through the pipeline.
- `frontend/` — Vite/React admin panel (vitest tests in `frontend/src/__tests__/`).
- `tests/` — pytest
- `ci/smoke.py` — smoke gate run by CI against the built image; read its module docstring
  before changing it.
- `docs/device-protocol.md` — the normative server↔device contract; a change to the protocol
  changes this file, `src/device_server.py` and `firmware/components/zakhar_link` together.
  It states the contract as it is, with no decisions log or history: the owner's words behind a
  change go into its commit message.
- `firmware/` — Voice PE speaker firmware, ESP-IDF C; the Music Assistant player
  (`main/music.cpp`, sendspin-cpp over the Sendspin protocol) is the only C++. Internal RAM is
  what runs out first: every `operator new` goes to PSRAM (`main/psram_new.cpp`), libraries'
  `.bss` too, a plain `malloc()` of 4 KB and more as well (`CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL`),
  and the link needs a free 6 KB internal block for its WebSocket task at every connect; the
  server logs a warning when a device's `stats` shows a largest block under 8 KB. Add nothing
  resident to the firmware while issue #18 (the speaker drops parts of its answer) is open; after
  adding anything resident, check free/largest in the boot line `music: player up`
  (serial console, or the port-23 network log held open across the reboot; the server's device
  log starts after it).
  `components/zakhar_link` is the
  device side of the protocol; the espanel wall panel takes it as an ESP-IDF component manager
  git dependency pinned to a commit of this repo, so a protocol change reaches the panel by
  moving that pin.
- `microWakeWord/` — wake-word model training research and the deployed models.
- `data/` — runtime state: `config.json`, per-device context files, `prompts.db` (SQLite store of named system-prompt profiles), `captures/` and `datasets/` (device recordings, `src/captures.py`), `device_logs/` (the speakers' own logs, `src/device_logs.py`) (gitignored, docker volume)
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

## Firmware
```bash
make fw-build                          # idf.py in docker on cirunner, never on the laptop
make fw-build-seed                     # one-time ESPHome switch image, Wi-Fi from sdkconfig.secrets
make fw-host-test fw-voicepe-host-test # host tests of the pure C cores
make fw-ota HOST=<address>             # fw-flash PORT=… over USB, fw-logs HOST=… for the log
```

Build the speaker firmware through `make fw-build`: it keeps ccache and the downloaded components in
the cirunner volume `zakhar-fw-cache`, and a rebuild takes 1–1.5 min instead of 4+ from scratch.
A hand-rolled `docker run` of `espressif/idf` (a detached container, the panel's firmware) mounts
the same `-v zakhar-fw-cache:/root/.cache` and is not pinned to CPUs (all 72 are allowed):
`firmware/CMakeLists.txt` caps a build at 24 compiles, which a cold cache needs to fit in
cirunner's 19 GB.

## CI and deploy
- `.gitea/workflows/tests.yml` (pull requests): pytest in `python:3.11-slim`, image build,
  `ci/smoke.py`; never pushes.
- `.gitea/workflows/image-check-publish.yml` (push to `main`): the same steps, then pushes
  `:<sha>` and `:latest` to `gitea.vvzvlad.xyz`. Production follows `:latest` via Portainer's
  image updater (`docker-compose.yml`).
- The image carries none of the server's own models (STT, TTS, server VAD); production mounts
  them on the `/app/models` volume.
  It does carry the speaker's wake-word, stop-word and VAD `.tflite` files, which the server sends to the
  devices (`Dockerfile`, `src/device_server.py`).

## Configuration

All config lives in `data/config.json` (created on first boot from
`templates/default_config.json`). There is no `.env` / environment-based config —
the only env-tunable values are the bind host/port of the admin panel
(`PANEL_HOST` / `PANEL_PORT`, defaults `0.0.0.0` / `8201`), applied at process
start. The agent MCP endpoint rides the panel server at `/mcp` (same port); its
only setting is `core.agent_mcp.enabled`, panel-editable and read live per request.

- `vad`/`wakeword`/`stress`/`followup` hold `{selected, instances}`; `stt`/`llm` hold only `{instances}` —
  which STT feeds which handler is `core.chains`, an ordered list of `{name, enabled, stt,
  handler, narrow, parallel}` tried in turn on the same audio (`src/chains.py`); `tts` holds
  `{variants, instances}`, an ordered fallback list. Each provider
  defines its own settings via a pydantic `ConfigModel` in `src/plugins/<stage>/<id>.py`.
  The config core (`ConfigService`/`config_store`) is provider-agnostic — it never names a
  provider's fields. Adding a provider = one new plugin file; zero changes to the core.
- `core.*` holds non-provider settings (context, audio, vad — end-pointing thresholds +
  mic conditioning, network, openweathermap, mcp_servers,
  calendar, assistant_tasks, agent_mcp, prompt, runs, reminders, ack, devices, commands —
  the command catalog both command handlers read (helpers in `src/commands.py`), telemetry —
  the MQTT broker the server publishes its and the devices' telemetry to (topics in
  `src/telemetry.py`), tts_timeout, log_level, device_log_days) as plain `CoreConfig` pydantic
  sections.
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
- No `EXPOSE` in the Dockerfile — the device port is published to the LAN via compose `ports`;
  the host's mDNS repeater relays `docker_main_net` to the LAN, and `core.audio.advertise_host`
  holds the host's LAN address the server advertises.

## Limits for agents
- Tests: at most three times the code (`wc -l` of `tests/` against the code). Over the limit,
  still write the test you need, and in the same change go through the old tests: drop or merge
  duplicates and tests of implementation rather than behaviour.
- Comments explain the code as it is and why it is built this way. History ("used to", "after
  the review", "since the incident") goes into the commit message.
- This file stays under ~150 lines: only what an agent would get wrong without it. No decision
  history, incident logs or retelling of the code.

## FPF

Read a pattern through the `fpf` MCP (`read_pattern <ID>`) before applying it.

- Changing the system-prompt profile, the LLM, an STT/TTS/VAD provider, an MCP server's tools or `max_tool_rounds` — SYSE.46 "Test an Agent and Its Support in Representative Work": replay real requests from the run log (the agent MCP serves them with rounds and stage timings), include ones that must end in a refusal or a clarifying question, and compare the tool calls and their effects, not only the spoken reply.
