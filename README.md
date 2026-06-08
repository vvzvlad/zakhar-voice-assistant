# zakhar-voice-assistant

Client-side voice assistant for a "HA Voice PE" ESP32 speaker that replaces Home
Assistant in the voice loop. The server connects to the speaker over the ESPHome
Native API (as the client) and runs the pipeline: STT (cloud Whisper, default
Groq) → LLM (default Claude Haiku 4.5 via OpenRouter) → smart-home tools via an
external MCP server → TTS (local TeraTTS HTTP service), returning the generated
audio to N speakers.

Smart-home control is an MCP integration: the app is an MCP client that connects
to an external smart-home MCP server (hosted in Node-RED via
node-red-contrib-mcp-server) at `MCP_SMARTHOME_URL`, advertises its tools to the
model and runs an agentic tool-calling loop.

## Quick start

Everything routine is wrapped in the `Makefile` (`make help` lists all targets):

```bash
make install                # create .venv + install dev/test deps
make config                 # seed data/config.json from the template, then edit it
make test                   # run tests
make run                    # run the app
```

Python targets (`make test`, `make run`) create and reuse a local `.venv`
automatically — you never need the system Python.

## Configuration

All config lives in a single JSON file, `data/config.json` (created on first run
from `templates/default_config.json`; `make config` seeds it explicitly). There is
no `.env` — edit the JSON and restart.

Shape (see `templates/default_config.json` for the full default):

- **Stage providers** — for each stage you pick one provider and configure it:
  - `stt`: `groq` (cloud Whisper) | `vosk` (offline) — `selected` + per-provider `instances`.
  - `llm`: `openrouter` | `groq` — model, api_key, temperature, max_tokens, max_tool_rounds.
  - `tts`: `teratts` (HTTP) | `piper` (offline) | `yandex` (SpeechKit) — voice/key/etc. per provider.
  Each provider declares its own settings schema (`src/plugins/<stage>/<id>.py`); adding a
  provider is one file, no changes to the config core.
- **`core`** — non-provider settings: `context` (max_turns / ttl_seconds / dir), `audio`
  (host / port / ttl / public_base_url), `vad` thresholds, `network.external_proxy`,
  `weather` (api_key / city), `mcp` (url / token), `esphome.port`, `prompt.system_prompt_path`,
  `devices` (list of `{name, host, psk}`), `tts_timeout`, `log_level`.

API keys are plain string fields in the JSON (this is a trusted-LAN service); `data/` is
gitignored, so `config.json` and its keys never get committed.

## What's here

| Path | Purpose |
|------|---------|
| `Makefile` | Single entry point for repeated actions: `install`, `test`, `run`. Run `make help`. |
| `src/` | Application code. Config core: `config_store.py` + `config_service.py`; providers under `src/plugins/`. |
| `tests/` | pytest suite (runs in CI before the image is built). |
| `data/` | Runtime state: `config.json`, per-device context, cached prompt. Gitignored, mounted as a volume. |
| `templates/` | Committed reference assets (`default_prompt.md`, `default_config.json`) seeded into `data/` on first boot. |
| `Dockerfile` | Slim single-stage build; deps cached before code; no `EXPOSE`. |
| `docker-compose.yml` | Deploy template — image from `ghcr.io`, volume, audio port published to LAN, watchtower label. |
| `.github/workflows/` | CI: `test` → `build` → push to `ghcr.io` (`latest` + sha). |
| `AGENTS.md` | Conventions / onboarding for agents. |

## Rules in one breath

All mutable state in `data/`; all config in `data/config.json` (providers self-describe
their settings; the config core never hardcodes provider fields); tests are mandatory and
gate the Docker build; deploy a prebuilt `ghcr.io` image via docker-compose. Unlike a public
web service, the audio port is published straight to the LAN (no Traefik) so speakers can
fetch audio.
