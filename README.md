# zakhar-voice-assistant

Client-side voice assistant for a "HA Voice PE" ESP32 speaker that replaces Home
Assistant in the voice loop. The server connects to the speaker over the ESPHome
Native API (as the client) and runs the pipeline: STT (Groq Whisper) → LLM
(Groq chat) → smart-home tools via an external MCP server → TTS (local TeraTTS
HTTP service), returning the generated audio to N speakers.

Smart-home control is an MCP integration: the app is an MCP client that connects
to an external smart-home MCP server (hosted in Node-RED via
node-red-contrib-mcp-server) at `MCP_SMARTHOME_URL`, advertises its tools to the
model and runs an agentic tool-calling loop.

## Quick start

Everything routine is wrapped in the `Makefile` (`make help` lists all targets):

```bash
make install                # create .venv + install dev/test deps
cp .env.example .env        # fill in the values
make test                   # run tests
make run                    # run the app
```

Python targets (`make test`, `make run`) create and reuse a local `.venv`
automatically — you never need the system Python.

## Environment

App, required (no default — missing → fail at startup): `GROQ_API_KEY`,
`WEATHER_API_KEY`, `MCP_SMARTHOME_URL` (external smart-home MCP server endpoint —
node-red-contrib-mcp-server in Node-RED, e.g. `http://10.0.0.5:8001/mcp`),
`TTS_BASE_URL`, `PUBLIC_BASE_URL`, `ESPHOME_DEVICES`
(`name|host|psk;name2|host2|psk2`).

App, optional (defaults in code): `MCP_SMARTHOME_TOKEN` (bearer for the MCP
server; empty = no auth), `ESPHOME_PORT`, `GROQ_MODEL`, `GROQ_STT_MODEL`,
`GROQ_PROXY`, `WEATHER_CITY`, `TTS_BACKEND`, `TTS_TIMEOUT`, `AUDIO_HOST`,
`AUDIO_PORT` (default 8200), `AUDIO_TTL`, `LOG_LEVEL`, `SYSTEM_PROMPT_PATH`,
`CONTEXT_DIR`. See `.env.example` for the full list.

## What's here

| Path | Purpose |
|------|---------|
| `Makefile` | Single entry point for repeated actions: `install`, `test`, `run`. Run `make help`. |
| `src/` | Application code; `settings.py` reads all config from ENV / `.env`. |
| `tests/` | pytest suite (runs in CI before the image is built). |
| `data/` | Runtime state (per-device context, cached prompt). Gitignored, mounted as a volume. |
| `templates/` | Static assets baked into the image (`default_prompt.md`). |
| `Dockerfile` | Slim single-stage build; deps cached before code; no `EXPOSE`. |
| `docker-compose.yml` | Deploy template — image from `ghcr.io`, volume, audio port published to LAN, watchtower label. |
| `.env.example` | Full list of env vars with placeholders. Copy to `.env`. |
| `.github/workflows/` | CI: `test` → `build` → push to `ghcr.io` (`latest` + sha). |
| `AGENTS.md` | Conventions / onboarding for agents. |

## Rules in one breath

All mutable state in `data/`; all config and credentials from ENV / `.env` (never
in code, never defaulted); tests are mandatory and gate the Docker build; deploy a
prebuilt `ghcr.io` image via docker-compose. Unlike a public web service, the audio
port is published straight to the LAN (no Traefik) so speakers can fetch audio.
