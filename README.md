# zakhar-voice-assistant

Client-side voice assistant for a "HA Voice PE" ESP32 speaker that replaces Home
Assistant in the voice loop. The server connects to the speaker over the ESPHome
Native API (as the client) and runs the pipeline: STT (cloud Whisper, default
Groq) → LLM (default Claude Haiku 4.5 via OpenRouter) → smart-home tools via an
external MCP server → TTS (local TeraTTS HTTP service), returning the generated
audio to N speakers.

Smart-home control is an MCP integration: the app is an MCP client that connects
to an external smart-home MCP server (hosted in Node-RED via
node-red-contrib-mcp-server) configured under `core.mcp_servers` in
`data/config.json`, advertises its tools to the model and runs an agentic
tool-calling loop.

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
from `templates/default_config.json`; `make config` seeds it explicitly). The only
env-configurable values are `PANEL_HOST` / `PANEL_PORT` (see below); everything else
is in the JSON. ALL JSON settings apply LIVE (hot) the moment they are saved via the
panel/API: backends, tool sources, the audio server, devices and reminders are
reconfigured in place — there is no restart concept in the panel. The one thing NOT
in the JSON config is the admin panel's own bind host/port: those come from the
`PANEL_HOST` / `PANEL_PORT` environment variables (defaults `0.0.0.0` / `8201`) and
are applied at process start.

Shape (see `templates/default_config.json` for the full default):

- **Stage providers** — for each stage you pick one provider and configure it:
  - `vad`: `webrtc` (WebRTC VAD) — aggressiveness / auto_gain; the end-pointing
    thresholds stay in `core.vad`.
  - `stt`: `groq` (cloud Whisper) | `vosk` (offline) — `selected` + per-provider `instances`.
  - `llm`: `openrouter` | `groq` — model, api_key, temperature, max_tokens, max_tool_rounds.
  - `tts`: `teratts` (HTTP) | `piper` (offline) | `yandex` (SpeechKit) — voice/key/etc. per provider.
  Each provider declares its own settings schema (`src/plugins/<stage>/<id>.py`); adding a
  provider is one file, no changes to the config core.
- **`core`** — non-provider settings: `context` (max_turns / ttl_seconds / dir), `audio`
  (host / port / ttl / public_base_url), `vad` thresholds, `network.external_proxy`,
  `openweathermap` (api_key / city), `mcp_servers` (list of
  `{name, url, token, transport, prompt, slow}` — marking a server `slow` makes the
  assistant speak a short filler line before calling its tools),
  `calendar` (url / username / password / calendar), `esphome.port`, `prompt.system_prompt_path`
  (legacy seed only — prompts now live as named profiles in `data/prompts.db`),
  `runs` (enabled / retention_days),
  `reminders` (enabled), `devices` (list of `{name, host, psk}`), `tts_timeout`, `log_level`.

API keys are plain string fields in the JSON (this is a trusted-LAN service); `data/` is
gitignored, so `config.json` and its keys never get committed.

## Agent MCP server

The app is also an MCP **server**, so an external agent (e.g. Claude) can drive the
assistant. Endpoint: `http://<host>:8202/mcp` (MCP streamable HTTP). Bind host/port
come from the `AGENT_MCP_HOST` / `AGENT_MCP_PORT` environment variables (defaults
`0.0.0.0` / `8202`), applied at process start — same env-only posture as the panel.
There is NO authentication (trusted-LAN service, like the admin panel). Tools:

- `list_runs(limit, device?, search?)` — recent voice interactions (user text, reply, timings).
- `get_run(run_id)` — full record of one interaction (tool rounds, timings, errors).
- `get_config()` — the full live config document.
- `update_config(patch)` — deep-merge a config patch; changes apply live (hot-reload).
- `list_devices()` — configured speakers with live status.
- `say(text, device?)` — speak arbitrary text on a speaker (TTS + announce).
- `ask(text, device?, speak?)` — full assistant turn (LLM + smart-home tools +
  conversation context) as if the user spoke it; the reply is optionally spoken.

## What's here

| Path | Purpose |
|------|---------|
| `Makefile` | Single entry point for repeated actions: `install`, `test`, `run`. Run `make help`. |
| `src/` | Application code. Config core: `config_store.py` + `config_service.py`; providers under `src/plugins/`. |
| `tests/` | pytest suite (runs in CI before the image is built). |
| `data/` | Runtime state: `config.json`, per-device context, `prompts.db` (named system-prompt profiles). Gitignored, mounted as a volume. |
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
