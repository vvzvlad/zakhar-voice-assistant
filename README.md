# zakhar-voice-assistant

*Also available in [Russian](README.ru.md).*

Voice assistant server for ESP32 voice devices — the "HA Voice PE" speaker and the
espanel wall panel — that replaces Home Assistant in the voice loop. The devices
connect to the server over its own WebSocket protocol
([docs/device-protocol.md](docs/device-protocol.md)) and the server runs the
pipeline: STT (cloud Whisper, default Groq) → LLM (default Claude Haiku 4.5 via
OpenRouter) → smart-home tools via an external MCP server → RuAccent (Russian
stress placement) → TTS (offline Piper / offline Silero / cloud Yandex SpeechKit /
Fish Audio), playing the reply back on the devices.

Smart-home control is an MCP integration: the app is an MCP client that connects
to an external smart-home MCP server (hosted in Node-RED via
node-red-contrib-mcp-server) configured under `core.mcp_servers` in
`data/config.json`, advertises its tools to the model and runs an agentic
tool-calling loop.

## Screenshots

The admin panel (served on `PANEL_PORT`, default 8201) is a single-page app: a live
pipeline overview, a searchable request log, and a per-run drill-down.

### Dashboard — pipeline overview

Service map of the three pipeline steps (Capture → Logic → Speech) with live metrics and
the most recent requests.

![Dashboard](docs/screenshots/dashboard.png)

### Request log

Every pipeline run with per-stage timing waterfalls, token counts and status, filterable
by text, device and result.

![Request log](docs/screenshots/request-log.png)

### Run detail

Click a row for the full stage timeline, the STT/LLM transcript, the captured utterance
audio and the per-round LLM tool calls.

![Run detail](docs/screenshots/run-detail.png)

## Quick start

Everything routine is wrapped in the `Makefile` (`make help` lists all targets):

```bash
make install                # create .venv + install dev/test deps
make config                 # seed data/config.json from the template, then edit it
make models                 # download the offline models (Vosk/Piper/Silero) into models/
make test                   # run tests
make run                    # run the app
```

Python targets (`make test`, `make run`) create and reuse a local `.venv`
automatically — you never need the system Python.

The offline backends need their model files under `models/`. `make models` fetches
them all; grab just one with `make models-vosk` / `make models-piper` /
`make models-silero-vad` / `make models-silero-tts` (each skips a file that is
already present). The Silero TTS voice runs on PyTorch — an optional dependency
deliberately NOT in `requirements.txt` or the Docker image — so install it
yourself (e.g. `pip install torch`) to use that provider.

Docker note: the published image ships WITHOUT the files of `models/` (the Dockerfile
neither downloads nor bundles them; it carries only the speaker's wake-word and VAD
`.tflite`), so `docker-compose.yml` mounts a second named
volume, `zakhar-voice-assistant-models`, at `/app/models` — and it has to be
populated. This is not optional the way "offline backends are optional" suggests:
`templates/default_config.json` selects the **Piper** TTS voice at
`models/ru_RU-ruslan-medium.onnx`, and `src/plugins/tts/piper.py` loads it EAGERLY
while `src/app.py` is still building backends — before the device server starts. A
container with an empty models volume therefore writes its first-boot config and
then exits with `FileNotFoundError`; it does not fall back to cloud-only. Run
`make models` on the host and copy the result into the volume, or switch
`tts.variants` to a cloud provider in `data/config.json` before the first boot.

## Deployment

Production runs the **prebuilt image** from
`gitea.vvzvlad.xyz/projects/zakhar-voice-assistant` — never build on the box. CI
builds the image on every push to `main`, smoke-tests it, and pushes it only if the
gate passes (`.gitea/workflows/image-check-publish.yml`); pull requests run the
identical tests, build and gate but never publish (`.gitea/workflows/tests.yml`).
`docker-compose.yml` in the repo root is the ready-to-use deploy template (pull,
don't build).

1. Copy `docker-compose.yml` to the host — it is the only file you need (the image
   carries the code; no checkout required on prod).
2. Pull and start:

   ```bash
   docker compose pull        # fetch the prebuilt image from the Gitea registry
   docker compose up -d       # start in the background
   ```

   The first boot seeds `data/config.json` from `templates/default_config.json`
   into the named volume.
3. Open the admin panel at `http://<host>:8201` and set providers, API keys and
   devices (or edit `data/config.json` directly inside the volume). All JSON
   settings apply live — no restart needed. See [Configuration](#configuration).

**Ports** (published straight to the LAN, no reverse proxy; the devices find the server by
its mDNS advertisement, which an mDNS repeater on the host relays from `docker_main_net` to
the LAN — set `core.audio.advertise_host` to the host's LAN address, since inside the bridge
the container sees only its own):

- `8201` — admin panel UI + API, and the agent MCP endpoint at `/mcp` (matches
  `PANEL_PORT`).
- `8200` — device server: the devices connect to `ws://<host>:8200/device`;
  `/health` is on the same port. **Must match `core.audio.port`** in
  `data/config.json` (default 8200).

**State** — all mutable state (config, per-device context, `prompts.db`) lives in
the `zakhar-voice-assistant` named volume mounted at `/app/data`, so it survives
restarts and image updates. A second volume, `zakhar-voice-assistant-models`, is
mounted at `/app/models` and holds the offline model files, which are not in the
image (see the Docker note under [Quick start](#quick-start)).

**Environment** — only `TZ` (container timezone) and the panel bind
`PANEL_HOST` / `PANEL_PORT` (defaults `0.0.0.0` / `8201`) are env-tunable;
everything else is in `data/config.json`.

**Image tags** — `:latest` (head of `main`) and `:<git-sha>` (pin an exact build;
swap the tag in `docker-compose.yml` to roll back).

**Updates** — the compose file carries the
`com.centurylinklabs.watchtower.enable=true` label, which opts this container in to
automatic redeployment when `:latest` moves. The key is Watchtower's for historical
reasons only: the label is read by Portainer's own native image updater (labelled
scope, five-minute poll, cleanup and rollback-on-failure), which understands it
alongside its own `io.portainer.update.enable` — so no Watchtower is involved.
Where nothing polls the registry, update manually:

```bash
docker compose pull && docker compose up -d
```

A built-in healthcheck probes `http://127.0.0.1:8200/health` every 30 s (`timeout:
5s`, `retries: 3`, `start_period: 40s`). The interval has to stay well under the
updater's `RollbackTimeout` (120 s on this fleet), because the updater judges the new
container by its health: a container whose first probe has not run yet is still
`starting` when that gate is decided, so the update gets rolled back on a service
that is perfectly fine. On Docker ≥ 25 the `start_period` also enables
`start_interval` (5 s by default), so the first probe lands seconds after start
rather than a whole interval later — in production the container reports healthy
about 6 s in.

> Offline STT/TTS models are **not** in the image; they live on the
> `zakhar-voice-assistant-models` volume mounted at `/app/models`. That volume must
> be populated — with the stock config the container does not boot without it (see
> the Docker note under [Quick start](#quick-start)).

## Configuration

All config lives in a single JSON file, `data/config.json` (created on first run
from `templates/default_config.json`; `make config` seeds it explicitly). The only
env-configurable values are `PANEL_HOST` / `PANEL_PORT` (see below); everything else
is in the JSON. ALL JSON settings apply LIVE (hot) the moment they are saved via the
panel/API: backends, tool sources, the device server, devices and reminders are
reconfigured in place — there is no restart concept in the panel. The one thing NOT
in the JSON config is the admin panel's own bind host/port: those come from the
`PANEL_HOST` / `PANEL_PORT` environment variables (defaults `0.0.0.0` / `8201`) and
are applied at process start.

Shape (see `templates/default_config.json` for the full default):

- **Stage providers** — each provider has one set of settings under `instances`:
  - `vad`: `webrtc` (WebRTC VAD) — aggressiveness; the end-pointing
    thresholds and the decision-only VAD auto gain (`mic_auto_gain`, in the core
    Voice-capture settings) stay in `core.vad`.
  - `stt`: `groq` (cloud Whisper) | `vosk` (offline) | `yandex` (SpeechKit v3 streaming, gRPC).
  - `llm` (handlers): `openrouter` | `groq` — model, api_key, temperature, max_tokens,
    max_tool_rounds; `simple-nlu` — the offline command engine (aliases, action verbs);
    with `decision_model` set (e.g. `typesafe/jev-1.14` on OpenRouter) a phrase it does
    not match and that has no number goes to that decision model, which picks the alias
    target and the verb.
  - Which STT feeds which handler is set by `core.chains`: an ordered list of
    `{name, enabled, stt, handler, narrow, parallel}`. One utterance goes through the
    enabled chains in order on the same audio until one answers; the next chain is tried
    when the STT fails or hears nothing, when the handler declines, or when it fails
    before running a tool. `narrow` restricts the chain's STT to its handler's vocabulary
    (Vosk + `simple-nlu`); `parallel` starts the chain's STT together with the first one.
    When another chain follows, `simple-nlu` runs strict: it executes a command only if
    every word is explained (alias, verb, number, filler word) and declines otherwise.
  - `stress`: `ruaccent` (offline Russian stress placement) — runs between LLM and TTS,
    marks the stressed vowels in the reply (the canonical `+vowel` text every TTS backend
    already adapts); `enabled` (on by default) + model size.
  - `tts`: `piper` (offline) | `silero` (offline, needs PyTorch) | `yandex` (SpeechKit) | `fishaudio` (cloud) — voice/key/etc. per provider.
    `tts.variants` is an ordered fallback list of `{provider, enabled}`: the next variant
    speaks when synthesis fails before the speaker receives audio.
  Each provider declares its own settings schema (`src/plugins/<stage>/<id>.py`); adding a
  provider is one file, no changes to the config core.
- **`core`** — non-provider settings: `context` (max_turns / ttl_seconds), `audio`
  (host / port of the device server), `vad` thresholds, `network.external_proxy`,
  `openweathermap` (api_key / city), `mcp_servers` (list of
  `{name, url, token, transport, prompt, slow}` — marking a server `slow` makes the
  assistant speak a short filler line before calling its tools),
  `calendar` (url / username / password / calendar), `prompt.system_prompt_path`
  (legacy seed only — prompts now live as named profiles in `data/prompts.db`),
  `agent_mcp` (enabled — the agent-facing MCP endpoint served by the panel at `/mcp`),
  `runs` (enabled / retention_days),
  `reminders` (enabled), `capture` (enabled / dir — save every utterance as WAV),
  `ack` (enabled / sound_path — the end-of-phrase chime), `devices` (list of
  `{name, label, enabled, wake_cutoff, vad_cutoff, wake_model, vad_model}`, see [Devices](#devices-and-firmware)),
  `tts_timeout`, `log_level`.

API keys are plain string fields in the JSON (this is a trusted-LAN service); `data/` is
gitignored, so `config.json` and its keys never get committed.

## Agent MCP server

The app is also an MCP **server**, so an external agent (e.g. Claude) can drive the
assistant. The endpoint is served by the admin panel itself:
`http://<host>:8201/mcp` (MCP streamable HTTP, on the panel port — `PANEL_PORT`
env, default 8201; no separate server or port). The on/off toggle is
`core.agent_mcp.enabled` in the JSON config, editable in the panel (System page)
and applied live per request. There is NO authentication (trusted-LAN service,
like the admin panel). Tools:

- `list_runs(limit, device?, search?)` — recent voice interactions (user text, reply, timings).
- `get_run(run_id)` — full record of one interaction (tool rounds, timings, errors).
- `get_config()` — the full live config document.
- `update_config(patch)` — deep-merge a config patch; changes apply live (hot-reload).
- `list_devices()` — configured speakers with live status.
- `say(text, device?)` — speak arbitrary text on a speaker (TTS, played as an announcement).
- `ask(text, device?, speak?)` — full assistant turn (LLM + smart-home tools +
  conversation context) as if the user spoke it; the reply is optionally spoken.

## Devices and firmware

The devices are the WebSocket clients: each one finds the server by mDNS (service
`_zakhar._tcp`, advertised by the server on its own network segment), connects to
`ws://<server>:8200/device` (the port is `core.audio.port`) and keeps the connection
up, reconnecting on its own. The contract is
[docs/device-protocol.md](docs/device-protocol.md). A device whose id is not in
`core.devices` yet is added there on its first connection; `enabled: false` refuses
it, and `wake_cutoff` / `vad_cutoff` (percent) are sent to it as its wake-word and
VAD thresholds.

Debug and dataset calls of the panel API (`http://<host>:8201`), meant for an agent with curl:

- `POST /api/device/detect` `{"device": "<name>", "enabled": false}` — turn the device's
  wake-word detection off (`true`: on). The model keeps running, so the wake probability
  still streams; detection is back on at the device's next connection.
- `POST /api/device/model` `{"device": "<name>"}` — send the device its wake-word and VAD
  models again, read from disk anew; the device swaps them without a restart.
- `curl -F file=@zakhar.tflite http://<host>:8201/api/wake-models` — upload a wake-word model
  as `uploaded/<stem>` (here `uploaded/zakhar`); devices already using that name get the new
  file at once.
- `POST /api/capture` `{"device": "<name>", "seconds": 60}` — record `seconds` (1–300) of
  the device's microphone into `data/captures/<device>/<YYYYmmdd-HHMMSS>-<id>.wav`. The
  server writes the audio as it arrives; a lost link or a server restart does not end the
  capture (docs/device-protocol.md, «Capture»), and audio that did get lost shows in `gaps`.
  An offline device answers 409 at once. `GET /api/capture?device=<name>` — its state
  (`starting` until the first audio, `recording`, `done`, `error`), countdown and gaps;
  `GET /api/capture/result?device=<name>` — its newest finished capture. Captures stay until
  deleted: `GET /api/captures` lists them, `GET /api/captures/<device>/<name>.wav` downloads
  one, `DELETE /api/captures/<device>/<name>` deletes it.
- `POST /api/datasets/<set>` `{"device": "<name>", "takes": 200, "seconds": 3}` — record a
  wake-word dataset: for each take the speaker beeps, then records `seconds` (1–10, default
  3) into `data/datasets/<set>/NNNN.wav`; a new session on the same set continues the
  numbering. A take that failed or lost audio is recorded again; while the device is offline
  the session waits for it (state `waiting`), and three failures in a row of one take end it.
  `GET /api/datasets` lists the sets with their clip counts and the session
  state, `POST /api/datasets/<set>/stop` stops after the current take,
  `GET /api/datasets/<set>.zip` downloads the set. Detection stays on between takes, so a
  word said after a take's window starts a voice session and the next take fails as busy —
  turn detection off for the session.

The Voice PE speaker firmware lives in `firmware/` — plain ESP-IDF C. Its
`components/zakhar_link` is the device side of the protocol and is shared with the
espanel wall panel. The firmware carries no server address — it finds the server by
mDNS, so one image serves any site; the device id is the speaker's Wi-Fi MAC (12 lowercase
hex), as with the panel, so one image gives every speaker its own id, and the name the
admin panel shows is the label set on the Devices page. The speaker gets its wake-word and VAD
models from the server at every connection: the release image carries every wake-word
model of `microWakeWord/` (the trained "Zakhar" versions and the ESPHome Okay Nabu, Hey
Jarvis, Hey Mycroft and Alexa) and the VAD model. A device's wake model is picked in its Edit
dialog on the Devices page (`core.devices[].wake_model`, empty for the release default);
the same dialog uploads any other `.tflite`, kept in `data/wake_models/` and listed as
`uploaded/<stem>`. `vad_model` points a device at another VAD file.
The firmware carries no Wi-Fi password: a speaker with no network stored shows a warm white twinkle and is set up with
Espressif's ESP BLE Provisioning app (iOS, Android), where it appears as
`PROV_ZAKHAR_<6 hex digits>` and asks for no code. Holding the center button for
10 s forgets the network and brings that back. The one-time switch from ESPHome
takes the image of `make fw-build-seed`: it is built with the network of
`firmware/sdkconfig.secrets` (gitignored, `CONFIG_ZAKHAR_WIFI_SSID` and
`CONFIG_ZAKHAR_WIFI_PASSWORD` lines) and stores it whenever no network is stored —
so on the seed image the button hold brings that same network back, not the app
setup; flash the normal image right after the switch.

```bash
make fw-build                       # build on the cirunner docker host into firmware/build/
make fw-build-seed                  # the same, plus the Wi-Fi of firmware/sdkconfig.secrets
make fw-ota HOST=<address>          # send it over Wi-Fi to a speaker running this firmware
make fw-flash PORT=/dev/cu.usbmodem…  # or over USB
make fw-logs HOST=<address>         # the speaker's live log over Wi-Fi
make fw-host-test fw-voicepe-host-test  # host tests of the pure C cores
```

## What's here

| Path | Purpose |
|------|---------|
| `Makefile` | Single entry point for repeated actions: `install`, `config`, `models`, `test`, `run`, `fw-*`. Run `make help`. |
| `src/` | Application code. Config core: `config_store.py` + `config_service.py`; providers under `src/plugins/`. |
| `tests/` | pytest suite (runs in CI before the image is built). |
| `docs/device-protocol.md` | The WebSocket protocol between the server and the devices. |
| `firmware/` | Voice PE speaker firmware (ESP-IDF C) and the shared `zakhar_link` component. |
| `microWakeWord/` | Wake-word model training and the deployed models. |
| `data/` | Runtime state: `config.json`, per-device context, `prompts.db` (named system-prompt profiles). Gitignored, mounted as a volume. |
| `models/` | Offline model files (Vosk / Piper / Silero), fetched by `make models`. Gitignored; not bundled in the Docker image. |
| `templates/` | Committed reference assets (`default_prompt.md`, `default_config.json`) seeded into `data/` on first boot. |
| `Dockerfile` | Two-stage build: a node stage builds the panel, the python stage carries only its `dist`; deps cached before code; no `EXPOSE`. |
| `docker-compose.yml` | Deploy template — image from the Gitea registry, `data` + `models` volumes, `docker_main_net` relayed to the LAN by the host's mDNS repeater, device port published to LAN, auto-update label. |
| `.gitea/workflows/` | CI: tests → build → smoke gate → push to `gitea.vvzvlad.xyz` (`<sha>` then `latest`). Pull requests run the same checks without publishing. |
| `ci/smoke.py` | The gate between build and push: image contract, version stamp, native VAD libs, frontend bundle, boot and `/health`. |
| `AGENTS.md` | Conventions / onboarding for agents. |

## Rules in one breath

All mutable state in `data/`; all config in `data/config.json` (providers self-describe
their settings; the config core never hardcodes provider fields); tests are mandatory and
gate the Docker build, and a smoke gate stands between the build and the registry; deploy a
prebuilt image from the Gitea registry via docker-compose. Unlike a public
web service, the device port is published straight to the LAN (no Traefik) and the host's
mDNS repeater relays the server's advertisement, so the devices can find it and connect.
