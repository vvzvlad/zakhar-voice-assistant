# Device protocol

How a voice device (the Voice PE speaker, the espanel wall panel) talks to the zakhar server.
It replaces the ESPHome Native API: there is no ESPHome on either end.

## Decisions

The owner's words:
- «я просто хочу переделать прошивку на си без хоума» — the device firmware is plain ESP-IDF C.
- «ок, берем вебсокет, приступай» — the transport is WebSocket.
- «переводим панель конечно тоже, выкидываем весь еспхоум» — the panel moves to this protocol
  too; ESPHome leaves the server, the speaker and the panel.
- «по воздуху конечно» — the speaker gets the new firmware over the air, the first time through
  the OTA of its current ESPHome firmware, not over USB.
- «сделай там логи по сети тока, как у еспхоума» — the speaker streams its log over Wi-Fi to
  whoever connects to it (TCP port 23, `make fw-logs HOST=<address>`), outside this protocol.
- «еще какой-то удобный интерфейс нужен для записи аудио, например если я хочу записать датасет
  захаров», option chosen: «По сигналу колонки» — a dataset take is a beep on the speaker, then a
  short `capture`; the clips go into a named set on the server and download as one archive.
  «фронт можно вокруг этого не дописывать, просто дать апи, один фиг я агента прошу всегда» —
  only the panel HTTP API (`/api/datasets`), no panel UI.
- «сделай команду отключения и включения, для дебага нужна» — `detect {on}` turns the device's
  wake-word detection off and on; panel API `POST /api/device/detect`.
- «надо сделать систему джойна в сеть, чтобы не надо было хардкодить пасс в прошивке», options
  chosen: «Приложение Espressif» — the speaker keeps its Wi-Fi in NVS and gets it over BLE from
  the «ESP BLE Provisioning» app; «Разовый образ с Wi-Fi» — only the image for the switch from
  ESPHome carries the Wi-Fi from the gitignored file and stores it on first boot, every later
  image has no password.
- «в прошивке вообще ничего не хранится, каждый старт грузим с сервера, все равно без сервера не
  имеет смысла колонка. что грузить — решает сервер. сервер может в любой момент взять и
  перезагрузить модель прямой командой без перезагрузки прошивки.» — the firmware carries no
  wake-word or VAD model; the server sends both at every connection and whenever it decides to,
  and the device swaps them without a restart.
- «файлы моделей должны приходить с релизом, внутри него», «всм просто с релизом» — the model
  files ship inside the server's release image, not on the models volume.
- «теперь надо сделать так чтобы я мог выбирать в панели нужную модель» — the panel's device
  dialog picks the device's wake-word model from the models the release carries.
- «надо сделать чтобы можно было через интерфейс загрузить любую модель вручную» — the same
  dialog uploads a `.tflite`: it is kept in `data/wake_models/` on the data volume, listed as
  `uploaded/<stem>`, and a new file of the same name replaces it and is sent at once to every
  device using it.
- «еще модели из esphome же можно притащить, джарвис, набу», option chosen: «Просто добавить в
  список» — the ESPHome models (Okay Nabu, Hey Jarvis, Hey Mycroft, Alexa) are in the list; the
  server's second wake-word check stays as it is, so wakes on them pass only with that check off.
- «надо как-то сделать так, чтобы прошивка колонки была универсальной. настройка вафай там уже
  есть, надо что-то сделать с сервером», option chosen: «делаем первый, ок» — the firmware carries
  no server address: the device finds the server by mDNS and the server advertises itself.
- «поставь транслятор» (instead of the container on the host network) — the server stays in the
  Docker bridge network `docker_main_net`, the mDNS repeater on its host relays that network to
  the LAN, and the address it advertises is set in `core.audio.advertise_host`.

## Transport

- The device is the client. Before every connection it browses mDNS for the service
  `_zakhar._tcp` and connects to `ws://<address>:<port>/device` of the first answer; when that
  answer came without the address, it asks for the A record of the answer's host by name, with a
  multicast answer (QM: a repeater relays multicast only). While
  nothing answers it retries with its reconnect backoff. The server advertises the instance and
  host `zakhar-<short host name>` (`zakhar-<short host name>.local`; in Docker the container's
  host name, set by compose `hostname:`) with the device port
  (`core.audio.port`, which also serves `/health`) and one IPv4 address: `core.audio.advertise_host`
  when set, else `core.audio.host` when it is a concrete address, otherwise the address the
  machine reaches the gateway of its main routing table from (a full-tunnel VPN keeps its default
  route elsewhere). In a Docker bridge network the container sees only its own address and the
  host's mDNS repeater relays its packets unchanged, so `advertise_host` holds the host's LAN
  address. mDNS does not cross subnets, so a device finds the server of its own network segment.
- Text frames carry one JSON object each, with a `type` field. Unknown types and unknown fields
  are ignored by both sides.
- Binary frames carry audio and models. The first byte is the kind, the rest is the payload:
  - `0x01` — microphone audio, device → server;
  - `0x02` — playback audio, server → device;
  - `0x03` — model bytes, server → device (see [Models](#models)).
- All audio is signed 16-bit little-endian PCM. No codec on either side. The microphone runs at
  16 kHz.
- Liveness: the server sends a WebSocket ping every 10 s and drops the connection when the pong
  does not come within 5 s; the device drops it when it has heard nothing for 30 s. The device
  reconnects with a backoff of 1, 2, 4 … 30 s.
- No encryption, no authentication: the devices live on a trusted LAN.

## Connection

1. Device → `hello`:
   ```json
   {"type": "hello", "id": "fc012cd1e5ec", "fw": "voicepe 0.1.0 (a1b2c3d)",
    "caps": {"speaker": true, "screen": false},
    "mic": {"channels": 2},
    "play": {"rate": 48000, "buffer_ms": 4000},
    "volume": 60}
   ```
   - `id` is the device's Wi-Fi MAC as 12 lowercase hex. It names the device in the server
     config (`core.devices[].name`) and keys its dialog context; the admin panel shows
     `core.devices[].label` instead when one is set. An unknown `id` is added to the config as
     an enabled device.
   - `play` and `volume` are present only when `caps.speaker` is true.
2. The server closes the connection with code 4000 when the device is disabled in the config, and
   closes an older connection of the same `id` with code 4001 when a new one says `hello`.
3. Server → `config`, right after `hello` and again whenever the values change:
   ```json
   {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30}
   ```
   Both are percent, 0..100. The device applies them as `uint8 = trunc(percent / 100 * 255)` to the
   wake-word model and to the on-device VAD gate. They live in the server config, not on the
   device; until the first `config` the device uses its compiled-in defaults.
4. Server → both models, right after `config`: see [Models](#models).

## Models

The device stores no model. The server sends the wake-word model and the VAD model that gates it
at every connection, again when the model configured for the device changes, on the direct
reload command (panel API `POST /api/device/model {device}`), and when a wake-word model file the
device uses is uploaded anew (panel API `POST /api/wake-models`). Which file goes out is the
server's choice: `core.devices[].wake_model` / `vad_model`, empty for the models the release
ships.

1. Server → `model`, then the bytes:
   ```json
   {"type": "model", "kind": "wake", "name": "zakhar/zakhar_v31", "size": 77848}
   ```
   `kind` is `wake` or `vad`, `name` is the model's name, opaque to the device, and `size` (> 0)
   is the length of the .tflite that follows in `0x03` frames of at most 4096 payload bytes.
   `model` carries no `session`, so the session filter lets it through. A new `model` while an
   earlier one is still arriving discards the earlier one; bytes past `size` are discarded.
2. Device → `model_loaded`, once all the bytes are in and the load was tried:
   ```json
   {"type": "model_loaded", "kind": "wake", "name": "zakhar/zakhar_v31", "ok": true}
   ```
   `ok` is false when the device could not allocate the buffer, the model did not load, or the
   device takes no models. A model that did not load leaves its kind with no model; when the buffer
   could not be allocated, the previous model stays. The server
   shows the `name` of the last wake model loaded as the device's model.

- The device swaps a model live, without a restart, and detects nothing until both are loaded.
- A device that does not know `model` (an older firmware) ignores it and its `0x03` frames.

## Voice session

```
device                                   server
  wake {session, word}            ──►
  0x01 mic: pre-roll burst, then live ─►
                                  ◄──    state listening        (wake word confirmed)
                                  ◄──    state transcribing     (end of phrase: stop the mic)
                                  ◄──    transcript {text}
                                  ◄──    state thinking
                                  ◄──    reply {text}
                                  ◄──    state speaking
                                  ◄──    play_start / 0x02 … / play_end
  played {id}                     ──►
                                  ◄──    state idle             (or state error {text})
```

- `wake`: `{"type": "wake", "session": 17, "word": "Захар"}`. `session` is a device counter, +1 for
  every wake. Right after `wake` the device streams the microphone: first the pre-roll (the audio
  before the detection, 2000 ms on the Voice PE) as fast as it can, then live audio in 32 ms
  frames (512 samples per channel, interleaved when `mic.channels` is 2). On the Voice PE
  channel 0 is the XMOS AGC output and channel 1 the XMOS noise-suppressed output.
- The device keeps streaming until a `state` of this session other than `listening` arrives. The
  end of the phrase is found by the server's VAD; the device has no end-pointing.
- `state`: `{"type": "state", "session": 17, "value": "listening"}`. Values: `listening`,
  `transcribing`, `thinking`, `speaking`, `idle`, `error` (`error` carries `text`). The device
  draws its indication from these only. Between `wake` and `listening` it shows nothing: the
  server may still reject the wake word with `state idle`.
- `transcript` and `reply` carry the recognised phrase and the answer text, for devices with a
  screen. `reply` without `session` is an announcement (a reminder, an agent's `speak`).
- Every server message of a session carries its `session`; the device drops messages whose
  `session` is not its current one. A new `wake` replaces the current session on both sides: the
  device flushes its playback, the server cancels the older run and, before anything else of the
  new session, sends `play_stop` — a stream the server started before it saw the `wake` is
  dropped on the device too.
- A device without a speaker (`caps.speaker: false`) never gets playback: the server skips TTS for
  it and sends the text only.

## Playback

- `play_start {id}`, then `0x02` frames of up to 4096 payload bytes, then `play_end {id}`. `id` is
  a server counter. The audio is mono at `hello.play.rate`; the server resamples to it.
- The device plays streams back to back in arrival order (the end-of-phrase chime, a filler, the
  answer) and sends `played {id}` when the last sample of a stream is out.
- `play_stop` flushes everything queued and playing; the device answers `played {id, "dropped":
  true}` for each stream it dropped. A new `wake` flushes the same way. `0x02` frames and
  `play_end` of a dropped stream that are still on the way are discarded.
- The server stops waiting for `played` of a stream after its duration plus `buffer_ms` plus 5 s
  and treats it as dropped.
- Pacing: the server keeps at most `buffer_ms − 500` ms of audio ahead of the device's playback,
  estimating that playback runs at real time from the first frame it sends. The device buffer
  holds `buffer_ms`.
- Indication while playing outside a session (an announcement) is the device's own choice.

## Settings and diagnostics

| message | direction | meaning |
|---|---|---|
| `set_volume {value}` | server → device | speaker volume, percent; the device stores it itself |
| `volume {value}` | device → server | the volume changed on the device (the dial) or after `set_volume` |
| `wake_peak_stream {on}` | server → device | start or stop the wake probability stream |
| `wake_peak {value}` | device → server | once a second while the stream is on: the highest wake-word probability of that second, percent |
| `detect {on}` | server → device | wake-word detection on or off; off: the models keep running (`wake_peak` too) but a detection starts no session. Every new connection starts with detection on |
| `capture {seconds}` | server → device | record a sample, see below |
| `audio_end {reason}` | device → server | the device stopped a stream on its own: `capture` (the sample is complete) or `busy` (refused, a session is running) |

`capture`: the device pauses wake-word detection, streams the microphone for `seconds` from the
moment of the command (no pre-roll) as `0x01` frames, then sends `audio_end {reason: "capture"}`
and detection returns to what `detect` set. It shows its own recording indication meanwhile.
