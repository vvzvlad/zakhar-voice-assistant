# Device protocol

How a voice device (the Voice PE speaker, the espanel wall panel) talks to the zakhar server.
It replaces the ESPHome Native API: there is no ESPHome on either end.

## Decisions

The owner's words:
- «я просто хочу переделать прошивку на си без хоума» — the device firmware is plain ESP-IDF C.
- «ок, берем вебсокет, приступай» — the transport is WebSocket.
- «переводим панель конечно тоже, выкидываем весь еспхоум» — the panel moves to this protocol
  too; ESPHome leaves the server, the speaker and the panel.

## Transport

- The device is the client. It connects to `ws://<server>:8200/device` — the device port
  (`core.audio.port`, which also serves `/health`); the server URL is set in the firmware build.
- Text frames carry one JSON object each, with a `type` field. Unknown types and unknown fields
  are ignored by both sides.
- Binary frames carry audio. The first byte is the kind, the rest is the payload:
  - `0x01` — microphone audio, device → server;
  - `0x02` — playback audio, server → device.
- All audio is signed 16-bit little-endian PCM. No codec on either side.
- Liveness: the server sends a WebSocket ping every 10 s; a side that hears nothing for 30 s drops
  the connection. The device reconnects with a backoff of 1, 2, 4 … 30 s.
- No encryption, no authentication: the devices live on a trusted LAN.

## Connection

1. Device → `hello`:
   ```json
   {"type": "hello", "id": "main_room", "fw": "voicepe 0.1.0 (a1b2c3d)", "model": "zakhar_v28",
    "caps": {"speaker": true, "screen": false},
    "mic": {"channels": 2},
    "play": {"rate": 48000, "buffer_ms": 4000},
    "volume": 60}
   ```
   - `id` names the device in the server config (`core.devices[].name`) and keys its dialog
     context. An unknown `id` is added to the config as an enabled device.
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
  device flushes its playback, the server cancels the older run.
- A device without a speaker (`caps.speaker: false`) never gets playback: the server skips TTS for
  it and sends the text only.

## Playback

- `play_start {id}`, then `0x02` frames of up to 4096 payload bytes, then `play_end {id}`. `id` is
  a server counter. The audio is mono at `hello.play.rate`; the server resamples to it.
- The device plays streams back to back in arrival order (the end-of-phrase chime, a filler, the
  answer) and sends `played {id}` when the last sample of a stream is out.
- `play_stop` flushes everything queued and playing; the device answers `played {id, "dropped":
  true}` for each stream it dropped. A new `wake` flushes the same way.
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
| `capture {seconds}` | server → device | record a sample, see below |
| `audio_end {reason}` | device → server | the device stopped a stream on its own: `capture` (the sample is complete) or `busy` (refused, a session is running) |

`capture`: the device pauses wake-word detection, streams the microphone for `seconds` from the
moment of the command (no pre-roll) as `0x01` frames, then sends `audio_end {reason: "capture"}`
and resumes detection. It shows its own recording indication meanwhile.
