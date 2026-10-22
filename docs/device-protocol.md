# Device protocol

How a voice device (the Voice PE speaker, the espanel wall panel) talks to the zakhar server.
It replaces the ESPHome Native API: there is no ESPHome on either end.

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
  - `0x01` — microphone audio of a voice session, device → server;
  - `0x02` — playback audio, server → device;
  - `0x03` — model bytes, server → device (see [Models](#models));
  - `0x04` — microphone audio of a capture, device → server (see [Capture](#capture)).
- All audio is signed 16-bit little-endian PCM. No codec on either side. The microphone runs at
  16 kHz.
- Liveness: the server sends a WebSocket ping every 10 s and drops the connection when the pong
  does not come within 5 s; the device drops it when it has heard nothing for 30 s. The device
  reconnects with a backoff of 1, 2, 4 … 30 s.
- No encryption, no authentication: the devices live on a trusted LAN.

## Connection

1. Device → `hello`:
   ```json
   {"type": "hello", "id": "fc012cd1e5ec", "fw": "voicepe 0.1.0 (a1b2c3d)", "uptime": 3725,
    "reset": "poweron",
    "caps": {"speaker": true, "screen": false, "follow_up": true},
    "mic": {"channels": 2},
    "play": {"rate": 48000, "buffer_ms": 4000},
    "volume": 60}
   ```
   - `id` is the device's Wi-Fi MAC as 12 lowercase hex. It names the device in the server
     config (`core.devices[].name`) and keys its dialog context; the admin panel shows
     `core.devices[].label` instead when one is set. An unknown `id` is added to the config as
     an enabled device.
   - `uptime` is the whole seconds since the device booted, taken when it sends the `hello`;
     the server counts the device's uptime on from it.
   - `reset` is why the device last booted: `poweron`, `ext`, `sw`, `panic`, `int_wdt`,
     `task_wdt`, `wdt`, `deepsleep`, `brownout`, `sdio`, `usb`, `jtag`, `efuse`, `pwr_glitch`,
     `cpu_lockup` or `unknown` (ESP-IDF's `esp_reset_reason()`).
   - `play` and `volume` are present only when `caps.speaker` is true.
   - `caps.follow_up` is true when the device reopens its microphone on a follow-up `listening`
     (see [Follow-up](#follow-up)); the server listens for a follow-up only on such a device.
   - `capture` (the running capture's `id`) is present only while the device holds a capture,
     see [Capture](#capture).
2. The server closes the connection with code 4000 when the device is disabled in the config, and
   closes an older connection of the same `id` with code 4001 when a new one says `hello`.
3. Server → `config`, right after `hello` and again whenever the values change:
   ```json
   {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30, "play_prime_ms": 700, "auto_volume": false}
   ```
   The cutoffs are percent, 0..100. The device applies them as `uint8 = trunc(percent / 100 * 255)`
   to the wake-word model and to the on-device VAD gate.

   `auto_volume`: true has the device measure the room's noise and send `ambient`, see
   [Auto volume](#auto-volume); default false.

   `play_prime_ms`, 0..2000, default 700, is the playback reserve: the audio the device holds
   before it starts a stream, and again after the stream runs dry (see [Playback](#playback)); a
   device without a speaker ignores it. A device that does not measure ignores `auto_volume`.

   Every field is required, and the server sends all of them to every device: a `config` that
   lacks one, or holds one out of its range or of the wrong type, is ignored whole. These values
   live in the server config (`core.devices[]`, whose defaults are the ones named above), not on
   the device, which keeps none of them across a restart and does nothing that depends on them
   before its first `config`: no wake word is detected, a `play_start` is answered with
   `played {id, "dropped": true}` and auto volume does not measure.
4. Server → `volume_boost {value}` right after `config`: 0, a new connection has no `ambient` yet;
   see [Auto volume](#auto-volume).
5. Server → `log_stream {on}` with the device's `core.devices[].log`, on or off, after
   `volume_boost` (and after what a capture the device holds needs, see [Capture](#capture)): see
   [Settings and diagnostics](#settings-and-diagnostics).
6. Server → the wake-word and VAD models, right after `log_stream`: see [Models](#models).

## Models

The device stores no model. The server sends the wake-word model and the VAD model that gates it
at every connection, again when the model configured for the device changes, on the direct reload command (panel API `POST /api/device/model {device}`), and when a
wake-word model file the device uses is uploaded anew (panel API `POST /api/wake-models`). Which
file goes out is the server's choice: `core.devices[].wake_model` / `vad_model`, empty for the
models the release ships.

1. Server → `model`, then the bytes:
   ```json
   {"type": "model", "kind": "wake", "name": "zakhar/zakhar_v31", "size": 77848}
   ```
   `kind` is `wake` or `vad`, `name` is the model's name, opaque to the device, and `size`
   (> 0) is the length of the .tflite that follows in `0x03` frames of at most 4096 payload bytes.
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

- The device swaps a model live, without a restart. It detects the wake word only while the wake
  and VAD models are loaded.

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
  played {id, dry, gap_ms}        ──►
                                  ◄──    state idle             (or state error {text})
                                         or, with follow-up on, state listening (see Follow-up)
```

- `wake`: `{"type": "wake", "session": 17, "word": "Захар"}`. `session` is the number the device
  proposes for the session the wake would start: a device counter, +1 for every wake, never
  reused. The device stays on its current session until the server takes the proposal (below).
  Right after `wake` the device streams the microphone: first the pre-roll (the 2000 ms
  of audio before the detection; the server's end-pointing takes the wake word to end where the
  pre-roll ends) as fast as it can, then live audio in 32 ms frames (512 samples per channel,
  interleaved when `mic.channels` is 2). On the Voice PE
  channel 0 is the XMOS AGC output and channel 1 the XMOS noise-suppressed output at the wake-word
  gain ×4 (applied before the cut to 16 bits, saturating): exactly the samples the wake-word model
  gets. During an auto volume measurement channel 0 is the raw microphone 0 as read, at gain 1;
  the switch each way takes the device two 32 ms frames, and a wake ends the measurement and
  switches it back (see [Auto volume](#auto-volume)).
- The device keeps streaming until the server refuses the wake or, once it takes it, until a
  `state` of this session other than `listening` arrives. The end of the phrase is found by the
  server's VAD; the device has no end-pointing.
- `state`: `{"type": "state", "session": 17, "value": "listening"}`. Values: `listening`,
  `transcribing`, `thinking`, `speaking`, `idle`, `error` (`error` carries `text`). The device
  draws its indication from the states of its current session only: between `wake` and the
  server's ruling it goes on showing what the current session shows, a running answer included.
- `transcript` and `reply` carry the recognised phrase and the answer text, for devices with a
  screen. `reply` without `session` is an announcement (a reminder, an agent's `speak`).
- Every server message of a session carries its `session`; the device drops messages whose
  `session` is neither its current one nor the one it proposed, except `play_start` and
  `play_end`, which it takes whatever their `session`: only `play_stop` drops the playback, and a
  wake the server has not taken leaves it playing.
  While the microphone streams (from `wake` to the server's ruling, and in a session from its
  `listening` or a follow-up `listening` to the next `state` other than `listening`) a wake word
  proposes no new session: the phrase is still being said, and a detection inside it must not
  cut it off.
- After that a new `wake` proposes the next session. A proposal changes only the microphone: its
  audio streams, while the current session, its playback and the indication stay as they are. A
  `wake` over a running answer (a run of the device is going or a stream is playing) may be the
  device's own voice: the server holds it until its second wake-word check confirms it on the
  proposal's microphone audio (at once when that check is off), and the older run goes on
  meanwhile; the check also tells apart words that sound like the wake word (a fixed list in
  the Vosk check, `ANSWER_SOUND_ALIKES`). Any other `wake` is confirmed at once. A confirmed wake is
  taken: the server cancels the older run and sends `play_stop`, which flushes the device's
  playback, a stream the server started before it saw the `wake` included, then `state listening`
  with the proposed `session`, which makes it the device's current session; the audio held
  meanwhile goes to its run. A wake over a playing stream goes no further than stopping it: its
  audio carries the device's own voice, so once the arbitration lets it through (below) its
  session ends with `state idle` — no transcript, no answer — and the run log shows it as `ok`,
  reason `wake_stop`; a command takes a new wake. A wake the check rejects gets `state idle` with the proposed
  `session` (any `state` of that number other than `listening` refuses it): the device drops the
  proposal without showing it, and its microphone stops unless the current session listens for a
  follow-up (the older run opened its follow-up window while the wake was held), which keeps it
  streaming into that window. The current session goes on under its own number, its playback to
  its end. A lost link drops a proposal. The run log shows the rejected wake as `rejected`, reason
  `wakeword_reject` (`wakeword_error` when the check timed out or failed and its policy rejects).
- A wake taken at once is confirmed by the wake word in its recognised phrase (when the second
  wake-word check is off, it is not checked): one of the first
  three words of the transcript must start with the `word` of the `wake`, so a wake word fused
  with the next one («захаркоторый час») counts. A phrase without it gets no `transcript` and no
  answer — `state transcribing`, then `state idle` — and the run log shows it as `rejected`,
  reason `wakeword_reject`, with the recognised text. A phrase with it gets the end-of-phrase
  chime once the wake word is found, and runs like any phrase.
- One phrase heard by several devices: the server holds a `wake` for 500 ms to collect the wakes
  of the other online devices, or less once every online device has woken; a lone online device
  is not held. With several wakes it rules only once
  each of them has its pre-roll in, or 500 ms after that device's own `wake`, and, for a `wake`
  over a running answer, the server's second wake-word check has ruled on it; a wake that check
  rejects takes no part. Any other wake takes part as it is and its phrase is checked for the
  wake word only once it has won, so a false one can win its group and start the cooldown. Once
  a device with a speaker is ready, a device without one is not waited for: it cannot win. A
  `wake` over a running answer (above) joins at its `wake` like any other.
  The server keeps one session — a device with a speaker over one without, between equals the
  higher signal-to-noise ratio of the pre-roll, then the earlier wake. Each held device gets
  `listening` as soon as its wake is confirmed (at once unless it is over a running answer),
  without waiting for the ruling; one ruled out by then — in the cooldown, or by the ruling its
  own confirmation completes — gets none, and is never taken: a running answer of that device plays
  on. The devices that lost stream their microphone on until the server's VAD finds the end of the
  phrase; then each one shown `listening` gets `state idle`, and each other one is ended as a
  rejected wake is (above). For 2000 ms after the first of those wakes, a `wake` from any other
  device is ended the same way. The ended runs
  show in the run log as `rejected`, reason `duplicate`, with the phrase recognized by the STT
  of the first enabled chain; the device gets no `transcript` of it. A device ended before the
  check ruled on it is ended as a rejected wake as soon as the check rejects it instead, and its
  run shows as `rejected`, reason `wakeword_reject`, untranscribed. When the kept session
  answers, each ended device without a speaker gets that answer as a `reply` without `session`,
  so its screen shows it; a device with a speaker gets nothing, nor does a later answer of the
  same session (a follow-up).
- «стоп» said as the phrase of a session — after the wake word or in the follow-up window — ends
  the session on the server's side: `state idle`, no sound, no `transcript`.
- A device without a speaker (`caps.speaker: false`) never gets playback: the server skips TTS for
  it and sends the text only.

### Follow-up

With the server stage `followup` on, a device with `caps.follow_up` keeps the session open after
the answer, so the next phrase needs no wake word:

```
  played {id, dry, gap_ms}        ──►    (the answer is out, not dropped)
                                  ◄──    state listening
  0x01 mic: live, no pre-roll     ─►
                                  ◄──    state transcribing     (end of phrase)
                                  ◄──    state thinking / reply / speaking … as above
  or no speech in the window:     ◄──    state idle
```

- A `listening` of the open session whose microphone is off turns the microphone on again: the
  device streams live audio from the moment the `listening` arrives, no pre-roll, in the frames of
  a wake. While a proposed wake's audio is streaming, the follow-up goes on in that stream from
  where it is.
- The server drops the first 300 ms of the window's audio, in which the end of the answer still
  sounds in the room, then waits `followup` window for speech to start. Silence ends the session with
  `state idle`, without a sound. A phrase that is «стоп» ends it the same way without asking jev. Any other phrase goes to the jev decision model with the assistant's name
  (the `word` of the session's `wake`), the last request and answer; one it does not take for a
  continuation of the dialog also ends the session with `state idle`, without a sound and without
  a transcript. One it takes gets the end-of-phrase chime and runs like any phrase, and its answer
  opens the window again.
- No follow-up after an answer that was dropped (`play_stop`, a new `wake`), outside a session (an
  announcement), or on a device without `caps.follow_up`. An announcement while the window still
  waits for speech ends the session with `state idle` before it plays; an answer with an
  announcement queued behind it opens no window.

## Playback

- `play_start {id}`, then `0x02` frames of up to 4096 payload bytes, then `play_end {id}`. `id` is
  a server counter. The audio is mono at `hello.play.rate`; the server resamples to it.
- The device plays streams back to back in arrival order (the end-of-phrase chime, a filler, the
  answer) and sends `played {id, dry, gap_ms}` when the last sample of a stream is out: `dry` is
  how many times the stream ran dry after it had started and before its `play_end` came (a
  `play_end` that comes after the last sample is out counts too), `gap_ms` the silence the speaker
  played meanwhile, in ms. The wait before a stream's first sample is not counted.
- `play_stop` flushes everything queued and playing; the device answers `played {id, "dropped":
  true}` for each stream it dropped. A new `wake` flushes nothing on the device, and the server
  sends `play_stop` once it takes the wake (see [Voice session](#voice-session)). `0x02` frames and `play_end` of a
  dropped stream that are still on the way are discarded.
- The server stops waiting for `played` of a stream after its duration plus `buffer_ms` plus 5 s
  and treats it as dropped.
- Pacing: the server keeps at most `buffer_ms − play_prime_ms − 500` ms of audio ahead of the
  device's playback, estimating that playback runs at real time from the first frame it sends;
  `play_prime_ms` is the last one it sent to that connection in `config`. The device buffer holds
  `buffer_ms`.
- The device starts a stream, and resumes one that ran dry before its `play_end`, once it holds
  `play_prime_ms` of it or its `play_end` has come; pacing leaves that much room in the buffer
  for the wait.
- Indication while playing outside a session (an announcement) is the device's own choice.

## Settings and diagnostics

| message | direction | meaning |
|---|---|---|
| `set_volume {value}` | server → device | speaker volume, percent; the device stores it itself |
| `volume {value}` | device → server | the volume changed on the device (the dial) or after `set_volume` |
| `ambient {value}` | device → server | the room's noise after an auto volume measurement, dBFS of the 32-bit microphone slot, one decimal: see [Auto volume](#auto-volume) |
| `volume_boost {value}` | server → device | percent points, −100..100, the speaker adds to its volume for the replies: see [Auto volume](#auto-volume) |
| `wake_peak_stream {on}` | server → device | start or stop the wake probability stream |
| `wake_peak {value}` | device → server | once a second while the stream is on: the highest wake-word probability of that second, percent |
| `detect {on}` | server → device | wake-word detection on or off; off: the models keep running (`wake_peak` too) but a detection starts no session. Every new connection starts with detection on |
| `capture {id, seconds}` | server → device | record a sample, see [Capture](#capture) |
| `capture_resume {id, from}` | server → device | go on with a capture the device kept across a lost link |
| `capture_stop {id}` | server → device | end a capture now, see [Capture](#capture) |
| `audio_end {reason, id}` | device → server | a capture ended on the device: `capture` (all its audio is out) or `busy` (refused, a session is running) |
| `log_stream {on}` | server → device | send the device log in `log` messages or stop; after every `hello` and whenever `core.devices[].log` changes. Off from boot. A lost link does not turn it off on the device: the lines logged meanwhile wait (16 KB on the Voice PE, a line that does not fit is dropped) for the next connection's `log_stream`, and off drops them |
| `stats {…}` | device → server | the device's health, right after `hello` and every 30 s while connected: `rssi` (dBm of the access point, absent while not associated), `cpu0`, `cpu1` (load of each core over the last 30 s, percent; `cpu1` absent on a single-core chip; both absent from firmware built without `CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS`, which costs about 1.8 KB of internal RAM — the Voice PE is built without it), `temp` (chip temperature, °C, one decimal; absent when it cannot be read), `heap`, `heap_min`, `heap_block` (internal RAM: free, the lowest free since boot, the largest free block; bytes), `psram`, `psram_min` (PSRAM free and the lowest free since boot; absent without PSRAM), `wifi_drops` (Wi-Fi disconnect events since `zl_start()` at boot — the ones before it are not counted — each failed reconnect attempt included; never reset by a reconnection to the server) and `wifi_reason` (the ESP-IDF reason code of the last one, absent before the first). The server publishes them to its telemetry |
| `log {text}` | device → server | one or more whole lines of the device's ESP log, each ending in `\n` (on the Voice PE up to 2 KB a message, a line cut at 512 bytes). The server appends them to `data/device_logs/<id>/<YYYY-MM-DD>.log` (its local day), each behind its own time of arrival `HH:MM:SS.mmm`, and deletes a day's file once that day ended more than `core.device_log_days` (7) days ago |

### Auto volume

- While `auto_volume` is on, the device measures the room's noise after every 15 s of quiet:
  nothing playing, no session open, no wake proposed and no capture held. The Voice PE switches the XMOS channel 0
  to the raw microphone 0 (stage NONE: before the XMOS's echo cancellation, noise suppression and
  AGC) for about a second, takes the RMS of its 32-bit slots and sends `ambient {value}`:
  `20·log10(rms / 2^31)` dBFS, one decimal. Playback starting, a wake or a capture starting ends
  a measurement at once and nothing is sent: channel 0 goes back to AGC. Channel 1, the wake word's input, does not change.
- The server keeps the connection's last 4 `ambient` values (a minute) and sends
  `volume_boost {value}`, `round((median − threshold) × gain)` within −100..100 (negative below
  the threshold), whenever it
  changes: after an `ambient`, and when `core.devices[].auto_volume`, `auto_volume_threshold`
  (dBFS, default −60) or `auto_volume_gain` (percent per dB, default 2.0) changes; 0 while
  `auto_volume` is off. The threshold and the gain stay on the server.
- The device plays the replies at its own volume plus the boost, within 1..100 % (0 would mute
  the reply), and the music of its Sendspin player at that volume without the boost; a volume of
  0 mutes both.
  It does not store the boost, does not report it as `volume`, and shows its volume without it.
  It is 0 from boot until the server sends one.

## Capture

A capture records a sample of the microphone for the server (a dataset take, a manual recording),
outside any voice session.

1. Server → `capture {id, seconds}`. `id` (> 0) is the server's number for this capture, opaque to
   the device. The device pauses wake-word detection and records `seconds` of the microphone from
   the moment of the command, no pre-roll. A new `capture` replaces a running one. While a session
   is open or a wake awaits the server's ruling the device refuses with
   `audio_end {reason: "busy", id}`.
2. Device → `0x04` frames: the kind byte, the offset of the frame's first sample as a u32
   little-endian count of frames (one sample per channel) from the capture's first frame, then
   32 ms of samples interleaved as in `0x01`. Offsets grow; a jump between two frames is audio
   that is lost.
3. Device → `audio_end {reason: "capture", id}` once its last frame is out. Detection returns to
   what `detect` set. The device shows its own recording indication from the command until then.

Server → `capture_stop {id}` ends the capture early. A device sending the capture stops recording,
sends the frames it recorded up to the command and then `audio_end {reason: "capture", id}` as in
step 3; one holding the capture without sending it (its link was lost and no `capture_resume`
came) ends it at once, without `audio_end`. A `capture_stop` for any other capture is ignored. The
server keeps a stopped capture as `done`, its `seconds` the length that came, with no tail gap;
one stopped before any audio came ends as `error` "no audio", like any capture without audio.

A lost link does not end a capture:

- The device keeps the capture and its audio in the microphone tap (about 7 s on the Voice PE and
  the panel) and sends nothing; its next `hello` carries `"capture": <id>`.
- The server that knows the capture answers `capture_resume {id, from}`, `from` being the offset of
  the first frame it lacks. The device goes on sending from there, or from its oldest frame still
  in the tap when that one is overwritten already. A `capture_resume` for any other capture is
  ignored.
- A server that does not take up the capture (it does not know it, or runs another one) answers
  `capture_stop {id}`, and the device ends it. A capture the device cannot send ends 2 s after
  its last frame was recorded, without `audio_end`.
- When the device comes back without the capture (it restarted), the server sends a new `capture`
  for the time that is left and puts its audio after the hole.

The server writes a capture's audio to its disk as the frames arrive, so neither a lost link
nor a restart of the server loses what already came, and a restarted server resumes a capture it
finds there by the `id` in `hello`.
