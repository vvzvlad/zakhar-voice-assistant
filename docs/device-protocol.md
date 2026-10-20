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
- «несколько раз были проблемы с записью захаров … предложи как поменять протокол/сервер чтобы
  такого не было», then «го первые 4 и 5» — a capture is written to the server's disk as its
  audio arrives and stays there until deleted through the panel API; a lost link does not end it;
  a dataset take the link broke is recorded again; a capture on an offline device fails at once
  and is "recording" only from its first audio; a capture has an id and its audio carries the
  offset, the device keeps it across a lost link and the server tells it where to go on. The
  proposed `capture_stop` (end a capture early) was not taken.
- On a capture across a dropped link (the speaker notices a dead link after its 3 s send timeout,
  then waits the 1 s backoff and ~1 s to reconnect), option chosen: «Оставить как есть» — no
  faster reconnect; audio the tap no longer holds shows as a hole in the capture's `gaps`.
- «у панели просралась часть аудио в центре. почему? мы фиксили чет такое у колонки» — a Wi-Fi
  stall longer than the room the tap had left after the 2 s pre-roll (about 1 s) made the device
  skip ahead and cut audio out of the phrase without a trace. Option chosen: «Буфер 8 с + TCP 32
  КБ» — the microphone tap of the Voice PE and the panel holds 8.192 s (7.2 s of it readable),
  so a send stalled up to the 3 s send timeout while the pre-roll goes out loses nothing, and the
  rectangular panel (JC4880P443) gets the 32 KB TCP send buffer the speaker and the round panel
  have.
- «надо еще некоторое время слушать после окончания фразы которую говорит колонка, если я хочу
  продолжить диалог без активационной фразы», «еще одна стадия в конце, которую можно включить?»,
  «через jev стоит оценивать, продолжение это диалога или нет», then «го делать» — after the
  spoken answer of a session the server sends `listening` again and the device reopens its
  microphone without a wake word (see [Follow-up](#follow-up)); the server stage `followup`
  turns it on and off. A phrase heard in that window goes on only when the jev decision model
  takes it for a continuation of the dialog; otherwise it is dropped without a sound.
- On the end-of-phrase chime of such a phrase, option chosen: «После «да» от jev» — it plays once
  jev accepted the phrase, never on speech jev drops.
- «давай ее добавим тож» (the stop word of the stock Voice PE firmware) — the speaker runs the
  stock «Stop» model (kahrendt/microWakeWord, release `stop`) from 1 s after its playback starts
  until the playback ends; a detection only stops: the playback and the device's run end, no new
  session starts.
- «надо бы сделать» (cancel the speaker's own playback in the microphone, to hear commands over
  its reply), option chosen: «Не делать» — no echo cancellation on the device. Measured on the
  Voice PE with ESP-SR's AEC against the exact playback: the echo drops by 28 dB on a raw
  microphone (XMOS stage NONE) and by about 1 dB on the XMOS-processed channels the wake word and
  the server use; hearing over the reply would mean taking a raw microphone from the XMOS. The
  findings: [echo-cancellation.md](echo-cancellation.md). Asked again for the stop word
  («можешь восстановить код, надо для слова стоп»), option chosen: «Только сохранить код» — the
  code is kept on the branch `echo-cancel`. Then, to the plan that while the speaker plays the
  XMOS channel 0 is switched to the raw microphone, the speaker's own playback is cancelled in it
  and the stop word listens to it, with «Захар» on channel 1 and the server getting that cleaned
  channel 0 for the pre-roll of a wake word said over the reply: «ну ок, давай попробуем» — while
  the speaker plays, channel 0 (to the stop word and to the server) is the raw microphone with the
  playback cancelled; channel 1 and the wake word stay as they are.
- «сделал бы переключалку, чтобы с сервера можно было настраивать, хотя бы вкл/выкл» —
  `echo_cancel` in `config`, a 0..3 control among the device's controls in the admin panel: 0 turns
  the device's cancellation of its own playback off, 1..3 set how hard it suppresses the echo left
  over.
- «так а можно ее перезапускать? хотелось бы потом потестировать на разных вариантах» (about the
  canceller's mode and filter length, fixed when it is created) — `echo_cancel_mode` and
  `echo_cancel_filter` in `config` and among the device's controls; the device re-creates its
  canceller when either changes, without a restart.
- «наверное надо на сервере сделать отдельную фичу, которая пишет непрерывно лог колонки за
  последние сколько-то дней», option chosen: «Колонка шлёт лог сама» — the speaker sends its log
  lines over this connection (`log`), port 23 stays free for watching by hand, and what it logs
  before it reaches the server is not kept; the server keeps 7 days. Then «сделать тока
  переключатель в прошивке, вкл дебаг или нет», option chosen: «В панели, без перепрошивки» — a
  per-device switch in the panel's device dialog (`core.devices[].log`), applied on the fly
  through `log_stream`; off, the device sends nothing.
- «поисследуй, как можно отбрасывать копии с разных колонок», then «давай сделаем 3, да» — when
  several devices wake on one phrase, the server keeps one session and ends the others with
  `state idle`. Of the rules offered (the first wake wins; the cleanest wake
  word wins; device priority; compare the texts after recognition), option 3: a device with a
  speaker beats one without, between equals the device that heard the wake word with the higher
  signal-to-noise ratio, then the earlier wake. Then «пусть тоже распознается» — a dropped
  duplicate's phrase is recognized for the run log too; «да пусть с индикацией, похуй, лишь бы
  потом показало что дубликат» — the losing device may show its listening indication meanwhile,
  the run log must show the run as a duplicate. Then, of «только тем панелям, что услышали эту же
  фразу и проиграли» and «всем устройствам без динамика — каждый ответ», «первый вариант наверное,
  да» — the answer of the kept session is shown on the screens of the devices without a speaker
  that lost that phrase.
- «дребезжит немного звук иногда на колонке», heard as «Щелчки, звук рвётся»: the speaker played
  with no reserve and a pause in the arriving audio became silence. Option chosen: «колонка копит
  ~0,3 с звука перед фразой», then, against waiting for Fish Audio's second chunk, «Оставить
  0,3 с». Measured later (5 of 34 phrases still ran dry for 50–320 ms, every time while Fish
  Audio paused 0.4–1 s after its first ~0.5 s of audio), option chosen: «Запас 0,7 с», then «надо
  бы вынести в ручку» — the reserve is `play_prime_ms` in `config`, a per-device control in the
  admin panel, default 700.
- «давай сбрасывать только если сервер подтвердил» — a wake word heard over the speaker's own
  playback no longer flushes it on the device; the playback is dropped only once the server's
  second wake-word check confirms the wake, and a rejected wake leaves it playing to the end.
  Then the check let through the speaker's own «завтра» as «Захар» (its grammar was only the wake
  word and "[unk]"); option chosen: «Похожие слова в проверку» — for a wake held over an answer the
  check's grammar also holds sound-alike words («завтра», «сахар», «захват», «заход», «захочет»),
  and a wake decoded as one of them is rejected.
- «изучи возможность сделать адаптивную громкость в зависимости от уровня громкости звуков вокруг»,
  then «можно изредка мониторить второй канал сырых данных, смотреть на максимальную среднюю
  громкость и подстараивать громкость ответов, если автогромкость включена, с каким-то множителем
  (его можно подобрать руками)»; to measuring on the device, on a raw XMOS stage, by a median, with
  the dial as the base the server only adds to: «го попробуем, ок»; to the plan: «ок» — with
  `auto_volume` on, every 15 s of quiet (nothing playing, no session open) the speaker switches its
  channel 0 to the raw microphone for a second, measures the level and sends it (`ambient`); a wake
  word in that second switches the channel back at once. The server takes the median of the last
  minute and adds (noise − threshold) × gain to the dial's volume (`volume_boost`), which the
  speaker adds without storing it. The panel's device dialog has the switch, the threshold, the
  gain and the live noise and boost. Channel 0, not channel 1: channel 1 is the wake word's input
  and does not change. Then, against the dial being the floor: «зачем ты сделал так, что оно не
  опускается само?» — below the threshold the boost is negative and the reply is quieter than the
  dial. See [Auto volume](#auto-volume).
- Replies too quiet over music even at 99 % of the dial: of «Сжать речь на сервере», «Поднять
  аналоговое усиление», «Всё вместе», «Поднять аналоговое усиление»; told the amplifier has only
  ~3 dB left, «+3 dB и сжатие», then «тоже за фичетогглом сделай»; told the compression measures
  +2–3 dB, «Да, +3 dB и сжатие» — the Voice PE drives its line outputs at +3 dB, and the server
  compresses the audio it plays on a device whose `core.devices[].compress_replies` is on.
- «я хочу аудио-выход на нее иметь, чтобы например музыку транслировать» (as the stock firmware
  had it: Music Assistant with its Spotify Connect plugin, over Music Assistant's Sendspin
  protocol), option chosen: «В прошивке колонки» — the speaker is itself a Sendspin player that
  Music Assistant finds by mDNS, outside this protocol; the music and this protocol's playback
  share the speaker on the device.

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
   - `play` and `volume` are present only when `caps.speaker` is true.
   - `caps.follow_up` is true when the device reopens its microphone on a follow-up `listening`
     (see [Follow-up](#follow-up)); the server listens for a follow-up only on such a device.
   - `capture` (the running capture's `id`) is present only while the device holds a capture,
     see [Capture](#capture).
2. The server closes the connection with code 4000 when the device is disabled in the config, and
   closes an older connection of the same `id` with code 4001 when a new one says `hello`.
3. Server → `config`, right after `hello` and again whenever the values change:
   ```json
   {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30,
    "echo_cancel": 1, "echo_cancel_mode": 2, "echo_cancel_filter": 4, "play_prime_ms": 700,
    "auto_volume": false}
   ```
   The cutoffs are percent, 0..100. The device applies them as `uint8 = trunc(percent / 100 * 255)`
   to the wake-word model and to the on-device VAD gate. The `echo_cancel` fields set the device's
   cancellation of its own playback in the microphone its stop word hears (on the Voice PE: while
   it plays, the XMOS channel 0 is the raw microphone and ESP-SR's AEC runs on it):
   - `echo_cancel`: 0 off, the stop word hears the wake word's input; 1..3 on, with the suppression
     of the echo left over from mild to strongest; default 1. The suppression exists in the
     full-duplex modes only; in the others 1..3 just mean on.
   - `echo_cancel_mode`: 0 SR low cost, 1 SR high performance, 2 FD (full-duplex) low cost, 3 FD
     high performance; default 2.
   - `echo_cancel_filter`: the filter length, 1..8; default 4.

   `auto_volume`: true has the device measure the room's noise and send `ambient`, see
   [Auto volume](#auto-volume); default false.

   A new mode or filter length re-creates the canceller on the device; when it cannot be created
   the device works as with `echo_cancel` 0. A device without echo cancellation ignores the three.
   `play_prime_ms`, 0..2000, default 700, is the playback reserve: the audio the device holds
   before it starts a stream, and again after the stream runs dry (see [Playback](#playback)); a
   device without a speaker ignores it. A device that does not measure ignores `auto_volume`.

   Every field is required, and the server sends all of them to every device: a `config` that
   lacks one, or holds one out of its range or of the wrong type, is ignored whole. These values
   live in the server config (`core.devices[]`, whose defaults are the ones named above), not on
   the device, which keeps none of them across a restart and does nothing that depends on them
   before its first `config`: no wake word or stop word is detected, a `play_start` is answered
   with `played {id, "dropped": true}`, the echo canceller is off and auto volume does not measure.
4. Server → `volume_boost {value}` right after `config`: 0, a new connection has no `ambient` yet;
   see [Auto volume](#auto-volume).
5. Server → `log_stream {on}` with the device's `core.devices[].log`, on or off, after
   `volume_boost` (and after what a capture the device holds needs, see [Capture](#capture)): see
   [Settings and diagnostics](#settings-and-diagnostics).
6. Server → the wake-word and VAD models, and to a device with a speaker the stop model, right
   after `log_stream`: see [Models](#models).

## Models

The device stores no model. The server sends the wake-word model, the stop model and the VAD
model that gates both at every connection, again when the model configured for the device
changes, on the direct reload command (panel API `POST /api/device/model {device}`), and when a
wake-word model file the device uses is uploaded anew (panel API `POST /api/wake-models`). Which
file goes out is the server's choice: `core.devices[].wake_model` / `vad_model`, empty for the
models the release ships. The stop model is always the release's
(`microWakeWord/stop/stop.tflite`, the stock «Stop» of kahrendt/microWakeWord) and is not
configurable: no device field, no panel setting, no cutoff in `config` — the firmware compiles in
its manifest's cutoff, 0.5. Only a device with a speaker (`caps.speaker: true`) gets it.

1. Server → `model`, then the bytes:
   ```json
   {"type": "model", "kind": "wake", "name": "zakhar/zakhar_v31", "size": 77848}
   ```
   `kind` is `wake`, `vad` or `stop`, `name` is the model's name, opaque to the device, and `size`
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
  and VAD models are loaded, and the stop word only while the stop and VAD models are.

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
  gets. While the Voice PE plays with `echo_cancel` on, channel 0 is instead the raw microphone 0
  with the speaker's own playback cancelled: the raw microphone at gain ×16 (applied before the
  cut to 16 bits, saturating) through the AEC, exactly the stop word's input; the switch each way
  takes the device two 32 ms frames. During an auto volume measurement channel 0
  is the raw microphone 0 as read, at gain 1, switched the same way; a wake ends the measurement
  and switches it back (see [Auto volume](#auto-volume)).
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
  audio streams, while the current session, its playback and the indication stay as they are —
  the detection may be the device's own voice. The server holds every `wake` until its second
  wake-word check confirms it on the proposal's microphone audio (at once when that check is off),
  and the older run goes on meanwhile; over a running answer (a run of the device is going or a
  stream is playing) the check also tells apart words that sound like the wake word (the Vosk
  check's `answer_sound_alikes`), since the device's own voice reaches it. A confirmed wake is
  taken: the server cancels the older run and sends `play_stop`, which flushes the device's
  playback, a stream the server started before it saw the `wake` included, then `state listening`
  with the proposed `session`, which makes it the device's current session; the audio held
  meanwhile goes to its run. A rejected one gets `state idle` with the proposed `session` (any
  `state` of that number other than `listening` refuses it): the device drops the proposal
  without showing it, and its microphone stops unless the current session listens for a follow-up
  (the older run opened its follow-up window while the wake was held), which keeps it streaming
  into that window. The current session goes on under its own number, its playback to its end.
  A lost link drops a proposal. The run log shows the rejected wake as `rejected`, reason
  `wakeword_reject` (`wakeword_error` when the check timed out or failed and its policy rejects).
- One phrase heard by several devices: the server holds a `wake` for 500 ms to collect the wakes
  of the other online devices, or less once every online device has woken; a lone online device
  is not held. With several wakes it rules only once
  each of them has its pre-roll in, or 500 ms after that device's own `wake`, and the
  server's second wake-word check has ruled on it; a wake that check rejects takes no part. Once
  a device with a speaker is ready, a device without one is not waited for: it cannot win. A
  `wake` over a running answer (above) joins at its `wake` like any other.
  The server keeps one session — a device with a speaker over one without, between equals the
  higher signal-to-noise ratio of the pre-roll, then the earlier wake. Each held device gets
  `listening` as soon as the second check confirms its wake (at once when that check is off),
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
- `stop`: `{"type": "stop", "session": 17}`, device → server: the stop word. The device runs the
  stop model from 1 s after its playback starts until the playback ends; a detection flushes the
  playback and sends `stop`, then `played {id, "dropped": true}` for each stream it dropped.
  `session` is the current session's number while it is open, absent outside a session; a
  proposed one is never named. A stop word starts no session.
  The server drops the playback as for a `wake` it takes and sends `play_stop`; in a session it
  then cancels the run and ends the session with `state idle`, outside one (an announcement, an
  agent's `speak`) only the playback stops.
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
- The server waits `followup` window for speech to start. Silence ends the session with
  `state idle`, without a sound. A phrase goes to the jev decision model with the assistant's name
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
  true}` for each stream it dropped. The stop word flushes the same way, and the server answers it
  with `play_stop`; a new `wake` flushes nothing on the device, and the server sends `play_stop`
  once it takes the wake (see [Voice session](#voice-session)). `0x02` frames and `play_end` of a
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
| `audio_end {reason, id}` | device → server | a capture ended on the device: `capture` (all its audio is out) or `busy` (refused, a session is running) |
| `log_stream {on}` | server → device | send the device log in `log` messages or stop; after every `hello` and whenever `core.devices[].log` changes. Off from boot. A lost link does not turn it off on the device: the lines logged meanwhile wait (16 KB on the Voice PE, a line that does not fit is dropped) for the next connection's `log_stream`, and off drops them |
| `log {text}` | device → server | one or more whole lines of the device's ESP log, each ending in `\n` (on the Voice PE up to 2 KB a message, a line cut at 512 bytes). The server appends them to `data/device_logs/<id>/<YYYY-MM-DD>.log` (its local day), each behind its own time of arrival `HH:MM:SS.mmm`, and deletes a day's file once that day ended more than `core.device_log_days` (7) days ago |

### Auto volume

- While `auto_volume` is on, the device measures the room's noise after every 15 s of quiet:
  nothing playing, no session open, no wake proposed and no capture held. The Voice PE switches the XMOS channel 0
  to the raw microphone 0 (stage NONE: before the XMOS's echo cancellation, noise suppression and
  AGC) for about a second, takes the RMS of its 32-bit slots and sends `ambient {value}`:
  `20·log10(rms / 2^31)` dBFS, one decimal. Playback starting, a wake or a capture starting ends
  a measurement at once and nothing is sent: channel 0 goes back to AGC, or stays
  raw for a playback's echo cancellation. Channel 1, the wake word's input, does not change.
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

A lost link does not end a capture:

- The device keeps the capture and its audio in the microphone tap (about 7 s on the Voice PE and
  the panel) and sends nothing; its next `hello` carries `"capture": <id>`.
- The server that knows the capture answers `capture_resume {id, from}`, `from` being the offset of
  the first frame it lacks. The device goes on sending from there, or from its oldest frame still
  in the tap when that one is overwritten already. A `capture_resume` for any other capture is
  ignored.
- A server that does not know the capture does not answer. A capture the device cannot send
  ends 2 s after its last frame was recorded, without `audio_end`.
- When the device comes back without the capture (it restarted), the server sends a new `capture`
  for the time that is left and puts its audio after the hole.

The server writes a capture's audio to its disk as the frames arrive, so neither a lost link
nor a restart of the server loses what already came, and a restarted server resumes a capture it
finds there by the `id` in `hello`.
