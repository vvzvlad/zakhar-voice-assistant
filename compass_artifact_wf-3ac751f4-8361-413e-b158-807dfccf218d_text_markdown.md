# Replacing Home Assistant with a Custom Server for the Voice PE — and Running Fast Russian+English STT/TTS on an Allwinner T507

## TL;DR
- **Question 1:** The Voice PE talks the **ESPHome native API** (protobuf over TCP port **6053**, optionally Noise-encrypted) — it is NOT a Wyoming device. To remove Home Assistant you must implement the ESPHome native-API *server* side and drive the `voice_assistant` state machine via a small set of protobuf messages (`SubscribeVoiceAssistantRequest`, `VoiceAssistantRequest`, `VoiceAssistantResponse`, `VoiceAssistantEventResponse`, `VoiceAssistantAudio`). This is a few hundred lines of Python, and **OHF-Voice/linux-voice-assistant already does exactly this** — fork it rather than starting from scratch.
- **Question 1 (start here):** Don't reflash to "a simpler protocol" — there is no lighter protocol baked into Voice PE firmware; `voice_assistant` always rides the native API. The cleanest path is to have your server impersonate Home Assistant over the ESPHome native API. Use the **API_AUDIO** subscribe flag so mic audio arrives in-band over TCP (no UDP port juggling).
- **Question 2:** Yes — a quad-core Cortex-A53 @ 1.5 GHz (the T507, no NPU) comfortably runs **Vosk small models** (RU + EN, ~39–50 MB each, streaming, roughly real-time) for STT and **Piper medium voices** (RU + EN, RTF ~0.2 on desktop CPU, real-time on Pi 4) for TTS. This is a realistic, genuinely usable CPU-only stack. whisper.cpp tiny/base is the accuracy-oriented alternative for STT but is batch-oriented (30 s windows) and gives a worse latency UX than Vosk's streaming.

## Key Findings

### Architecture reality
- The Voice PE (ESP32-S3 running ESPHome's `home-assistant-voice.yaml`) connects to HA over the **ESPHome native API**: a custom TCP protocol using protocol buffers on **port 6053**, implemented client-side by the `aioesphomeapi` Python library. Wake word runs **on-device** via microWakeWord; HA orchestrates STT/intent/TTS.
- Wyoming is a **separate** protocol (JSONL + PCM over TCP, ports like 10200/10300/10400/10700). HA uses Wyoming to talk to *backend services* (Whisper for STT, Piper for TTS, openWakeWord for wake). **The Voice PE itself does NOT speak Wyoming** — so you cannot "just point the Voice PE at a Wyoming server." Removing HA means reimplementing the ESPHome native-API server side that drives the voice_assistant pipeline.
- Therefore: **Voice PE → (ESPHome native API) → your server → (your own STT/intent/TTS).** Your server has to *look like Home Assistant* to the device.

### The exact protocol your server must implement
From the official `api.proto` (esphome/esphome dev branch and esphome/aioesphomeapi main), the voice-assistant message set and ESPHome message IDs (these IDs and the core Request/Response/Audio/Event fields are verbatim from the proto):

| Message | id | Direction | Key fields |
|---|---|---|---|
| `SubscribeVoiceAssistantRequest` | 89 | HA→device | `subscribe` (bool), `flags` (uint32) |
| `VoiceAssistantRequest` | 90 | HA→device | `start`(bool), `conversation_id`, `flags`, `audio_settings`, `wake_word_phrase` |
| `VoiceAssistantResponse` | 91 | device→HA | `port` (uint32, UDP port device opened), `error` (bool) |
| `VoiceAssistantEventResponse` | 92 | device→HA / HA→device | `event_type` (enum), `data` (repeated name/value) |
| `VoiceAssistantAudio` | 106 | both | `data` (bytes), `end` (bool), `data2` (2nd mic) |
| `VoiceAssistantTimerEventResponse` | 115 | device→HA | timer fields |
| `VoiceAssistantAnnounceRequest` | 119 | HA→device | `media_id`, `text`, `preannounce_media_id`, `start_conversation` |
| `VoiceAssistantAnnounceFinished` | 120 | device→HA | `success` |
| `VoiceAssistantConfigurationRequest` | 121 | device→HA | external wake words |
| `VoiceAssistantConfigurationResponse` | 122 | HA→device | available/active wake words, max |

Subscribe flags (`VoiceAssistantSubscribeFlag`): `VOICE_ASSISTANT_SUBSCRIBE_NONE = 0` (device opens a **UDP** socket and reports its port in `VoiceAssistantResponse`) and `VOICE_ASSISTANT_SUBSCRIBE_API_AUDIO = 1` (audio in-band via `VoiceAssistantAudio` over the existing TCP connection). Request flags (`VoiceAssistantRequestFlag`): `NONE=0`, `USE_VAD=1`, `USE_WAKE_WORD=2`. The audio format is **16 kHz, 16-bit, mono** (per the ESPHome voice_assistant docs).

The event enum (`VoiceAssistantEvent`) your server emits/consumes to drive LEDs and pipeline stages:
`ERROR=0, RUN_START=1, RUN_END=2, STT_START=3, STT_END=4, INTENT_START=5, INTENT_END=6, TTS_START=7, TTS_END=8, WAKE_WORD_START=9, WAKE_WORD_END=10, STT_VAD_START=11, STT_VAD_END=12, TTS_STREAM_START=98, TTS_STREAM_END=99, INTENT_PROGRESS=100.` Payloads ride in the generic repeated `data` name/value pairs — e.g. `STT_END` carries `name:"text"` (the transcript), `TTS_END` carries `name:"url"` (the audio URL the device fetches/plays).

### The handshake / flow
1. **Connect + handshake** (TCP 6053): `HelloRequest`/`HelloResponse`, then (optionally) Noise encryption negotiation, then `DeviceInfoRequest`/`Response`, `ListEntitiesRequest`, `SubscribeStatesRequest`.
2. Your server sends `SubscribeVoiceAssistantRequest{subscribe=true, flags=API_AUDIO}`.
3. On-device wake word fires → device sends `VoiceAssistantRequest{start=true, wake_word_phrase=...}`. (In UDP mode the device first replies with `VoiceAssistantResponse{port}`; in API_AUDIO mode audio just flows in-band.)
4. Device streams mic audio as `VoiceAssistantAudio{data=...}` chunks (16 kHz/16-bit/mono); `end=true` marks end of stream.
5. Your server runs STT, then emits `VoiceAssistantEventResponse` events (`STT_START`, `STT_END` with `name:"text"`, `INTENT_START/END`, `TTS_START`, `TTS_END` with `name:"url"`). For playback you either send a media URL the device fetches, or stream TTS audio back.
6. Device plays the response, sends `VoiceAssistantAnnounceFinished`/run-end; pipeline returns to idle.

### Existing projects that replace HA as the server endpoint
- **OHF-Voice/linux-voice-assistant** (Apache-2.0, Python; **469 stars / 79 forks as of June 2026**, GitHub repo header) — by the Open Home Foundation voice team (Michael Hansen / synesthesiam). It is the **official successor to wyoming-satellite** (confirmed by Hansen; OHF backlog issue #45 is titled "Replace Wyoming satellite with Linux Voice Assistant"). It acts as the ESPHome device/server side, listening on TCP 6053; HA connects to it as an ESPHome client. Per its README it "Works with Home Assistant using the ESPHome protocol/API (via aioesphomeapi)," and supports local wake word (OpenWakeWord/microWakeWord), announcements, start/continue conversation, and timers. This is the single best codebase to study/fork — it contains the complete ESPHome server-side voice plumbing you need.
- **peterkeen/aioesphomeserver** — "Python implementation of ESPHome native API server"; lets your Python program appear to HA as an ESPHome device, serving the native API plus a web server compatible with the on-device dashboard. Excellent scaffolding for the handshake/entity side; you'd add the voice-assistant message handling.
- **emme99/hybrid-voice-assistant** — a Python relay that talks the ESPHome Native API; its README diagrams the exact `VoiceAssistantRequest`/`VoiceAssistantAudio`/`VoiceAssistantEvent` exchange on port 6053. A concise worked example of the message flow.
- **esphome/aioesphomeapi** itself — its `client.py` exposes voice-assistant server helpers (the start callback returns the UDP port the server opens; plus `handle_stop` and `handle_announcement_finished`). Reading these clarifies the contract even though aioesphomeapi is normally a *client*.

### Reflashing / YAML options
You don't need to reflash to change protocol — there is no lighter protocol option in the firmware; `voice_assistant` always rides the native API. But you *can* reconfigure the ESPHome `voice_assistant:` YAML and reflash for control: `microphone`, `micro_wake_word`, `speaker`/`media_player`, `use_wake_word`, `noise_suppression_level`, `auto_gain`, `volume_multiplier`, `conversation_timeout`, plus automations (`on_wake_word_detected`, `on_stt_end`, `on_tts_start`, `on_error`, `on_client_connected/disconnected`, etc.). The official Voice PE firmware lives at **esphome/home-assistant-voice-pe** on GitHub. If you were willing to write a *custom ESPHome component* and reflash, you could even stream audio over a plain UDP/WebSocket path — but that is *more* work than impersonating the native-API server, which existing libraries already handle.

### How much code?
Your instinct is right: this is **not much code** — on the order of a few hundred to ~1,000 lines of Python if you fork `linux-voice-assistant` or build on `aioesphomeapi`'s protocol layer. The protobuf definitions ship with `aioesphomeapi`; you implement the handshake + the ~6 voice messages + your STT/intent/TTS glue. Do **not** hand-roll protobuf — reuse `api_pb2`.

### Question 2 — Allwinner T507 hardware (confirmed)
- **Specs:** Allwinner T507/T507-H = quad-core ARM Cortex-A53, 1.4–1.5 GHz (per CNX Software, May 13 2022: "Allwinner T507 quad-core Cortex-A53 @ 1.5GHz with Arm Mali-G31 MP2 GPU with support for OpenGL ES 3.2/2.0/1.0, Vulkan1.1, OpenCL 2.0; '2.25W power consumption under load'"). 32-bit DDR3/DDR4/LPDDR3/LPDDR4 (boards ship 1–4 GB). **AEC-Q100 automotive-grade, −40 °C to +85 °C, Allwinner promising "over 10 years of longevity."**
- **No NPU** — confirmed; a vendor note explicitly states "Limited AI performance without an NPU (e.g., requires external AX620A)." (The T527 is the NPU-equipped sibling.) For reference, linux-sunxi.org notes the T507 "uses the same die [as the H616], but routes out additional pins to expose LCD and camera functionality."
- **CPU positioning:** The T507 sits between a Raspberry Pi 3 (Cortex-A53 @ 1.2–1.4 GHz) and a Pi 4 (Cortex-A72 @ 1.5 GHz). Pi 3/Pi 4 A53/A72 benchmarks are the right proxy; treat it as **Pi 3-class** for conservative planning.

### STT options on A53 (CPU only, priority = speed)
- **Vosk (recommended).** Kaldi-based, offline, **streaming** API. Per alphacephei.com: "Vosk models are small (50 Mb) but provide continuous large vocabulary transcription, zero-latency response with streaming API." The English small model `vosk-model-small-en-us-0.15.zip` is **39.3 MB** on disk; the Russian small model is similar; both need ~300 MB RAM at runtime and are explicitly designed for Raspberry Pi/Android/embedded. There are dedicated **Russian and English** small models. Real-world data point: on a Pi 3B, Vosk transcribed a 60-second clip in ~90 s on a single core (~25% of 4 cores) on an *older* version — but maintainers shipped v0.3.17 with "great speed improvements specifically for small devices." Crucially, streaming means partial results arrive *as the user speaks*, so perceived latency is far better than raw RTF implies. The SEPIA author summarizes Vosk as "very small, fast, supports streaming audio" and "probably your best open-source ASR choice on low-end hardware."
- **whisper.cpp tiny/base (alternative, accuracy-first).** Pure C/C++, ARM NEON, int8/q5 quantization. tiny ≈ 75 MB, base ≈ 142 MB; both fit in <1 GB RAM. Whisper operates on **30-second padded chunks**, so it's batch/window-oriented and streaming UX is poor — SEPIA testing reports "for Raspberry Pi 4 based voice assistants you have to wait usually >3 s after finishing your input to get a result (bad UX)." Reported Pi 5 figures are tiny ~15× real-time and base ~6× real-time, but the T507 is Pi 3-class, so expect roughly real-time (tiny) to a few× slower than real-time (base). Multilingual tiny/base support RU + EN. **Verdict: use Vosk for live voice UX; reserve whisper.cpp for higher-accuracy, latency-tolerant use.**

### TTS options on A53 (CPU only)
- **Piper (recommended).** VITS-based neural TTS by Rhasspy/OHF-Voice, ONNX runtime, **CPU-first**. Piper's source is "optimized for Raspberry Pi 4 hardware but can run on various platforms" (DeepWiki, sourced to `src/cpp/piper.cpp`). It has **Russian** voices (`ru_RU` — denis, dmitri, irina, ruslan; medium quality, 22.05 kHz, ~15–20 M params) and many **English** voices (`en_US-lessac-medium`, `en_US-amy-medium`, etc.). Performance: arXiv:2512.08006v1 (Dec 2025) reports Piper "achieving a real-time factor (RTF) of approximately 0.2 … synthesis five times faster than real-time" on an i7 CPU-only, noting Piper's "lightweight 15-20M parameters"; on the RK3588 CPU it measured ~0.65; on a Raspberry Pi 5 a Piper-vs-Coqui benchmark reports INT8 RTF **0.12** ("synthesize one second of speech in just 120 ms"). Community consensus: Piper runs **real-time on a Pi 4** with medium models, and "low"-tier models generate "3 to 5 times faster than high quality." On a T507 (Pi 3-class) expect medium voices to be near-real-time to ~2× slower than playback — usable, especially with the low/medium tiers.
- **eSpeak-NG (fallback).** Formant synth, "generates speech in milliseconds," robotic but intelligible, 40+ languages incl. RU + EN. Use as an ultra-low-latency fallback or for phonemization (Piper already embeds espeak-ng for grapheme-to-phoneme conversion).

## Details

### Why you can't avoid the ESPHome native API
Forum users repeatedly try to "just send the audio somewhere else." The `voice_assistant` component is hard-wired to the native API: it sends a `VoiceAssistantRequest` to start and (in UDP mode) "asks Home Assistant to start a UDP server and then sends the received microphone data via UDP to that server" (PR #4648). The PR author (jesserockz) notes the component "is specifically set up to transmit the data to HA via the random port it receives," though "any component can latch onto a microphone and request the data stream and do whatever it wants." So a custom ESPHome component + reflash *could* stream over a simpler path — but impersonating the native-API server is less work and is already solved by existing libraries.

### Practical pipeline latency expectation on T507
With Vosk streaming STT (partial results live) + a fast intent matcher + Piper medium TTS, a short command ("turn on the kitchen light") should complete in roughly **1–2.5 s end-to-end** on a T507, dominated by TTS synthesis of the response plus network/audio buffering — comparable to a Pi 4 local Assist pipeline. whisper.cpp would push STT latency alone to >3 s, which is why Vosk wins for live voice.

### RAM/footprint
Vosk small (~300 MB runtime) + Piper medium (a few hundred MB) + your server process fits comfortably on a 2 GB T507 board; a 1 GB board is tight but workable if you avoid whisper base and run headless (no desktop GUI frees 100–200 MB).

## Recommendations

**Stage 1 — Prove the protocol (1–2 days).** Fork **OHF-Voice/linux-voice-assistant**. Run it first on any Linux box, then on the T507. Point your Voice PE at it over 6053. Confirm you see `RUN_START`/`STT_START`/`VoiceAssistantAudio` events flowing. Reuse the `aioesphomeapi` protobuf layer; never hand-roll protobuf.

**Stage 2 — Swap in your pipeline (2–4 days).** Replace the project's STT/TTS calls with **Vosk** (small RU + EN models) for STT, your own intent logic, and **Piper** (`ru_RU-irina/dmitri-medium` + `en_US-lessac-medium`) for TTS. Emit `STT_END{text}`, `TTS_START`, then return audio (a served URL the device fetches, or streamed PCM). Use the **API_AUDIO** subscribe flag to avoid UDP/firewall issues entirely (a known gotcha: ESPHome voice UDP needs explicit firewall allowances).

**Stage 3 — Optimize for speed.** Give Vosk 3 of 4 cores and keep one for the server/audio I/O; pin Piper to the remaining cores between turns. Add a local VAD so STT runs only on speech. If TTS latency is too high, drop to Piper "low" voices or pre-synthesize fixed phrases.

**Benchmarks/thresholds that change the plan:**
- If measured **Vosk small STT RTF > ~1.0** on your T507 (can't keep up with streaming audio), switch to **grammar-constrained recognition** (a fixed command vocabulary), which dramatically speeds Kaldi decoding for command-style use.
- If **Piper medium RTF > ~1.0** (synthesis slower than playback), switch to Piper low quality or eSpeak-NG.
- If you need **dictation-grade accuracy** (not just commands) and can tolerate latency, switch STT to whisper.cpp base q5 and accept ~3 s post-utterance delay.
- If you later move to a **T527 (NPU)** or **RK3588**, you can run larger Whisper/Piper models or NPU-accelerated Piper (see the "Paroli" project for RK3588 NPU acceleration, which reached RTF ~0.15).

## Caveats
- **linux-voice-assistant is designed to connect to HA**, not to be your standalone brain — you're repurposing its ESPHome-server-side plumbing and replacing the HA-facing pipeline with your own. Some refactoring is required; it's a fork, not a drop-in.
- The exact `.py` filename in linux-voice-assistant implementing the server could not be verified in this research (GitHub blocked automated source-tree fetches); inspect the `linux_voice_assistant/` package directly — the server logic, port 6053, and aioesphomeapi-protocol usage are confirmed.
- A few protobuf field *tag numbers* (inside `VoiceAssistantAnnounceRequest`, `VoiceAssistantConfigurationResponse`, `VoiceAssistantSetConfiguration`) are reported from C++/aioesphomeapi usage rather than verbatim proto; confirm against `api.proto` if you hand-encode them. The message **IDs** and the core Request/Response/Audio/Event **fields** are verbatim.
- Vosk Pi 3 timing (90 s for 60 s audio) is from an **older version** and single-core; current small models on a faster quad A53 should be near or under real-time, but **benchmark on your actual board** — vendor "full load" behavior and thermal throttling vary.
- Piper RTF figures (0.2 on i7 desktop, 0.12 INT8 on Pi 5, 0.65 on RK3588 CPU, "real-time on Pi 4") come from mixed sources/hardware; treat the T507 as Pi 3-class and expect medium voices to be usable but not instantaneous. Measure before committing to a voice tier.
- The original Piper repo (rhasspy/piper) was archived ~Oct 2025; development moved to a GPL-3.0 fork under the Open Home Foundation. Voices remain on Hugging Face (rhasspy/piper-voices).
- Real-time-factor numbers are **speed** metrics, not accuracy. Vosk small and Whisper tiny both trade accuracy for speed, and **Russian** small-model accuracy is generally lower than English — test with your actual phrases and accents.