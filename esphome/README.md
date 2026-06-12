# ESPHome firmware for Voice PE with the «Захар» wake word

Takes control of a **Home Assistant Voice PE** (ESP32-S3) and runs the custom
on-device wake word **«Захар»**. The stock firmware is pulled unchanged from the
official Nabu Casa repo as a remote package — we override the device name, Wi-Fi/API
credentials, the `micro_wake_word` model, and the `voice_assistant` component (a
**pre-roll fork**, so «Захар <команда>» streams as one continuous utterance) in
[`zakhar-voice-preroll.yaml`](zakhar-voice-preroll.yaml).

## What's here

| File | Purpose |
|------|---------|
| `zakhar-voice-preroll.yaml` | The config. Pulls official Voice PE firmware `@26.5.0` + adds «Захар». |
| `secrets.yaml` | Wi-Fi creds + API key (gitignored). **Edit the two Wi-Fi lines.** |
| `.gitignore` | Keeps `secrets.yaml` and `.esphome/` out of git. |

The model is referenced by a **local path** — `../microWakeWord/v27/model/zakhar.json`
— so `zakhar.json` and `zakhar.tflite` are read straight from this repo at build
time (no network, no push needed). A remote `github://` / raw URL does **not**
work for this model: ESPHome can't resolve the manifest's relative
`"model": "zakhar.tflite"` when the manifest is fetched that way.

## How a micro_wake_word model gets onto the device

It is **compiled into the firmware** at build time — there is no runtime file
upload. So "loading «Захар» onto the speaker" = building this firmware and
flashing it.

## Steps

1. **Edit `secrets.yaml`** — set `wifi_ssid` and `wifi_password`. The
   `api_encryption_key` is already generated; leave it.

2. **(optional) Validate the merged config:**
   ```bash
   cd esphome
   esphome config zakhar-voice-preroll.yaml
   ```

3. **First flash over USB-C** (the factory OTA password is unknown to us, so the
   first take-over must be wired). Plug the Voice PE into this computer:
   ```bash
   esphome run zakhar-voice-preroll.yaml   # choose the /dev/cu.usbmodem… serial port
   ```
   Compiling the full Voice PE firmware downloads the esp-idf toolchain on first
   run (large, slow) — expect a long first build. Subsequent updates can go OTA.

4. **Point the server at the new API key.** Taking control replaces the factory
   API key. Update `ESPHOME_DEVICES` in the repo `.env` so the PSK for this
   device equals `api_encryption_key` from `secrets.yaml`, otherwise the
   voice-assistant server can't connect.

5. **Wake word.** «Захар» is first in the model list, so it's the default active
   model on a fresh flash. The other stock wake words (hey_jarvis, hey_mycroft,
   okay_nabu) remain selectable via the device's wake-word `select`.

## Tuning (no retrain needed)

- The current model is **v27** (v16 recipe + synthetic short-«захар» negatives). It is the
  first single model to beat v16: on the honest device eval it slightly improves drawn-out
  recall (FRR 21→**19.3%** @0.90) AND fixes the field **short-«захар» false-trigger** — firing
  on the plain name «Захар» dropped **65%→23%** — while keeping every FAPH class low (music
  5.8→2.9/h). Same size as v16. Synthetic shorts will be swapped for REAL ones next round.
- `probability_cutoff` defaults to `90%` in `zakhar-voice-preroll.yaml` — v27's eval knee
  (FRR 19.3%, FAPH ~1.2/h with VAD). The DET sweep showed v16's old `95%` was OVER-tightening
  (its FAPH plateaus 0.85→0.95), so v27 ships at `90%`. v27's silence-FAPH @0.90 is ~0.8/h
  (low); if real silence false-fires appear in the field, raise to `95%` via the panel
  (FRR 22.7%, FAPH 0.86/h). The `vad:` gate stays on (halves FAPH at no recall cost). The
  model keys on the onset, not vowel duration — short-«захар» is fixed by the short negatives,
  not by a duration feature (see [`../microWakeWord/v16/DURATION_CAUSALITY.md`](../microWakeWord/v16/DURATION_CAUSALITY.md)).
- Both the **wake cutoff** and the **speaker volume** are now adjustable **live from
  the panel's Devices page** (the **Wake Probability Cutoff** and **Speaker Volume**
  numbers, 0–100) with NO re-flash; the cutoff value persists across reboots.
  `sliding_window_size` is compile-time only and still needs a re-flash.
- The firmware also exposes **Config Version** and **Model Version** read-only
  diagnostic entities, so you can confirm exactly which config/model is flashed.

## Updating the base firmware later

Bump the `@26.5.0` ref on the `home-assistant-voice` package in
`zakhar-voice-preroll.yaml` to a newer release tag and re-flash.
