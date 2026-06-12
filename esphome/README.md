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

The model is referenced by a **local path** — `../microWakeWord/v16/model/zakhar.json`
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

- The current model is **v16** (v8 recipe + real device-tract positives + 4 real
  negative classes: silence/music/speech/vacuum, all recorded through the device). On
  the honest leakage-safe device eval it matches v8's recall (~21% FRR) but with **FAPH
  0 across all classes** under the VAD pre-gate, where v8 false-fired ~12.5/h in real
  silence — so it fixes the field silence false-fire bug at no recall cost.
- `probability_cutoff` defaults to `95%` in `zakhar-voice-preroll.yaml` — FIELD-CORRECTED:
  at `80%` v16 false-fired in real silence (heard «захар» in a quiet room), so it was
  raised to `95%`. v16's DET curve is ~flat (recall floor ~21% FRR across cutoffs), so
  `95%` removes the silence false-fires WITHOUT a meaningful recall cost. The `vad:` gate
  from the stock config stays on; lower live from the panel only if real «захааар» starts
  getting missed. The ~21% recall floor is fundamental this round — the model keys on the
  onset, not vowel duration (see [`../microWakeWord/v16/DURATION_CAUSALITY.md`](../microWakeWord/v16/DURATION_CAUSALITY.md)).
- Both the **wake cutoff** and the **speaker volume** are now adjustable **live from
  the panel's Devices page** (the **Wake Probability Cutoff** and **Speaker Volume**
  numbers, 0–100) with NO re-flash; the cutoff value persists across reboots.
  `sliding_window_size` is compile-time only and still needs a re-flash.
- The firmware also exposes **Config Version** and **Model Version** read-only
  diagnostic entities, so you can confirm exactly which config/model is flashed.

## Updating the base firmware later

Bump the `@26.5.0` ref on the `home-assistant-voice` package in
`zakhar-voice-preroll.yaml` to a newer release tag and re-flash.
