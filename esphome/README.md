# ESPHome firmware for Voice PE with the «Захар» wake word

Takes control of a **Home Assistant Voice PE** (ESP32-S3) and runs the custom
on-device wake word **«Захар»**. The stock firmware is pulled unchanged from the
official Nabu Casa repo as a remote package — only the device name, Wi-Fi/API
credentials and the `micro_wake_word` model list are overridden in
[`zakhar-voice.yaml`](zakhar-voice.yaml).

## What's here

| File | Purpose |
|------|---------|
| `zakhar-voice.yaml` | The config. Pulls official Voice PE firmware `@26.5.0` + adds «Захар». |
| `secrets.yaml` | Wi-Fi creds + API key (gitignored). **Edit the two Wi-Fi lines.** |
| `.gitignore` | Keeps `secrets.yaml` and `.esphome/` out of git. |

The model is referenced by a **local path** — `../microWakeWord/v8/model/zakhar.json`
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
   esphome config zakhar-voice.yaml
   ```

3. **First flash over USB-C** (the factory OTA password is unknown to us, so the
   first take-over must be wired). Plug the Voice PE into this computer:
   ```bash
   esphome run zakhar-voice.yaml      # choose the /dev/cu.usbmodem… serial port
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

- `probability_cutoff` is set to `95%` in `zakhar-voice.yaml`. If continuous
  music ever triggers it, raise to `97%`. The `vad:` gate from the stock config
  stays on and already suppresses most music false-accepts.

## Updating the base firmware later

Bump the `@26.5.0` ref on the `home-assistant-voice` package in
`zakhar-voice.yaml` to a newer release tag and re-flash.
