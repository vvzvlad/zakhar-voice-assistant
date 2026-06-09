#!/usr/bin/env python3
"""Playback re-recording: play each source clip through the Mac speaker while the
Voice PE records its (post-frontend) microphone, and save the device recording.

This captures synthetic «захааар» clips as they actually sound through the real
device audio tract (XMOS + noise-suppression/AGC), to close the train/serve gap.
The recorded WAV is the device's post-frontend audio (16 kHz mono) and may include
pre-roll/tail silence — the training pipeline trims silence later.

Recording goes through this project's panel API (server must be running: `make run`,
device online). The endpoint POST {panel}/api/capture {"device","seconds"} records
`seconds` of mic audio and returns the WAV bytes in the HTTP response.

See capture_playback.md (next to this file) for the full why/how.

Usage:
  python3 microWakeWord/capture_playback.py               # play positive_samples_generated/*
  python3 microWakeWord/capture_playback.py --limit 5     # quick test on 5 clips
"""

import argparse
import glob
import json
import math
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

# Defaults are resolved relative to this script's directory, so it works no matter
# what the current working directory is.
SCRIPT_DIR = Path(__file__).resolve().parent

# Native-API-backed capture is clamped to this server-side; mirror it here.
MAX_SECONDS = 300
TRANSIENT_STATUSES = (409, 504)  # device busy/offline or never streamed in time -> retry


def clip_duration_s(path: str) -> float:
    """Duration of a WAV file in seconds (via the stdlib wave module)."""
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate() or 1
        return frames / float(rate)


def post_capture(panel: str, device: str, seconds: int, holder: dict) -> None:
    """Blocking POST /api/capture; store WAV bytes or (status, body) error in holder.

    Runs in a worker thread so the main thread can play the clip concurrently. The
    request blocks until the device finishes recording `seconds` and streams it back.
    """
    url = panel.rstrip("/") + "/api/capture"
    body = json.dumps({"device": device, "seconds": seconds}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    try:
        with urllib.request.urlopen(req, timeout=seconds + 30) as resp:
            holder["wav"] = resp.read()
            holder["status"] = resp.status
    except urllib.error.HTTPError as e:
        # Non-2xx: capture the status + (JSON) error body for the caller to handle.
        holder["status"] = e.code
        try:
            holder["error"] = e.read().decode("utf-8", "replace")
        except Exception:
            holder["error"] = ""
    except Exception as e:  # connection refused, timeout, etc.
        holder["status"] = None
        holder["error"] = f"{type(e).__name__}: {e}"


def record_one(panel, device, clip, seconds, pre_roll, player_argv):
    """Start the capture, wait pre_roll, play the clip, then collect the recording.

    Returns (wav_bytes_or_None, holder) — holder carries status/error for diagnostics.
    """
    holder: dict = {}
    t = threading.Thread(target=post_capture, args=(panel, device, seconds, holder))
    t.start()
    # Give the device time to arm and actually start recording before playback.
    time.sleep(pre_roll)
    subprocess.run([*player_argv, str(clip)], check=False)
    t.join(timeout=seconds + 35)
    return holder.get("wav"), holder


def main() -> int:
    ap = argparse.ArgumentParser(description="Play clips and re-record them via the Voice PE device.")
    ap.add_argument("--src", default=str(SCRIPT_DIR / "positive_samples_generated"),
                    help="folder of source .wav clips to play")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "positive_samples_recorded"),
                    help="folder to write device recordings into")
    ap.add_argument("--device", default="voice0a332d", help="ESPHome device name")
    ap.add_argument("--panel", default="http://127.0.0.1:8201", help="panel API base URL")
    ap.add_argument("--pre-roll", type=float, default=1.0,
                    help="seconds to wait after starting capture before playback")
    ap.add_argument("--tail", type=float, default=1.5,
                    help="extra seconds of recording after the clip ends")
    ap.add_argument("--gap", type=float, default=0.7, help="pause between clips")
    ap.add_argument("--player", default="afplay",
                    help="playback command; the clip path is appended (macOS: afplay)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N clips")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra retries on transient capture errors (409/504)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        print(f"error: source dir not found: {src}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    player_argv = shlex.split(args.player)

    clips = sorted(glob.glob(str(src / "*.wav")))
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        print(f"error: no .wav files in {src}", file=sys.stderr)
        return 2

    print(f"{len(clips)} clips | device={args.device} | panel={args.panel} | out={out}")
    ok = 0
    failed = 0
    for i, clip in enumerate(clips, 1):
        stem = Path(clip).stem
        try:
            dur = clip_duration_s(clip)
        except Exception as e:
            print(f"[{i}/{len(clips)}] SKIP {stem}: cannot read ({e})")
            failed += 1
            continue
        seconds = max(1, min(MAX_SECONDS, math.ceil(args.pre_roll + dur + args.tail)))

        attempt = 0
        while True:
            wav, holder = record_one(args.panel, args.device, clip, seconds, args.pre_roll, player_argv)
            if wav:
                outfile = out / f"{stem}__dev.wav"
                outfile.write_bytes(wav)
                print(f"[{i}/{len(clips)}] OK   {stem} -> {outfile.name} ({seconds}s window, {len(wav)//1024} KB)")
                ok += 1
                break

            status = holder.get("status")
            err = holder.get("error", "")
            if status in (503, None):
                # 503 = capture not wired; None = no HTTP response at all (panel
                # unreachable). Either way the whole run is futile -> abort fast.
                print(f"\nFATAL: panel not usable (status={status}). Is `make run` "
                      f"running and device '{args.device}' online? Aborting.\n{err}", file=sys.stderr)
                return 1
            if status in TRANSIENT_STATUSES and attempt < args.retries:
                attempt += 1
                print(f"[{i}/{len(clips)}] retry {stem} (status {status}, attempt {attempt}/{args.retries})")
                time.sleep(1.5)
                continue
            print(f"[{i}/{len(clips)}] FAIL {stem}: status={status} {err}")
            failed += 1
            break

        time.sleep(args.gap)

    print(f"Done: {ok} ok, {failed} failed -> {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        # Most likely the --player binary is missing.
        print(f"error: command not found ({e}). Is the --player binary installed (e.g. afplay on macOS)?",
              file=sys.stderr)
        raise SystemExit(2) from e
    except KeyboardInterrupt as exc:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from exc
