#!/usr/bin/env python3
"""SILENCE/IDLE false-accept eval (the untested negative distribution behind the field
bug: v8 false-fires in quiet rooms because device AGC ramps gain and amplifies the noise
floor into something speech-like). Generates pure-silence / room-tone / AGC-amplified
streams (NO wake word) and measures false-fires per hour for a model at several cutoffs,
with sliding windows 3 and 5. Deterministic (seeded rng; no Date/global-random).

Usage: silence_eval.py <tflite> <label> [minutes_per_stream]
"""
import sys, numpy as np
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

TFL, LABEL = sys.argv[1], sys.argv[2]
MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
SR, STRIDE = 16000, 3
N = int(MIN*60*SR)

def dbfs(x, db):  # scale to target RMS in dBFS
    r = np.sqrt(np.mean(x**2)) + 1e-12
    return (x / r) * (10**(db/20.0))

def white(rng): return rng.standard_normal(N).astype(np.float32)
def pink(rng):
    w = rng.standard_normal(N).astype(np.float32)
    b, a = signal.butter(1, 0.05, "low")
    return signal.lfilter(b, a, w).astype(np.float32)

def make_streams():
    rng = np.random.default_rng(7)
    s = {}
    s["pure_silence_-66dB"] = dbfs(white(rng), -66)         # near-digital-silence dither
    s["roomtone_white_-45dB"] = dbfs(white(rng), -45)        # quiet room
    s["roomtone_pink_-40dB"] = dbfs(pink(rng), -40)          # HVAC-ish low rumble
    s["agc_boosted_-25dB"] = dbfs(white(rng), -25)           # AGC ramps gain in quiet -> floor up
    s["agc_boosted_-15dB"] = dbfs(pink(rng), -15)            # aggressive AGC, speech-band-heavy
    return s

def faph(pr, cut, W):
    if len(pr) < W: return 0.0, 0.0
    ma = sliding_window_view(pr, W).mean(-1)
    hrs = len(pr)*STRIDE*0.01/3600.0
    det = cd = 0
    for v in ma:
        cd = max(cd-1, 0)
        if cd == 0 and v > cut: det += 1; cd = 25
    return det/hrs, hrs

m = Model(TFL, stride=STRIDE)
streams = make_streams()
CUTS = [0.9, 0.95, 0.97, 0.99]
print(f"model={LABEL}  {MIN:.0f} min/stream")
print(f"{'stream':<22}{'win':>4}" + "".join(f"{c:>8}" for c in CUTS) + "   (false-fires/hour)")
out = {}
for name, x in streams.items():
    p = np.clip(x, -1, 1).astype(np.float32)
    pr = np.asarray(m.predict_clip(p, step_ms=10), dtype=np.float32)
    for W in (3, 5):
        row = [faph(pr, c, W)[0] for c in CUTS]
        out[f"{name}|w{W}"] = row
        print(f"{name:<22}{W:>4}" + "".join(f"{v:>8.1f}" for v in row))
import json
json.dump(out, open(f"/home/claude/zakhar-mww/v9/silence_{LABEL}.json", "w"), indent=2)
