#!/usr/bin/env python3
"""Device-faithful offline eval for the CORRECT mWW input channel.
Firmware (home-assistant-voice.yaml@26.5.0): micro_wake_word eats i2s_mics CHANNEL 1 x gain_factor 4.
gain_factor=4 for a 16-bit source reduces BIT-EXACT to clamp(4*x, -32768, 32767) (verified 0 mismatches
over all int16). We apply the gain on int16 and feed the result STRAIGHT into generate_features_for_clip
(which, for an int16 dtype, skips the float *32768 round-trip) -> same pymicro-features MicroFrontend as
ESP32 -> streaming tflite. Fresh state per clip. win5-mean trace, 25-step (750ms) lockout, count_det.

Usage: device_faithful_stream.py <tflite> <label> <filelist_or_glob> <gain:int> [channel:int]
  gain=1  -> no amplification (use when the recording is ALREADY post-gain, e.g. a direct mWW-source dump)
  gain=4  -> apply the device's saturating x4 (use for RAW channel-1 int16 without gain)
  channel -> for multi-channel wavs, which column to take (default 0; for our ch1 captures the file is
             typically already mono ch1, so leave 0)
"""
import sys, os, glob, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view

STRIDE = 3; W = 5; LOCK = 25
CUTS = [0.40,0.50,0.60,0.70,0.72,0.74,0.76,0.78,0.80,0.85,0.90,0.95]
TFL, LAB, SRC = sys.argv[1], sys.argv[2], sys.argv[3]
GAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 4
CHAN = int(sys.argv[5]) if len(sys.argv) > 5 else 0

def apply_gain_i16(x_i16, g):
    if g == 1:
        return x_i16.astype(np.int16)
    return np.clip(x_i16.astype(np.int32) * g, -32768, 32767).astype(np.int16)

def load_i16(p, chan):
    d, sr = sf.read(p, dtype="int16")
    if d.ndim > 1:
        d = d[:, chan] if chan < d.shape[1] else d[:, 0]
    assert sr == 16000, f"{p}: sr={sr} (expected 16000)"
    return d.astype(np.int16)

def count_det(ma, cut):
    det = 0; cd = 0
    for v in ma:
        cd = max(cd - 1, 0)
        if cd == 0 and v >= cut:
            det += 1; cd = LOCK
    return det

def files_from(src):
    if os.path.isfile(src) and src.endswith(".txt"):
        return [l.strip() for l in open(src) if l.strip() and not l.startswith("#")]
    if os.path.isdir(src):
        src = os.path.join(src, "*.wav")
    return sorted(p for p in glob.glob(src) if not os.path.basename(p).startswith("._"))

def main():
    files = files_from(SRC)
    assert files, f"no files from {SRC}"
    rows = []
    for p in files:
        x = apply_gain_i16(load_i16(p, CHAN), GAIN)
        m = Model(TFL, stride=STRIDE)  # fresh state per clip
        pr = np.asarray(m.predict_clip(x, step_ms=10), dtype=np.float32)
        ma = sliding_window_view(pr, W).mean(-1) if len(pr) >= W else np.array([0.0])
        rows.append({"f": os.path.basename(p), "n": len(x),
                     "maxwin": round(float(ma.max()), 3),
                     "det": {f"{c:.2f}": count_det(ma, c) for c in CUTS}})
    n = len(rows)
    fire = lambda c: sum(1 for r in rows if r["det"][f"{c:.2f}"] > 0)
    print("DFJSON " + json.dumps({"label": LAB, "gain": GAIN, "chan": CHAN,
                                  "n": n, "cuts": CUTS, "rows": rows}, ensure_ascii=False))
    print(f"\n=== {LAB}  gain={GAIN} chan={CHAN}  n={n} (fresh state/clip, win5, lockout {LOCK}) ===")
    print("cut   clips_firing  rate")
    for c in CUTS:
        f = fire(c)
        print(f"{c:.2f}   {f:4d}/{n:<4d}   {f/n:.3f}")
    mw = np.array([r["maxwin"] for r in rows])
    print(f"maxwin distribution: min {mw.min():.3f}  med {np.median(mw):.3f}  "
          f"p90 {np.percentile(mw,90):.3f}  max {mw.max():.3f}")

if __name__ == "__main__":
    main()
