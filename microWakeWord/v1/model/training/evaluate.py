#!/usr/bin/env python3
"""Evaluate the quantized streaming microWakeWord model the same way the framework
does (stride = model input stride = 3; sliding-window mean; cooldown after accept),
on held-out real «Захар» recordings (recall) and on long ambient audio (FAPH).

Usage: evaluate.py <tflite> <heldout_dir> <ambient_wav> [out_json]
"""
import sys, os, glob, json
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model

tflite_path = sys.argv[1]
heldout_dir = sys.argv[2]
ambient_wav = sys.argv[3] if len(sys.argv) > 3 else None
out_json = sys.argv[4] if len(sys.argv) > 4 else None

STEP_MS = 10
STRIDE = 3            # = mixednet --stride (model input time stride)
STEP_S = STEP_MS / 1000.0
COOLDOWN = 25         # ignore_slices_after_accept (framework default)
CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
WINDOWS = [3, 5, 10]


def load_16k(path):
    d, sr = sf.read(path, dtype="int16")
    if d.ndim > 1:
        d = d[:, 0]
    assert sr == 16000, f"{path} sr={sr}"
    return d.astype(np.float32) / 32768.0


model = Model(tflite_path, stride=STRIDE)
print(f"Model input {model.input_details[0]['shape']} stride={model.stride}")

# --- Held-out positives: streaming probabilities per clip ---
heldout = sorted(glob.glob(os.path.join(heldout_dir, "*.wav")))
print(f"Held-out positives: {len(heldout)}")
pos_probs = []
for p in heldout:
    pr = np.asarray(model.predict_clip(load_16k(p), step_ms=STEP_MS), dtype=np.float32)
    pos_probs.append(pr)

# --- Ambient probabilities (one long track) ---
amb_ma_by_w = {}
amb_hours = 0.0
if ambient_wav and os.path.exists(ambient_wav):
    amb = load_16k(ambient_wav)
    amb_pr = np.asarray(model.predict_clip(amb, step_ms=STEP_MS), dtype=np.float32)
    # duration: each probability slice represents STRIDE*STEP_S seconds
    amb_hours = len(amb_pr) * STRIDE * STEP_S / 3600.0
    print(f"Ambient: {amb_hours:.3f} h, {len(amb_pr)} prob slices")


def recall_at(window, cutoff):
    hit = 0
    for pr in pos_probs:
        if len(pr) <= COOLDOWN + window:
            tail = pr
        else:
            tail = pr[COOLDOWN:]
        if len(tail) < window:
            ma_max = float(tail.mean()) if len(tail) else 0.0
        else:
            ma = sliding_window_view(tail, window).mean(axis=-1)
            ma_max = float(ma.max())
        if ma_max >= cutoff:
            hit += 1
    return hit / len(pos_probs)


def faph_at(window, cutoff, amb_pr):
    if len(amb_pr) < window:
        return 0.0
    ma = sliding_window_view(amb_pr, window).mean(axis=-1)
    det = 0
    cooldown = 0
    for v in ma:
        cooldown = max(cooldown - 1, 0)
        if cooldown == 0 and v > cutoff:
            det += 1
            cooldown = COOLDOWN
    return det / amb_hours if amb_hours > 0 else None


amb_pr = amb_pr if ambient_wav and os.path.exists(ambient_wav) else None
grid = []
print("\ncutoff  window  recall   faph")
for w in WINDOWS:
    for c in CUTOFFS:
        rec = recall_at(w, c)
        fp = faph_at(w, c, amb_pr) if amb_pr is not None else None
        grid.append({"cutoff": c, "window": w, "recall": rec, "faph": fp})
        print(f"{c:<6} {w:<7} {rec:0.3f}    {('%.2f'%fp) if fp is not None else 'n/a'}")

rec = None
feasible = [g for g in grid if g["faph"] is not None]
if feasible:
    good = [g for g in feasible if g["faph"] <= 1.0 and g["recall"] >= 0.85]
    pool = good or [g for g in feasible if g["faph"] <= 2.0 and g["recall"] >= 0.80] or feasible
    rec = sorted(pool, key=lambda g: (-g["recall"], g["faph"]))[0]
print("\nRECOMMENDED:", json.dumps(rec) if rec else "n/a")
if out_json:
    json.dump({"grid": grid, "recommended": rec, "ambient_hours": amb_hours,
               "n_heldout": len(heldout)}, open(out_json, "w"), indent=2)
    print("wrote", out_json)
