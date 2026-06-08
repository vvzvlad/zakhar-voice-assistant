#!/usr/bin/env python3
"""Adversarial false-trigger mining: run the current best model over audio that does
NOT contain «Захар»; wherever the windowed probability crosses a threshold, extract
the ~1.1s audio window as a hard-negative clip. (No real positives in the corpus, so
every high-prob window is a genuine false trigger — safe to mine; we still cap per
file and cooldown to avoid flooding from one track.)

Usage: mine_false.py <tflite> <filelist.txt> <out_dir> <thr> <tag> [max_per_file]
"""
import sys, os
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model

tflite, filelist, out_dir, thr, tag = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
max_per_file = int(sys.argv[6]) if len(sys.argv) > 6 else 6
W = 3
STRIDE = 3
SLICE = STRIDE * 160          # samples per prob slice (30 ms)
CLIP = 16000                  # 1.0 s hard-neg clip
COOLDOWN = 34                 # ~1 s between mined peaks in one file
os.makedirs(out_dir, exist_ok=True)
model = Model(tflite, stride=STRIDE)
paths = [l.strip() for l in open(filelist) if l.strip()]

made = 0
for pi, p in enumerate(paths):
    try:
        d, sr = sf.read(p, dtype="int16")
    except Exception:
        continue
    if d.ndim > 1:
        d = d[:, 0]
    if sr != 16000 or len(d) < CLIP:
        continue
    df = d.astype(np.float32) / 32768.0
    pr = np.asarray(model.predict_clip(df, step_ms=10), dtype=np.float32)
    if len(pr) < W:
        continue
    ma = sliding_window_view(pr, W).mean(-1)
    cd = 0
    n_file = 0
    for j, v in enumerate(ma):
        cd = max(cd - 1, 0)
        if cd == 0 and v >= thr and n_file < max_per_file:
            center = (j + W) * SLICE
            a = max(0, center - CLIP // 2)
            b = min(len(d), a + CLIP)
            a = max(0, b - CLIP)
            clip = d[a:b]
            if len(clip) == CLIP:
                sf.write(os.path.join(out_dir, f"mine_{tag}_{pi:06d}_{j:05d}.wav"),
                         clip, 16000, subtype="PCM_16")
                made += 1
                n_file += 1
            cd = COOLDOWN
print(f"{tag}: mined {made} hard-neg clips from {len(paths)} files (thr={thr})")
