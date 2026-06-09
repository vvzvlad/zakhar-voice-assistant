#!/usr/bin/env python3
"""G3: quantify the PTQ headroom. Compare the FLOAT (keras, non-streaming) student
vs the INT8 streaming tflite on the device held-out positives. A large float→int8
FRR gap = QAT could help; a ~0 gap = PTQ is already lossless and QAT is not worth it.

Usage: quant_gap.py <weights.h5> <int8_tflite>
"""
import sys, glob, numpy as np
sys.path.insert(0, "/home/claude/zakhar-mww")
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from v9.extract_embeddings import extract_dir
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
import soundfile as sf

W, TFL = sys.argv[1], sys.argv[2]
POS = sorted(glob.glob("/home/claude/zakhar-mww/v8/dev_heldout_pos/*.wav"))

# FLOAT non-streaming: max windowed prob per clip (extract_dir returns argmax-window prob)
fprobs, _ = extract_dir(W, POS)

# INT8 streaming: same maxavg(window=3) as evaluate_device.py
m = Model(TFL, stride=3)
def load16k(p):
    d, sr = sf.read(p, dtype="int16")
    if d.ndim > 1: d = d[:, 0]
    return d.astype(np.float32)/32768.0
def maxavg(c):
    pr = np.asarray(m.predict_clip(c, step_ms=10), dtype=np.float32)
    return float(sliding_window_view(pr,3).mean(-1).max()) if len(pr) >= 3 else (float(pr.mean()) if len(pr) else 0)
iprobs = np.array([maxavg(load16k(p)) for p in POS])

print(f"clips: {len(POS)}")
print("cutoff   float_FRR   int8_FRR   gap(int8-float)")
for c in [0.5, 0.7, 0.8, 0.9, 0.95]:
    ff = float((fprobs < c).mean()); fi = float((iprobs < c).mean())
    print(f"{c:<8} {ff:0.3f}      {fi:0.3f}      {fi-ff:+0.3f}")
print(f"\nmean prob: float={fprobs.mean():.3f} int8={iprobs.mean():.3f}")
print("NOTE: float=non-streaming, int8=streaming, so gap mixes quant + streaming-state effects (upper bound on QAT headroom).")
