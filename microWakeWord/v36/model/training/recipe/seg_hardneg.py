#!/usr/bin/env python3
"""Segment the operator's REAL room-background call into serve-level hard-negative clips for re-finetune.
Device-faithful: ch1 int16 -> clamp(4x) (the exact serve transform that reproduced his 26/50), then
~3.2 s windows with 1.6 s hop written as .wav. These are NEGATIVES (his conversational speech, NOT «захар»)
so the model learns «захар» != his general speech. LEAKAGE-SAFE: the held-out eval file is EXCLUDED here.
Usage: seg_hardneg.py <out_clip_dir> <holdout_basename_substr>   Prints a count + rms/peak distribution."""
import sys, os, glob, numpy as np, soundfile as sf
ROOT = "/home/claude/zakhar-mww"
SRC = f"{ROOT}/device_captures/room_bg_ch1_20261016"
OUT = sys.argv[1]; HOLD = sys.argv[2] if len(sys.argv) > 2 else "145427"
WIN = int(3.2 * 16000); HOP = int(1.6 * 16000)
os.makedirs(OUT, exist_ok=True)
assert not glob.glob(f"{OUT}/*.wav"), f"REFUSE: {OUT} already has wavs — pick a fresh dir"

def load_i16(p):
    d, sr = sf.read(p, dtype="int16"); assert sr == 16000
    return (d[:, 0] if d.ndim > 1 else d).astype(np.int16)
def gain4(x): return np.clip(x.astype(np.int32) * 4, -32768, 32767).astype(np.int16)

files = sorted(glob.glob(f"{SRC}/*.wav"))
train_files = [f for f in files if HOLD not in os.path.basename(f)]
held = [f for f in files if HOLD in os.path.basename(f)]
assert held, f"holdout substr {HOLD} matched no file — refuse (would leak eval into train)"
assert train_files, "no training files left"
print(f"TRAIN files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
print(f"HELD-OUT (excluded, for FAPH eval): {[os.path.basename(f) for f in held]}")

n = 0; rms_all = []; pk_all = []
for f in train_files:
    x = gain4(load_i16(f)); tag = os.path.basename(f).split('_')[-1].split('.')[0]
    for i, s in enumerate(range(0, max(1, len(x) - WIN + 1), HOP)):
        seg = x[s:s + WIN]
        if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
        rms = float(np.sqrt(np.mean((seg.astype(np.float64)) ** 2))); pk = int(np.abs(seg).max())
        rms_all.append(rms); pk_all.append(pk)
        sf.write(f"{OUT}/hn_{tag}_{i:04d}.wav", seg, 16000, subtype="PCM_16"); n += 1
rms_all = np.array(rms_all); pk_all = np.array(pk_all)
print(f"WROTE {n} hard-neg clips ({WIN/16000:.1f}s win, {HOP/16000:.1f}s hop, x4 serve gain) -> {OUT}")
print(f"clip rms: p10 {np.percentile(rms_all,10):.1f} p50 {np.percentile(rms_all,50):.1f} "
      f"p90 {np.percentile(rms_all,90):.1f} max {rms_all.max():.1f}")
print(f"clip peak: p50 {int(np.percentile(pk_all,50))} p90 {int(np.percentile(pk_all,90))} max {pk_all.max()} "
      f"(sat@32767: {int((pk_all>=32767).sum())} clips)")

