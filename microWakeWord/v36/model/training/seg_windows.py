#!/usr/bin/env python3
"""recipe/seg_hardneg.py (the v32 builder of features_hardneg_speech) with the source and gain as arguments.

Same windows (3.2 s, 1.6 s hop, zero-padded tail), same clip names hn_<HHMMSS>_<iiii>.wav and the same
printed rms/peak summary. Two arguments were hardcoded in the original:
  - the source: the original globbed device_captures/room_bg_ch1_20261016/*.wav, which then held only the
    four ambient_ch1_* files; two main_room_* files were added to that directory later, so the v32 set is
    rebuilt with the glob ambient_ch1_*.wav;
  - the gain: the original applied the device's saturating x4 to raw ch1; the studio background of v36 is
    already ch1 x4 (the model input), so it is cut at gain 1.
The clip tag is the last 6-digit group of the file name (HHMMSS): for ambient_ch1_20261016_143253.wav it
is 143253 as in the original, for 20f83b0a332d_20261020_222742_ch1.wav it is 222742.
Usage: seg_windows.py <out_clip_dir> <holdout_basename_substr> <src_glob> <gain>"""
import sys, os, re, glob, numpy as np, soundfile as sf
OUT, HOLD, SRC_GLOB, GAIN = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
WIN = int(3.2 * 16000); HOP = int(1.6 * 16000)
os.makedirs(OUT, exist_ok=True)
assert not glob.glob(f"{OUT}/*.wav"), f"REFUSE: {OUT} already has wavs — pick a fresh dir"

def load_i16(p):
    d, sr = sf.read(p, dtype="int16"); assert sr == 16000
    return (d[:, 0] if d.ndim > 1 else d).astype(np.int16)
def gain(x, g):
    if g == 1: return x
    return np.clip(x.astype(np.int32) * g, -32768, 32767).astype(np.int16)

files = sorted(glob.glob(SRC_GLOB))
train_files = [f for f in files if HOLD not in os.path.basename(f)]
held = [f for f in files if HOLD in os.path.basename(f)]
assert held, f"holdout substr {HOLD} matched no file — refuse (would leak eval into train)"
assert train_files, "no training files left"
print(f"TRAIN files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
print(f"HELD-OUT (excluded, for FAPH eval): {[os.path.basename(f) for f in held]}")

n = 0; rms_all = []; pk_all = []
for f in train_files:
    x = gain(load_i16(f), GAIN); tag = re.findall(r"\d{6}", os.path.basename(f))[-1]
    for i, s in enumerate(range(0, max(1, len(x) - WIN + 1), HOP)):
        seg = x[s:s + WIN]
        if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
        rms = float(np.sqrt(np.mean((seg.astype(np.float64)) ** 2))); pk = int(np.abs(seg).max())
        rms_all.append(rms); pk_all.append(pk)
        sf.write(f"{OUT}/hn_{tag}_{i:04d}.wav", seg, 16000, subtype="PCM_16"); n += 1
rms_all = np.array(rms_all); pk_all = np.array(pk_all)
print(f"WROTE {n} hard-neg clips ({WIN/16000:.1f}s win, {HOP/16000:.1f}s hop, x{GAIN} gain) -> {OUT}")
print(f"clip rms: p10 {np.percentile(rms_all,10):.1f} p50 {np.percentile(rms_all,50):.1f} "
      f"p90 {np.percentile(rms_all,90):.1f} max {rms_all.max():.1f}")
print(f"clip peak: p50 {int(np.percentile(pk_all,50))} p90 {int(np.percentile(pk_all,90))} max {pk_all.max()} "
      f"(sat@32767: {int((pk_all>=32767).sum())} clips)")
