#!/usr/bin/env python3
"""Prep the studio «захар» take (Voice PE speaker 20f83b0a332d, ch1 at gain 1) for the v34 finetune.
Adapted from panel_prep.py (v33):
  - segmentation is the timeline.py energy finder (30 ms RMS frames, threshold p10 + 15 dB, >= 6 frames,
    8-frame gap bridging), so the held-out words are a subset of the same 53 segments the baseline counts;
  - word-like filter as panel_prep.py: 0.18 <= dur <= 0.90 s AND peak frame RMS >= 3x noise
    (noise = median frame RMS, linear);
  - take-disjoint split by md5(onset rounded to 0.01 s), first len//3 held out;
  - TRAIN crops [onset-0.12 s, end+0.20 s] get the device's saturating x4 (clamp(4*x)), the same gain
    domain as features_real_pos (prep_single_voice.py applies x4 to the speaker crops).
Does not build features or train.
Usage: studio_prep.py <take.wav> <OUT_ROOT>
"""
import sys, os, hashlib, numpy as np, soundfile as sf

SR = 16000; FR = 480; STEP = 0.03
src, out = sys.argv[1], os.path.abspath(sys.argv[2])
raw, sr = sf.read(src, dtype="int16")
assert sr == SR and raw.ndim == 1, (sr, raw.shape)

# --- timeline.py energy segmentation ---
f = len(raw) // FR
frames = raw[:f * FR].astype(np.float64).reshape(f, FR)
rms = np.sqrt((frames ** 2).mean(1))
r = 20 * np.log10(np.sqrt(((raw[:f * FR].astype(np.float32) / 32768).reshape(f, FR) ** 2).mean(1)) + 1e-9)
act = r > np.percentile(r, 10) + 15
segs, i = [], 0
while i < f:
    if act[i]:
        j = i
        while j < f and act[j:j + 8].any(): j += 1
        if j - i >= 6: segs.append((i, j))
        i = j
    else: i += 1
noise = float(np.median(rms))

# --- word-like filter (panel_prep.py thresholds) ---
rows = []
for k, (i0, i1) in enumerate(segs):
    a, b = round(i0 * STEP, 2), round(i1 * STEP, 2)
    dw = round((i1 - i0) * STEP, 2)
    pm = round(float(rms[i0:i1].max() / noise), 1)
    ok = (0.18 <= dw <= 0.90) and (pm >= 3.0)
    rows.append({"k": k, "a": a, "b": b, "dur": dw, "pm": pm, "split": "kept" if ok else "dropped"})
kept = [x for x in rows if x["split"] == "kept"]

# --- take-disjoint split by md5(onset) ---
order = sorted(kept, key=lambda x: hashlib.md5(f"{x['a']}".encode()).hexdigest())
n_ho = max(1, len(kept) // 3)
for n, x in enumerate(order):
    x["split"] = "heldout" if n < n_ho else "train"

tr_dir = os.path.join(out, "studio_pos_train"); os.makedirs(tr_dir, exist_ok=True)
for x in rows:
    if x["split"] != "train": continue
    i0 = max(0, int((x["a"] - 0.12) * SR)); i1 = min(len(raw), int((x["b"] + 0.20) * SR))
    w = np.clip(raw[i0:i1].astype(np.int32) * 4, -32768, 32767).astype(np.int16)
    sf.write(os.path.join(tr_dir, f"studio_{x['k']:03d}.wav"), w, SR, subtype="PCM_16")

with open(os.path.join(out, "studio_split_manifest.csv"), "w") as fh:
    fh.write("seg,a_s,b_s,dur_s,peak_mult,split\n")
    for x in rows: fh.write(f"{x['k']},{x['a']},{x['b']},{x['dur']},{x['pm']},{x['split']}\n")

ntr = sum(x["split"] == "train" for x in rows); nho = sum(x["split"] == "heldout" for x in rows)
dws = sorted(x["dur"] for x in kept)
print(f"STUDIO_PREP: segments={len(segs)} kept(word-like)={len(kept)} dropped={len(segs) - len(kept)} noise_rms={noise:.2f}")
print(f"  kept dur s: min={dws[0]:.2f} med={dws[len(dws) // 2]:.2f} max={dws[-1]:.2f}")
print(f"  split: train={ntr} heldout={nho} -> {tr_dir} (x4 crops)")
print(f"  dropped (seg,a,b,dur,peakx): {[(x['k'], x['a'], x['b'], x['dur'], x['pm']) for x in rows if x['split'] == 'dropped']}")
print("STUDIO_PREP_DONE")
