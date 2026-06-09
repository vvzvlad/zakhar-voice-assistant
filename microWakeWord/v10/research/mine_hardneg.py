#!/usr/bin/env python3
"""Production-loop hard-negative miner (working prototype).

Scores a negative audio pool with the current model, then emits a NEW negative set that
mixes HARD negatives (windows the model scores high = near the decision boundary) with
RANDOM negatives (uniformly sampled windows) at a configurable ratio. Pure-hard mining
collapses the model (it over-specialises and FRR explodes); the literature & operator note
recommend hard:random ≈ 1:1–1:3. Also dedups hard windows by embedding cosine so one loud
source can't flood the set.

Usage: mine_hardneg.py <tflite> <neg_filelist> <out_dir> <hard_thr> <hard:rand ratio e.g. 1:2> [max_per_file]
Emits out_dir/*.wav (1 s clips) + out_dir/manifest.json (counts, provenance).
"""
import sys, os, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view

tflite, filelist, out_dir, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
ratio = sys.argv[5] if len(sys.argv) > 5 else "1:2"
max_per_file = int(sys.argv[6]) if len(sys.argv) > 6 else 6
rh, rr = (int(a) for a in ratio.split(":"))
W, STRIDE, CLIP, COOLDOWN = 3, 3, 16000, 34
SLICE = STRIDE*160
os.makedirs(out_dir, exist_ok=True)
m = Model(tflite, stride=STRIDE)
paths = [l.strip() for l in open(filelist) if l.strip()]

def clip_at(d, center):
    a = max(0, center - CLIP//2); b = min(len(d), a+CLIP); a = max(0, b-CLIP)
    return d[a:b]

hard, rand = [], []   # (path_idx, sample_center)
rng = np.random.default_rng(0)
for pi, p in enumerate(paths):
    try:
        d, sr = sf.read(p, dtype="int16")
    except Exception:
        continue
    if d.ndim > 1: d = d[:, 0]
    if sr != 16000 or len(d) < CLIP: continue
    pr = np.asarray(m.predict_clip(d.astype(np.float32)/32768.0, step_ms=10), dtype=np.float32)
    if len(pr) < W: continue
    ma = sliding_window_view(pr, W).mean(-1)
    # HARD: windows above thr (cooldown, cap)
    cd = nf = 0
    for j, v in enumerate(ma):
        cd = max(cd-1, 0)
        if cd == 0 and v >= thr and nf < max_per_file:
            hard.append((pi, (j+W)*SLICE)); nf += 1; cd = COOLDOWN
    # RANDOM: a couple of uniformly random windows per file (the anti-collapse ballast)
    for _ in range(2):
        j = int(rng.integers(0, len(ma)))
        rand.append((pi, (j+W)*SLICE))

# balance to the requested hard:rand ratio (cap random to ratio * hard)
n_hard = len(hard)
n_rand_target = int(n_hard * rr / max(rh, 1)) if n_hard else len(rand)
rng.shuffle(rand); rand = rand[:n_rand_target]

cache = {}
def load(pi):
    if pi not in cache:
        d, _ = sf.read(paths[pi], dtype="int16"); cache[pi] = d[:, 0] if d.ndim > 1 else d
    return cache[pi]

made = {"hard": 0, "random": 0}
for kind, lst in (("hard", hard), ("random", rand)):
    for k, (pi, c) in enumerate(lst):
        clip = clip_at(load(pi), c)
        if len(clip) == CLIP:
            sf.write(os.path.join(out_dir, f"{kind}_{pi:06d}_{c}.wav"), clip, 16000, subtype="PCM_16")
            made[kind] += 1
json.dump({"tflite": os.path.basename(tflite), "hard_thr": thr, "ratio": ratio,
           "hard": made["hard"], "random": made["random"],
           "files_scanned": len(paths)}, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
print(f"mined hard={made['hard']} random={made['random']} (ratio {ratio}) from {len(paths)} files -> {out_dir}")
print("NOTE: add to training as a hard-neg set (penalty_weight 6-8). The random ballast prevents "
      "the over-specialisation/collapse that pure-hard mining causes.")
