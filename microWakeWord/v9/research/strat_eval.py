#!/usr/bin/env python3
"""Stratified programmatic eval: degrade the held-out device positives into
acoustic modes (far-field reverb / media mix / babble / quiet / muffled /
Lombard tilt / combined) and report per-mode FRR for a given tflite. This
exposes worst-case modes BEFORE real long-tail device data arrives.

Usage: strat_eval.py <tflite> [out.json]
Deterministic: per-file degradation seeded by file index (no Date/random global).
"""
import sys, glob, json, os
import numpy as np, soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model

TFLITE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
STRIDE, W, SR = 3, 3, 16000
POS = sorted(glob.glob("/home/claude/zakhar-mww/v8/dev_heldout_pos/*.wav"))
RIRS = sorted(glob.glob("/home/claude/zakhar-mww/rirs_v9/*.wav"))
MUSIC = sorted(glob.glob("/home/claude/zakhar-mww/fma_16k/*.wav"))
# babble/TV bed: the long held-out device-neg stream (real room+media)
BABBLE_SRC = "/home/claude/zakhar-mww/v8/device_neg_test.wav"

def load16k(p):
    d, sr = sf.read(p, dtype="int16")
    if d.ndim > 1: d = d[:, 0]
    d = d.astype(np.float32) / 32768.0
    if sr != SR:
        d = signal.resample_poly(d, SR, sr)
    return d

_babble = load16k(BABBLE_SRC)

def rms(x): return float(np.sqrt(np.mean(x**2)) + 1e-9)

def mix_snr(sig, noise, snr_db, rng):
    if len(noise) < len(sig):
        noise = np.tile(noise, int(np.ceil(len(sig)/len(noise))))
    off = rng.integers(0, max(1, len(noise)-len(sig)))
    noise = noise[off:off+len(sig)]
    g = rms(sig) / (rms(noise) * (10**(snr_db/20.0)))
    out = sig + g*noise
    p = np.max(np.abs(out))
    return out/p*0.97 if p > 1 else out

def reverb(sig, rng):
    rir = load16k(RIRS[rng.integers(0, len(RIRS))])
    rir = rir / (np.max(np.abs(rir))+1e-9)
    pk = int(np.argmax(np.abs(rir)))
    out = signal.fftconvolve(sig, rir)[pk:pk+len(sig)]  # align direct path
    p = np.max(np.abs(out)); return out/p*0.97 if p > 0 else out

def lowpass(sig, fc):  # muffled far-field
    b, a = signal.butter(4, fc/(SR/2), "low"); return signal.lfilter(b, a, sig).astype(np.float32)

def hshelf(sig, gain_db):  # crude Lombard: raise highs (spectral tilt up)
    b, a = signal.butter(2, 1500/(SR/2), "high")
    hi = signal.lfilter(b, a, sig)
    return (sig + (10**(gain_db/20.0)-1)*hi).astype(np.float32)

MODES = {
    "clean":        lambda s, rng: s,
    "reverb":       lambda s, rng: reverb(s, rng),
    "music_snr10":  lambda s, rng: mix_snr(s, load16k(MUSIC[rng.integers(0,len(MUSIC))]), 10, rng),
    "music_snr5":   lambda s, rng: mix_snr(s, load16k(MUSIC[rng.integers(0,len(MUSIC))]), 5, rng),
    "babble_snr10": lambda s, rng: mix_snr(s, _babble, 10, rng),
    "quiet_-18dB":  lambda s, rng: s*(10**(-18/20.0)),
    "muffled_lp3k": lambda s, rng: lowpass(s, 3000),
    "lombard_+6dB": lambda s, rng: hshelf(s, 6),
    "reverb+music10": lambda s, rng: mix_snr(reverb(s, rng), load16k(MUSIC[rng.integers(0,len(MUSIC))]), 10, rng),
}
CUTS = [0.5, 0.7, 0.8, 0.9, 0.95]

m = Model(TFLITE, stride=STRIDE)
def score(clip):
    pr = np.asarray(m.predict_clip(clip, step_ms=10), dtype=np.float32)
    if len(pr) >= W: return float(sliding_window_view(pr, W).mean(-1).max())
    return float(pr.mean()) if len(pr) else 0.0

print(f"tflite: {os.path.basename(TFLITE)}  positives: {len(POS)}  RIRs:{len(RIRS)} music:{len(MUSIC)}")
res = {}
for mode, fn in MODES.items():
    scores = []
    for i, p in enumerate(POS):
        rng = np.random.default_rng(1000+i)  # deterministic per-file
        scores.append(score(fn(load16k(p), rng)))
    scores = np.array(scores)
    res[mode] = {f"{c}": float((scores < c).mean()) for c in CUTS}  # FRR
    res[mode]["mean_score"] = float(scores.mean())
print("\nFRR per mode (rows=mode, cols=cutoff):")
print(f"{'mode':<16}" + "".join(f"{c:>8}" for c in CUTS) + f"{'mean_p':>9}")
for mode in MODES:
    r = res[mode]
    print(f"{mode:<16}" + "".join(f"{r[str(c)]:>8.3f}" for c in CUTS) + f"{r['mean_score']:>9.3f}")
# worst-case row at cutoff 0.9
worst = max(MODES, key=lambda mo: res[mo]["0.9"])
print(f"\nWORST-CASE @0.9: {worst}  FRR={res[worst]['0.9']:.3f}")
if OUT:
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)
