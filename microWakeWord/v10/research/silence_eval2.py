#!/usr/bin/env python3
"""v5 vs v8 false-fires on REAL available negative streams + a syllable-modulated noise
proxy (3-7 Hz amplitude modulation = the speech-like envelope that fools KWS on an
AGC-amplified floor). Reports false-fires/hour at deploy-relevant cutoffs & windows.

Usage: silence_eval2.py <tflite> <label>
"""
import sys, glob, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view

TFL, LABEL = sys.argv[1], sys.argv[2]
SR, STRIDE = 16000, 3

def load(p):
    d, sr = sf.read(p, dtype="int16"); d = d[:,0] if d.ndim>1 else d
    return d.astype(np.float32)/32768.0

def dbfs(x, db):
    r = np.sqrt(np.mean(x**2))+1e-12; return (x/r)*(10**(db/20.0))

# syllable-modulated noise: white noise * (slow 3-7Hz envelope) -> speech-like rhythm
def syllable_noise(seconds, db, seed):
    rng = np.random.default_rng(seed); n = seconds*SR
    w = rng.standard_normal(n).astype(np.float32)
    t = np.arange(n)/SR
    env = np.zeros(n, np.float32)
    for f in (3.0, 4.5, 6.0):  # syllabic rates
        env += (0.5+0.5*np.sin(2*np.pi*f*t + rng.uniform(0,6))).astype(np.float32)
    env /= env.max()
    return dbfs(np.clip(w*env, -1, 1), db)

streams = {}
streams["REAL_device_neg(6min)"] = load("/home/claude/zakhar-mww/v8/device_neg_test.wav")
streams["REAL_device_neg_AGC+20dB"] = np.clip(dbfs(load("/home/claude/zakhar-mww/v8/device_neg_test.wav"), -18), -1, 1)
streams["REAL_ambient_music(40min)"] = load("/home/claude/zakhar-mww/ambient_music.wav")
streams["syllable_mod_-25dB"] = syllable_noise(600, -25, 11)
streams["syllable_mod_-15dB"] = syllable_noise(600, -15, 12)

def faph(pr, cut, W):
    if len(pr)<W: return 0.0
    ma = sliding_window_view(pr, W).mean(-1); hrs = len(pr)*STRIDE*0.01/3600.0
    det=cd=0
    for v in ma:
        cd=max(cd-1,0)
        if cd==0 and v>cut: det+=1; cd=25
    return det/hrs

m = Model(TFL, stride=STRIDE)
CUTS=[0.9,0.95,0.97]
print(f"model={LABEL}")
print(f"{'stream':<26}{'win':>4}"+"".join(f"{c:>8}" for c in CUTS)+"  (FF/hour)")
import json; out={}
for name,x in streams.items():
    pr=np.asarray(m.predict_clip(np.clip(x,-1,1).astype(np.float32),step_ms=10),dtype=np.float32)
    for W in (3,5):
        row=[faph(pr,c,W) for c in CUTS]; out[f"{name}|w{W}"]=row
        print(f"{name:<26}{W:>4}"+"".join(f"{v:>8.1f}" for v in row))
json.dump(out, open(f"/home/claude/zakhar-mww/v9/silence2_{LABEL}.json","w"), indent=2)
