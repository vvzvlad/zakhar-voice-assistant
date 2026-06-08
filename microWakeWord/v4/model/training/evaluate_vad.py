#!/usr/bin/env python3
"""Evaluate held-out real recall + ambient FAPH WITH and WITHOUT a VAD pre-gate.
A wake-word detection is only counted (VAD-gated) if its time falls inside a
Silero-VAD speech segment (precomputed JSON). This mirrors ESPHome micro_wake_word
`vad:` which blocks KWS when there's no speech — continuous instrumental music is
non-speech and gets gated before the KWS decision.

Usage: evaluate_vad.py <tflite> <heldout_dir> <ambient_wav> <vad_json> [out_json]
"""
import sys, os, glob, json
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model

tflite_path, heldout_dir, ambient_wav, vad_json = sys.argv[1:5]
out_json = sys.argv[5] if len(sys.argv) > 5 else None
STEP_MS, STRIDE = 10, 3
STEP_S = STEP_MS/1000.0
SLICE_SAMPLES = int(STRIDE * STEP_MS/1000 * 16000)  # 480 samples per prob slice
COOLDOWN = 25
CUTOFFS = [0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99]
WINDOWS = [3,5,10]

def load16k(p):
    d,sr=sf.read(p,dtype="int16")
    if d.ndim>1: d=d[:,0]
    assert sr==16000
    return d.astype(np.float32)/32768.0

model = Model(tflite_path, stride=STRIDE)

# held-out positive probs
heldout=sorted(glob.glob(os.path.join(heldout_dir,"*.wav")))
pos=[np.asarray(model.predict_clip(load16k(p),step_ms=STEP_MS),dtype=np.float32) for p in heldout]

amb=load16k(ambient_wav)
amb_pr=np.asarray(model.predict_clip(amb,step_ms=STEP_MS),dtype=np.float32)
amb_hours=len(amb_pr)*STRIDE*STEP_S/3600.0

# VAD speech mask over samples -> boolean per prob-slice
vad=json.load(open(vad_json))
speech=np.zeros(vad["n_samples"]+SLICE_SAMPLES, dtype=bool)
for s,e in vad["segments"]:
    speech[s:e]=True
def slice_is_speech(j):
    a=j*SLICE_SAMPLES; b=a+SLICE_SAMPLES
    return speech[a:b].any() if a < len(speech) else False

def recall_at(w,c):
    hit=0
    for pr in pos:
        tail=pr[COOLDOWN:] if len(pr)>COOLDOWN+w else pr
        if len(tail)<w: m=float(tail.mean()) if len(tail) else 0.0
        else: m=float(sliding_window_view(tail,w).mean(-1).max())
        if m>=c: hit+=1
    return hit/len(pos)

def faph_at(w,c,gated):
    if len(amb_pr)<w: return 0.0
    ma=sliding_window_view(amb_pr,w).mean(-1)
    det=0; cd=0
    for j,v in enumerate(ma):
        cd=max(cd-1,0)
        if cd==0 and v>c:
            if (not gated) or slice_is_speech(j+w):
                det+=1; cd=COOLDOWN
            else:
                cd=COOLDOWN  # still suppress re-trigger window even if gated
    return det/amb_hours if amb_hours>0 else None

rows=[]
print(f"ambient {amb_hours:.3f}h | VAD speech fraction {100*sum(e-s for s,e in vad['segments'])/vad['n_samples']:.0f}%")
print("cutoff  win  recall  faph_noVAD  faph_VAD")
for w in WINDOWS:
    for c in CUTOFFS:
        r=recall_at(w,c); f0=faph_at(w,c,False); f1=faph_at(w,c,True)
        rows.append({"cutoff":c,"window":w,"recall":r,"faph":f0,"faph_vad":f1})
        print(f"{c:<6} {w:<4} {r:0.3f}   {f0:8.2f}   {f1:8.2f}")
if out_json: json.dump({"grid":rows,"ambient_hours":amb_hours},open(out_json,"w"),indent=2)
