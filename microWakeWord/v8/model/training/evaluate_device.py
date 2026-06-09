#!/usr/bin/env python3
"""Device-eval: FRR on held-out device positives + FAPH on held-out device negatives,
through the real tract. With and without VAD gate. Window 3."""
import sys, glob, json
import numpy as np, soundfile as sf
sys.path.insert(0,"/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
tflite=sys.argv[1]; vadj=sys.argv[2] if len(sys.argv)>2 else None
STRIDE=3; SLICE=STRIDE*160; COOLDOWN=25; W=3
def load16k(p):
    d,sr=sf.read(p,dtype="int16")
    if d.ndim>1:d=d[:,0]
    return d.astype(np.float32)/32768.0
m=Model(tflite,stride=STRIDE)
def maxavg(c):
    pr=np.asarray(m.predict_clip(c,step_ms=10),dtype=np.float32)
    return float(sliding_window_view(pr,W).mean(-1).max()) if len(pr)>=W else (float(pr.mean()) if len(pr) else 0)
pos=sorted(glob.glob("v8/dev_heldout_pos/*.wav"))
pmax=np.array([maxavg(load16k(p)) for p in pos])
amb=load16k("v8/device_neg_test.wav"); ap=np.asarray(m.predict_clip(amb,step_ms=10),dtype=np.float32)
amb_h=len(ap)*STRIDE*0.01/3600.0
ma=sliding_window_view(ap,W).mean(-1)
speech=None
if vadj:
    v=json.load(open(vadj)); speech=np.zeros(v["n_samples"]+SLICE,bool)
    for s,e in v["segments"]: speech[s:e]=True
print(f"device held-out positives: {len(pos)} | device-neg test: {amb_h*60:.1f} min")
print("cutoff  FRR    recall  FAPH_noVAD  FAPH_VAD")
for cut in [0.5,0.7,0.8,0.9,0.95,0.97,0.99]:
    rec=float((pmax>=cut).mean()); frr=1-rec
    det=detv=cd=0
    for j,val in enumerate(ma):
        cd=max(cd-1,0)
        if cd==0 and val>cut:
            det+=1
            if speech is not None:
                a=(j+W)*SLICE
                if speech[a:a+SLICE].any(): detv+=1
            cd=COOLDOWN
    f0=det/amb_h; f1=(detv/amb_h) if speech is not None else None
    print(f"{cut:<7} {frr:0.3f}  {rec:0.3f}   {f0:8.2f}   {('%.2f'%f1) if f1 is not None else 'n/a'}")
