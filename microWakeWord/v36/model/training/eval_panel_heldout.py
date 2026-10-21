#!/usr/bin/env python3
"""HONEST v33 panel eval. Stream full panel file x1 (firmware DC-blocked). Recall on the 19 HELD-OUT panel
words (v33 never trained on them — the honest number), plus the 58-word and 120-energy-event denominators
for comparability with the earlier report. @0.80 + sweep. Fresh Model per model. Numbers only."""
import sys,numpy as np,soundfile as sf,csv
sys.path.insert(0,"/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
ROOT="/home/claude/zakhar-mww"; STRIDE=3; W=5; LOCK=25; SR=16000
PR="/home/claude/zakhar-mww/v36/prep/panel_run"   # panel_prep.py output (rebuilt)
P=f"{ROOT}/device_captures/panel_round_zakhar_20261016/fc012cd1e5ec_20261016_171350.wav"
MODELS=dict(a.split(":",1) for a in sys.argv[1:])   # label:tflite ...
CUTS=[0.5,0.6,0.7,0.8,0.9,0.95]; HIT_PRE=0.3; HIT_POST=0.8
d,sr=sf.read(P,dtype="int16"); d=(d[:,0] if d.ndim>1 else d).astype(np.int16)
x=d.astype(np.float64); dur=len(d)/SR
FR=320;HOP=160; nf=1+(len(x)-FR)//HOP
fe=np.array([np.sqrt(np.mean(x[i*HOP:i*HOP+FR]**2)) for i in range(nf)]); ft=np.arange(nf)*HOP/SR
noise=np.median(fe); env=np.convolve(fe,np.ones(9)/9,mode="same")
pk,_=find_peaks(env,height=noise*3,distance=int(1.2*SR/HOP)); PK=ft[pk]
# denominators
HO=[(float(a),float(b)) for a,b in csv.reader(open(f"{PR}/panel_heldout_windows.csv")) if a!="a_s"]
ALL=[(float(r[0]),float(r[1])) for r in csv.reader(open(f"{PR}/panel_split_manifest.csv")) if r[0]!="a_s"]
def env_at(ts): return env[min(int(ts*SR/HOP),nf-1)]
print(f"denominators: held-out words={len(HO)}  all words={len(ALL)}  energy-events={len(PK)}  (noise={noise:.1f})")
def rec_windows(ma,t,c,wins):
    h=0
    for a,b in wins:
        m=(t>=a-HIT_PRE)&(t<=b+HIT_POST)
        if m.any() and (ma[m]>=c).any(): h+=1
    return h
def rec_events(dets,c):
    return sum(1 for p in PK if ((dets>=p-HIT_PRE)&(dets<=p+HIT_POST)).any()) if len(dets) else 0
for ml,mp in MODELS.items():
    m=Model(mp,stride=STRIDE); pr=np.asarray(m.predict_clip(d,step_ms=10),dtype=np.float32)
    ma=sliding_window_view(pr,W).mean(-1); sps=dur/len(ma); t=np.arange(len(ma))*sps
    print(f"\n=== {ml} ===  (max_win={ma.max():.3f})")
    print(f"  {'cut':>4} {'HELD-OUT/19':>12} {'all/58':>9} {'events/120':>11} {'hardFA':>7} {'totaldets':>10}")
    for c in CUTS:
        dets=[]; cd=0
        for i in range(len(ma)):
            if cd>0: cd-=1; continue
            if ma[i]>=c: dets.append(t[i]); cd=LOCK
        dets=np.array(dets)
        rho=rec_windows(ma,t,c,HO); ra=rec_windows(ma,t,c,ALL); re=rec_events(dets,c)
        hf=sum(1 for td in dets if (np.min(np.abs(PK-td)) if len(PK) else 9)>1.5 and env_at(td)<noise*1.5)
        print(f"  {c:>4.2f} {100*rho/len(HO):>9.1f}% {100*ra/len(ALL):>7.1f}% {100*re/len(PK):>9.1f}% {hf:>7} {len(dets):>10}")
print("\nEVAL_PANEL_HELDOUT_DONE")
