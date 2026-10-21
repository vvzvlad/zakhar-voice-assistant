#!/usr/bin/env python3
"""Prep panel «захар» recording for an OPTIONAL v33 finetune (v32-style). NON-DESTRUCTIVE: segments the
one long panel file into word crops, quality-filters (drop movement/breath by duration+shape), does a
take-disjoint split (hold out ~1/3 of words so recall isn't train=eval leaked), saves TRAIN crops at x1
(panel firmware already DC-blocks -> samples ARE model input). Does NOT build features or train.
Reports label-quality numbers so we can eyeball before committing. ml-researcher #6 (validate labels at build).
Usage: panel_prep.py <OUT_ROOT>
"""
import sys,os,numpy as np,soundfile as sf,hashlib
sys.path.insert(0,"/home/claude/zakhar-mww/micro-wake-word")
SR=16000
OUT=os.path.abspath(sys.argv[1]) if len(sys.argv)>1 else "/tmp/panel_prep"
P="/home/claude/zakhar-mww/device_captures/panel_round_zakhar_20261016/fc012cd1e5ec_20261016_171350.wav"
d,sr=sf.read(P,dtype="int16"); d=(d[:,0] if d.ndim>1 else d).astype(np.int16); assert sr==SR
x=d.astype(np.float64); dur=len(d)/SR
FR=320;HOP=160; nf=1+(len(x)-FR)//HOP
fe=np.array([np.sqrt(np.mean(x[i*HOP:i*HOP+FR]**2)) for i in range(nf)])
noise=np.median(fe)
# --- energy-VAD segments (same finder as recall denom, mult=4) ---
def segs(mult,merge_gap_s=0.30,min_dur_s=0.12):
    thr=max(noise*mult,8.0); sp=fe>thr; out=[]; i=0
    while i<nf:
        if sp[i]:
            j=i
            while j<nf and sp[j]: j+=1
            out.append([i*HOP/SR,(j-1)*HOP/SR+FR/SR]); i=j
        else: i+=1
    merged=[]
    for s in out:
        if merged and s[0]-merged[-1][1]<=merge_gap_s: merged[-1][1]=s[1]
        else: merged.append(s)
    return [s for s in merged if s[1]-s[0]>=min_dur_s]
S=segs(4)
# --- quality filter: word «захар» ~0.25-0.65s, two-burst; drop clicks(<0.18s) & long smears(>0.9s = merged/movement) ---
def peak_mult(a,b):
    i0=int(a*SR/HOP); i1=max(i0+1,int(b*SR/HOP)); return fe[i0:i1].max()/noise
kept=[]; dropped=[]
for a,b in S:
    dw=b-a; pm=peak_mult(a,b)
    ok = (0.18<=dw<=0.90) and (pm>=3.0)
    (kept if ok else dropped).append((round(a,2),round(b,2),round(dw,2),round(pm,1)))
# --- take-disjoint split: deterministic by md5(onset) -> hold out ~1/3 ---
order=sorted(range(len(kept)),key=lambda i:hashlib.md5(f"{kept[i][0]}".encode()).hexdigest())
n_ho=max(1,len(kept)//3)
ho_idx=set(order[:n_ho]); tr_idx=set(order[n_ho:])
os.makedirs(os.path.join(OUT,"panel_pos_train"),exist_ok=True)
rows=[]
for k,(a,b,dw,pm) in enumerate(kept):
    split="heldout" if k in ho_idx else "train"
    rows.append((a,b,dw,pm,split))
    if split=="train":
        i0=max(0,int((a-0.12)*SR)); i1=min(len(d),int((b+0.20)*SR))
        sf.write(os.path.join(OUT,"panel_pos_train",f"panel_{k:03d}.wav"),d[i0:i1],SR,subtype="PCM_16")  # x1
# held-out word windows (for recall eval on full-file streaming run)
with open(os.path.join(OUT,"panel_heldout_windows.csv"),"w") as f:
    f.write("a_s,b_s\n")
    for a,b,dw,pm,sp in rows:
        if sp=="heldout": f.write(f"{a},{b}\n")
with open(os.path.join(OUT,"panel_split_manifest.csv"),"w") as f:
    f.write("a_s,b_s,dur_s,peak_mult,split\n")
    for r in rows: f.write(",".join(str(v) for v in r)+"\n")
ntr=sum(1 for r in rows if r[4]=="train"); nho=sum(1 for r in rows if r[4]=="heldout")
dws=[r[2] for r in rows]
print(f"PANEL_PREP: VAD segs(mult4)={len(S)}  kept(word-like)={len(kept)}  dropped={len(dropped)}")
print(f"  kept dur s: min={min(dws):.2f} med={sorted(dws)[len(dws)//2]:.2f} max={max(dws):.2f}")
print(f"  split: train={ntr}  heldout={nho}  -> {OUT}/panel_pos_train (x1 crops)")
print(f"  dropped (a,b,dur,peakx) sample: {dropped[:12]}")
print(f"  NOTE positives are AUTO-segmented from an unmarked file -> label noise possible (movement/breath).")
print(f"  NOTE single voice/session/device/day -> HIS-voice SAME-condition, optimistic vs unseen (documented).")
print("PANEL_PREP_DONE")

