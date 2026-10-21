#!/usr/bin/env python3
"""Single-voice take-disjoint prep for operator ch1 short-«захар» takes.
- crop each clip to its detected word region (+margins), apply device gain x4 (int16, serve level)
  -> real_pos_train/*.wav  (TRAIN takes only; feature builder input)
- keep FULL ORIGINAL clips for the honest eval (device_faithful harness applies x4 itself)
  -> pos_real_heldout.txt (HELDOUT takes), pos_real_train_full.txt (TRAIN takes, for optional paired check)
Deterministic: takes ordered by md5(basename); first N_HELDOUT -> heldout. No RNG.
Single voice => speaker-disjoint IMPOSSIBLE; this is TAKE(utterance)-disjoint. Honest for his-voice recall,
optimistic vs a truly-unseen speaker (logged as a caveat). ml-researcher #1/#3.
Usage: prep_single_voice.py <incoming_dir> <OUT_ROOT> [n_heldout=18] [gain=4]
"""
import sys,os,glob,hashlib,numpy as np,soundfile as sf
inc=os.path.abspath(sys.argv[1]); out=os.path.abspath(sys.argv[2])
NHO=int(sys.argv[3]) if len(sys.argv)>3 else 18
G=int(sys.argv[4]) if len(sys.argv)>4 else 4
SR=16000
files=sorted(glob.glob(inc+"/*.wav"))
assert files, "no wavs"
def env(x,fr=0.025,hop=0.010):
    n=int(fr*SR); h=int(hop*SR); o=[]
    for i in range(0,max(len(x)-n,0),h): o.append(np.sqrt(np.mean(x[i:i+n].astype(np.float64)**2)))
    return np.array(o),h
def word_bounds(x):
    e,h=env(x)
    if len(e)<3 or e.max()<=0: return 0,len(x)
    # smooth
    k=np.ones(3)/3; es=np.convolve(e,k,mode="same")
    thr=max(0.20*es.max(), 2.0)   # 2.0 int16-RMS floor to ignore dead silence
    idx=np.where(es>=thr)[0]
    if len(idx)==0: return 0,len(x)
    a=idx[0]*h; b=(idx[-1]+1)*h
    a=max(0,a-int(0.12*SR)); b=min(len(x),b+int(0.20*SR))
    if b-a < int(0.5*SR):  # min-length guard: center-expand
        c=(a+b)//2; a=max(0,c-int(0.35*SR)); b=min(len(x),c+int(0.35*SR))
    return a,b
def gain_i16(x,g):
    if g==1: return x
    return np.clip(x.astype(np.int32)*g,-32768,32767).astype(np.int16)
order=sorted(files,key=lambda p:hashlib.md5(os.path.basename(p).encode()).hexdigest())
heldout=set(order[:NHO]); train=set(order[NHO:])
assert not (heldout&train)
tr_dir=os.path.join(out,"real_pos_train"); os.makedirs(tr_dir,exist_ok=True)
rows=[]
for p in files:
    x,sr=sf.read(p,dtype="int16")
    if x.ndim>1: x=x[:,0]
    assert sr==SR, f"{p} sr={sr}"
    a,b=word_bounds(x)
    split="heldout" if p in heldout else "train"
    rows.append((os.path.basename(p),split,len(x)/SR,round(a/SR,2),round(b/SR,2),int(np.max(np.abs(x)))))
    if split=="train":
        w=gain_i16(x[a:b],G)
        sf.write(os.path.join(tr_dir,"op__"+os.path.basename(p)),w,SR,subtype="PCM_16")
with open(os.path.join(out,"pos_real_heldout.txt"),"w") as f:
    for p in sorted(heldout): f.write(p+"\n")
with open(os.path.join(out,"pos_real_train_full.txt"),"w") as f:
    for p in sorted(train): f.write(p+"\n")
with open(os.path.join(out,"split_manifest.csv"),"w") as f:
    f.write("basename,split,dur_s,word_a_s,word_b_s,peak_i16\n")
    for r in rows: f.write(",".join(str(x) for x in r)+"\n")
ntr=sum(1 for r in rows if r[1]=="train"); nho=sum(1 for r in rows if r[1]=="heldout")
print(f"PREP: train={ntr} heldout={nho} gain=x{G}  -> {tr_dir}")
print(f"PREP: wrote pos_real_heldout.txt({nho}) pos_real_train_full.txt({ntr}) split_manifest.csv")
wl=[r[4]-r[3] for r in rows]; print(f"PREP: word-window s: min={min(wl):.2f} med={sorted(wl)[len(wl)//2]:.2f} max={max(wl):.2f}")
