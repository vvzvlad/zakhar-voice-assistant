#!/usr/bin/env python3
"""Process device captures + leakage-safe source-level split.

Positives: section_<seg>_ru.<uuid>__dev.wav (recapture of original real uuid) and
clean_irina_<NNN>__dev.wav (recapture of a synth clip). Source = uuid / clean-id.
Held-out device sources MUST be disjoint from training, AND aligned with the 93
real held-out uuids (so neither eval leaks). Energy-crop each positive to ~2s around
the RMS peak (the word), no STT filtering (device tract makes STT unreliable; all 272
have real signal per operator QC)."""
import os, glob, re, hashlib, shutil
import numpy as np, soundfile as sf

DEV_POS = "device_captures/positive_samples_recorded"
CROP_S = 2.0
SR = 16000

def src_id(fn):
    m = re.search(r"ru\.([0-9a-f-]{8,})__dev", fn)
    if m: return m.group(1)            # uuid for section_*
    m = re.search(r"(clean_[a-z]+_\d+)__dev", fn)
    if m: return m.group(1)            # clean-id
    return fn

def energy_crop(d):
    if len(d) <= int(CROP_S*SR):
        return d
    win = int(0.025*SR); hop = int(0.010*SR)
    e = np.array([np.sqrt(np.mean(d[i:i+win]**2)) for i in range(0, len(d)-win, hop)])
    peak = int(np.argmax(e))*hop
    half = int(CROP_S*SR)//2
    a = max(0, peak-half); b = min(len(d), a+int(CROP_S*SR)); a = max(0, b-int(CROP_S*SR))
    return d[a:b]

# 93 real held-out uuids
HU = set()
for f in glob.glob("real_heldout/*.wav"):
    m = re.search(r"ru\.([0-9a-f-]{8,})", os.path.basename(f))
    if m: HU.add(m.group(1))

dev = sorted(glob.glob(os.path.join(DEV_POS, "*.wav")))
by_src = {}
for f in dev:
    by_src.setdefault(src_id(os.path.basename(f)), []).append(f)
srcs = sorted(by_src)
# held-out device sources: all that are in HU, plus extra (hash) up to ~25% of sources
held = set(s for s in srcs if s in HU)
target = int(round(0.25*len(srcs)))
for s in srcs:
    if len(held) >= target: break
    if s not in held and int(hashlib.md5(s.encode()).hexdigest(), 16) % 4 == 0:
        held.add(s)

os.makedirs("v8/dev_train_pos", exist_ok=True)
os.makedirs("v8/dev_heldout_pos", exist_ok=True)
n_tr=n_ho=0
for s, files in by_src.items():
    dst = "v8/dev_heldout_pos" if s in held else "v8/dev_train_pos"
    for f in files:
        d, sr = sf.read(f, dtype="int16")
        if d.ndim>1: d=d[:,0]
        if sr!=SR: continue
        d = energy_crop(d.astype(np.float32)/32768.0)
        out=os.path.join(dst, os.path.basename(f))
        sf.write(out, (np.clip(d,-1,1)*32767).astype(np.int16), SR)
        if s in held: n_ho+=1
        else: n_tr+=1

# real_train positives = real_train minus any uuid in held-out device sources
os.makedirs("v8/real_train_clean", exist_ok=True)
kept_real=0
for f in glob.glob("real_train/*.wav"):
    m=re.search(r"ru\.([0-9a-f-]{8,})", os.path.basename(f))
    uuid=m.group(1) if m else None
    if uuid in held:    # its device-recapture is held out -> keep dry original out too
        continue
    shutil.copy(f, "v8/real_train_clean/"+os.path.basename(f)); kept_real+=1

print(f"device sources: {len(srcs)} (held-out {len(held)}, incl {len(HU&set(srcs))} aligned w/ 93-real-heldout)")
print(f"device positives: train {n_tr}, held-out {n_ho}")
print(f"real_train_clean (dry, minus held-out sources): {kept_real}")
