#!/usr/bin/env python3
"""Training clips of the owner's read session (channel 1 already x4: written as recorded, no gain), from the
forced alignment (align_owner.py), by the v36 Fish crop rule (v36/prep_fused.py):
  owner_pos_train/  the 40 «захар + команда» clips cropped to [0, «захар» end + d];
  owner_neg_train/  the 30 negatives whole, plus the 22 confusable ones cropped to [0, confusable end + d]
                    (named neg_NN_crop.wav);
d uniform in 100-300 ms from md5(file name), capped at the clip's end.
Cross-check of the alignment: for the phrases the speaker caught live, the detection time in clip
coordinates (detect - start_s + 1 s pad) minus the aligned «захар» end.
Usage: prep_owner.py <LABELS.tsv> <owner_align.tsv> <pos_dir> <neg_dir> <out_root>"""
import sys, os, csv, glob, hashlib, numpy as np, soundfile as sf

LAB, ALI, PDIR, NDIR, OUT = sys.argv[1:6]
SR = 16000
h = lambda s: hashlib.md5(s.encode()).hexdigest()
delta = lambda f: 0.1 + 0.2 * int(h(f)[:8], 16) / 0xFFFFFFFF   # as v36/prep_fused.py
labels = {r["clip"]: r for r in csv.DictReader(open(LAB), delimiter="\t")}
ali = {r["file"]: r for r in csv.DictReader(open(ALI), delimiter="\t")}

def out_dir(name):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    assert not glob.glob(os.path.join(d, "*.wav")), f"REFUSE: {d} not empty"
    return d

def read(d, f):
    x, sr = sf.read(os.path.join(d, f), dtype="int16"); assert sr == SR and x.ndim == 1, f; return x

def crop(d, f, dst, name, log):
    x = read(d, f); r = ali[f]; te = float(r["t_end"]); dd = delta(f)
    cb = min(len(x), int(round((te + dd) * SR)))
    sf.write(os.path.join(dst, name), x[:cb], SR, subtype="PCM_16")
    log.append([name, r["text"], r["target"], te, r["next_start"], round(dd, 3), round(cb / SR, 3), round(len(x) / SR, 3)])

def write_log(dst, log):
    with open(dst.rstrip("/") + ".crops.tsv", "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["file", "text", "target", "t_end", "next_start", "delta", "crop_end", "dur"]); wr.writerows(log)

pos = sorted(f for f in labels if labels[f]["kind"] == "pos"); neg = sorted(f for f in labels if labels[f]["kind"] == "neg")
assert len(pos) == 40 and len(neg) == 30 and all(f in ali for f in pos)
P = out_dir("owner_pos_train"); plog = []
for f in pos: crop(PDIR, f, P, f, plog)
write_log(P, plog)
N = out_dir("owner_neg_train"); nlog = []
for f in neg:
    sf.write(os.path.join(N, f), read(NDIR, f), SR, subtype="PCM_16")
    if f in ali: crop(NDIR, f, N, f.replace(".wav", "_crop.wav"), nlog)
write_log(N, nlog)
for tag, log in (("pos crops", plog), ("neg crops", nlog)):
    k = np.array([l[6] - l[3] for l in log])
    print(f"{tag}: {len(log)}; kept after the target s min/p50/max {k.min():.3f}/{np.median(k):.3f}/{k.max():.3f}; "
          f"capped at clip end {sum(1 for l in log if l[6] >= l[7] - 1e-3)}")
print(f"owner_pos_train: {len(glob.glob(P + '/*.wav'))} wav; owner_neg_train: {len(glob.glob(N + '/*.wav'))} wav")
lag = [(f, float(labels[f]["device_v36_detect_s"]) - float(labels[f]["start_s"]) + 1.0 - float(ali[f]["t_end"]))
       for f in pos if labels[f]["device_v36_detect_s"]]
print(f"live detections after the aligned «захар» end, s ({len(lag)} phrases): "
      + ", ".join(f"{f[4:6]} {v:+.2f}" for f, v in lag))
