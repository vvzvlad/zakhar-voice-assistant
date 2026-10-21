#!/usr/bin/env python3
"""Studio take recall, timeline.py method: device-faithful x4, win5 mean, lockout 25 steps, cut 0.80;
a segment is caught by a detection within [onset-0.3 s, end+1.0 s]. Counts all 53 energy segments and
the held-out words of studio_split_manifest.csv. Prints one STUDIOJSON line per model.
Usage: studio_eval.py <take.wav> <manifest.csv> <label:tflite> ..."""
import sys, csv, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view

STEP = 0.03; W = 5; LOCK = 25; CUT = 0.80
take, man = sys.argv[1], sys.argv[2]
raw, sr = sf.read(take, dtype="int16")
x4 = np.clip(raw.astype(np.int32) * 4, -32768, 32767).astype(np.int16)
rows = list(csv.DictReader(open(man)))
segs = [(float(r["a_s"]), float(r["b_s"]), r["split"]) for r in rows]
for spec in sys.argv[3:]:
    lab, tfl = spec.split(":", 1)
    pr = np.asarray(Model(tfl, stride=3).predict_clip(x4, step_ms=10), dtype=np.float32)
    ma = sliding_window_view(pr, W).mean(-1)
    dets, cd = [], 0
    for k, val in enumerate(ma):
        cd = max(cd - 1, 0)
        if cd == 0 and val >= CUT: dets.append((k + W - 1) * STEP); cd = LOCK
    hit = [any(a - 0.3 <= t <= b + 1.0 for t in dets) for a, b, _ in segs]
    by = lambda s: [h for h, (_, _, sp) in zip(hit, segs) if sp == s]
    res = {"label": lab, "cut": CUT, "dets": len(dets),
           "all": [sum(hit), len(hit)],
           "heldout": [sum(by("heldout")), len(by("heldout"))],
           "train": [sum(by("train")), len(by("train"))],
           "dropped": [sum(by("dropped")), len(by("dropped"))],
           "hit": [int(h) for h in hit]}
    print("STUDIOJSON " + json.dumps(res))
