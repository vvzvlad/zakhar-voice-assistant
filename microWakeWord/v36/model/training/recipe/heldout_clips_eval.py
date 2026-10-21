#!/usr/bin/env python3
"""Recall on the held-out beep-cued studio clips, timeline.py method per clip: the whole raw 3 s clip
(beep included, as recorded) gets the device's saturating x4, streams through a fresh Model, win5 mean,
25-step (750 ms) lockout, cut 0.80. The clip counts as caught when a detection lands within
[word onset - 0.3 s, word end + 1.0 s] (the studio_eval.py window). Detections outside that window
(e.g. on the beep) are counted separately and do not count as hits.
Prints one HOJSON line per model. Usage: heldout_clips_eval.py <heldout_clips.csv> <label:tflite> ..."""
import sys, csv, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view

STEP = 0.03; W = 5; LOCK = 25; CUT = 0.80
rows = list(csv.DictReader(open(sys.argv[1])))
for spec in sys.argv[2:]:
    lab, tfl = spec.split(":", 1)
    per = []
    for r in rows:
        raw, sr = sf.read(r["path"], dtype="int16"); assert sr == 16000 and raw.ndim == 1
        x4 = np.clip(raw.astype(np.int32) * 4, -32768, 32767).astype(np.int16)
        pr = np.asarray(Model(tfl, stride=3).predict_clip(x4, step_ms=10), dtype=np.float32)
        ma = sliding_window_view(pr, W).mean(-1)
        dets, cd = [], 0
        for k, val in enumerate(ma):
            cd = max(cd - 1, 0)
            if cd == 0 and val >= CUT: dets.append(round((k + W - 1) * STEP, 2)); cd = LOCK
        a, b = float(r["a_s"]), float(r["b_s"])
        inwin = [t for t in dets if a - 0.3 <= t <= b + 1.0]
        k0 = max(0, int((a - 0.3) / STEP) - W + 1); k1 = int((b + 1.0) / STEP)
        per.append({"clip": r["path"].rsplit("/", 1)[-1], "hit": int(bool(inwin)), "dets": dets,
                    "outside": len(dets) - len(inwin), "maxwin_word": round(float(ma[k0:k1].max()), 3)})
    print("HOJSON " + json.dumps({"label": lab, "cut": CUT, "n": len(per), "hits": sum(p["hit"] for p in per),
                                  "outside": sum(p["outside"] for p in per), "rows": per}))
