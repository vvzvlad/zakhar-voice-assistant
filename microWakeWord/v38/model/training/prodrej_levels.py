#!/usr/bin/env python3
"""Levels of both channels of every run recording (stereo: left = the pipeline channel as STT got it, right =
the other raw channel), to pick channel 1 (the wake-word input, much quieter than channel 0) and to find the
firmware's gain step on it.
Per channel, 30 ms frames: floor = p10 frame RMS (dBFS), voice = p95 frame RMS, peak dBFS.
Channel 1 = the channel whose voice level is lower by >= SEP dB; ambiguous below that.
normalized = the left channel's peak within 0.3 dB of -3 dBFS (the server's mic_normalize target).
Usage: prodrej_levels.py <selected_runs.json> <audio_dir> <out.tsv>"""
import sys, os, json, datetime as dt, numpy as np, soundfile as sf
RUNS, AUD, OUT = sys.argv[1:4]
SEP = 15.0
def stats(x):
    n = len(x) // 480; f = x[:n * 480].astype(np.float64).reshape(n, 480)
    db = 20 * np.log10(np.sqrt((f ** 2).mean(1)) / 32768 + 1e-9)
    return np.percentile(db, 10), np.percentile(db, 95), 20 * np.log10(np.abs(x).max() / 32768 + 1e-9)
rows = []
for r in json.load(open(RUNS)):
    x, sr = sf.read(os.path.join(AUD, f"run_{r['id']}.wav"), dtype="int16"); assert sr == 16000 and x.shape[1] == 2
    (lf, lv, lp), (rf, rv, rp) = stats(x[:, 0]), stats(x[:, 1])
    d = lv - rv
    ch1 = "R" if d >= SEP else "L" if d <= -SEP else "?"
    f1 = rf if ch1 == "R" else lf if ch1 == "L" else float("nan")
    rows.append([r["id"], dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S"), r["result"], r["reason"],
                 round(len(x) / sr, 2), round(lf, 1), round(lv, 1), round(lp, 1), round(rf, 1), round(rv, 1), round(rp, 1),
                 round(d, 1), ch1, round(f1, 1), int(abs(lp + 3.0) < 0.3)])
hdr = ["run", "time", "result", "reason", "dur", "L_floor", "L_voice", "L_peak", "R_floor", "R_voice", "R_peak",
       "L-R_voice", "ch1", "ch1_floor", "L_normalized"]
with open(OUT, "w") as fh:
    fh.write("\t".join(hdr) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
from collections import Counter
print("ch1 side:", dict(Counter(r[12] for r in rows)), " left normalized:", sum(r[14] for r in rows))
for r in rows: print("\t".join(map(str, r)))
