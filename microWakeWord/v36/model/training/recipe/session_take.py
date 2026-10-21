#!/usr/bin/env python3
"""Rebuild the beep-cued session as one continuous take for studio_eval.py (the timeline method of the
v34 studio check): the 40 raw clips (ch1, gain 1) back to back in recording order, so the frontend and
the streaming state run settled through the session as on the device instead of starting fresh inside
each clip's beep. Writes <out>/session_take.wav and <out>/session_manifest.csv in the
studio_split_manifest.csv format (seg,a_s,b_s,dur_s,peak_mult,split): one row per kept clip with its
word segment shifted to the take's timeline and its split (heldout/train); rejected clips get no row.
Usage: session_take.py <raw_dir> <sort_manifest.csv> <out_dir>"""
import sys, os, csv, glob, numpy as np, soundfile as sf

raw_dir, man, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = {r["clip"]: r for r in csv.DictReader(open(man))}
parts, off, segs = [], 0, []
for k, p in enumerate(sorted(glob.glob(os.path.join(raw_dir, "*.wav")))):
    a, sr = sf.read(p, dtype="int16"); assert sr == 16000 and a.ndim == 1
    r = rows[os.path.basename(p)]
    if r["verdict"] == "kept":
        t0 = off / sr
        segs.append((k, round(t0 + float(r["a_s"]), 3), round(t0 + float(r["b_s"]), 3), r["dur_s"], r["peak_mult"], r["split"]))
    parts.append(a); off += len(a)
sf.write(os.path.join(out, "session_take.wav"), np.concatenate(parts), 16000, subtype="PCM_16")
with open(os.path.join(out, "session_manifest.csv"), "w") as fh:
    fh.write("seg,a_s,b_s,dur_s,peak_mult,split\n")
    for s in segs: fh.write(",".join(str(x) for x in s) + "\n")
print(f"SESSION_TAKE: {len(parts)} clips, {off / 16000:.1f} s, {len(segs)} kept words "
      f"({sum(s[5] == 'heldout' for s in segs)} held out)")
