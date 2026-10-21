#!/usr/bin/env python3
"""Beep-free copies of the beep-cued clips for the held-out check, prepared like the training crops (which
start after the beep): the clip's first beep_end seconds (segment_clips.py) are replaced by the clip's own
last beep_end seconds, i.e. room noise recorded after the word, so the timing of the word and the noise
floor stay as recorded and only the speaker's chime is gone. Raw ch1 gain 1, as the inputs.
Writes <out>/<clip>.wav and <out>/heldout_clips.csv (paths to the copies, same columns as the original).
Usage: make_nobeep.py <raw_dir> <sort_manifest.csv> <heldout_clips.csv> <out_dir>"""
import sys, os, csv, numpy as np, soundfile as sf

raw_dir, man, ho, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for r in csv.DictReader(open(man)):
    a, sr = sf.read(os.path.join(raw_dir, r["clip"]), dtype="int16"); assert sr == 16000 and a.ndim == 1
    n = int(round(float(r["beep_end_s"]) * sr))
    b = a.copy(); b[:n] = a[-n:]
    sf.write(os.path.join(out, r["clip"]), b, sr, subtype="PCM_16")
rows = list(csv.DictReader(open(ho)))
with open(os.path.join(out, "heldout_clips.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "beep_end_s", "a_s", "b_s"])
    for r in rows: w.writerow([os.path.join(os.path.abspath(out), os.path.basename(r["path"])), r["beep_end_s"], r["a_s"], r["b_s"]])
print(f"NOBEEP: wrote beep-free copies to {out}")
