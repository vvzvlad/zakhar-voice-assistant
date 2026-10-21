#!/usr/bin/env python3
"""Cross-check of the MMS_FA word ends against an independent recognizer: Vosk (small Russian model, free
recognition with word times). For every clip whose first recognized word is «захар»-like (difflib ratio
>= 0.66), compare Vosk's end of that word with the aligner's. Prints the distribution of the difference.
Usage: check_align_vosk.py <align.tsv> <audio_dir>"""
import sys, os, csv, json, difflib, wave, numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
m = Model("/home/claude/zakhar-mww/v8/vosk-model-small-ru-0.22")
d_all, d_fused, n_rec = [], [], 0
for r in rows:
    wf = wave.open(os.path.join(sys.argv[2], r["file"]), "rb")
    rec = KaldiRecognizer(m, wf.getframerate()); rec.SetWords(True)
    while True:
        b = wf.readframes(4000)
        if not b: break
        rec.AcceptWaveform(b)
    res = json.loads(rec.FinalResult()).get("result", [])
    if not res: continue
    w = res[0]
    if difflib.SequenceMatcher(None, w["word"], "захар").ratio() < 0.66: continue
    n_rec += 1
    d = float(r["t_end"]) - w["end"]
    (d_fused if not r["text"].startswith("захар ") else d_all).append(d)
for name, d in (("spaced texts", d_all), ("fused texts", d_fused)):
    if d:
        d = np.array(d)
        print(f"{name}: n={len(d)} aligner end - vosk end, s: p10 {np.percentile(d, 10):+.3f} p50 {np.median(d):+.3f} "
              f"p90 {np.percentile(d, 90):+.3f}; |diff| <= 0.10 s in {np.mean(np.abs(d) <= 0.10) * 100:.0f}%")
print(f"clips: {len(rows)}, Vosk's first word «захар»-like in {n_rec}")
