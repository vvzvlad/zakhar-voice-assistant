#!/usr/bin/env python3
"""v35 sort + split + train-clip prep for the 40 beep-cued studio clips (raw ch1, gain 1).

Inputs: <sort_dir>/segments.json (segment_clips.py) and <sort_dir>/stt_out/<clip>.json (Groq
whisper-large-v3, language ru, temperature 0, no prompt, on the clip after the beep, peak-normalised).

A clip is KEPT iff
  energy: exactly one segment after the beep, and it is word-like (studio_prep.py v34 filter:
          0.18 <= dur <= 0.90 s, peak frame RMS >= 3x noise) and touches neither the beep nor the clip end;
  STT:    the transcript holds exactly one «захар»-like token and no other word (fuzzy match as
          microWakeWord/samples/_asr_filter_v03.py: targets захар/сахар/захаар/захор, ratio >= 0.66).
Split: kept clips sorted by md5(file name), the first len//3 held out (the v34 rule, keyed on the take).
Train prep, exactly as studio_prep.py (v34): crop [onset - 0.12 s, end + 0.20 s] of the RAW clip, then the
device's saturating x4 (clamp(4*x)). The only addition: the crop never starts before the beep's end, so the
beep tail stays out of the positive.
Writes <out>/sort_manifest.csv, <out>/heldout_clips.csv and <out>/studio2_pos_train/*.wav.
Usage: sort_split_prep.py <raw_dir> <sort_dir> <out_dir>"""
import sys, os, re, csv, json, hashlib, difflib, numpy as np, soundfile as sf

SR = 16000
TARGETS = ("захар", "сахар", "захаар", "захор")
MATCH_RATIO = 0.66
raw_dir, sort_dir, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])


def tokens(text):
    return re.sub(r"[^а-яё ]+", " ", text.lower()).split()


def is_zahar(w):
    return max(difflib.SequenceMatcher(None, w, t).ratio() for t in TARGETS) >= MATCH_RATIO


rows = []
for c in json.load(open(os.path.join(sort_dir, "segments.json"))):
    stt = json.load(open(os.path.join(sort_dir, "stt_out", c["clip"].replace(".wav", ".json"))))
    text = stt["text"].strip()
    toks = tokens(text)
    nz = sum(is_zahar(w) for w in toks); other = [w for w in toks if not is_zahar(w)]
    segs = c["segs"]; s = segs[0] if len(segs) == 1 else None
    reasons = []
    if not segs:
        reasons.append("energy: no segment after the beep")
    elif len(segs) > 1:
        reasons.append(f"energy: {len(segs)} segments after the beep")
    else:
        if not s["wordlike"]: reasons.append(f"energy: segment not word-like (dur {s['dur']} s, x{s['peak_mult']})")
        if s["touches_end"]: reasons.append("energy: segment runs into the clip end")
        if s["touches_beep"]: reasons.append("energy: segment starts on the beep")
    if c["beep_overlap"]: reasons.append("energy: word overlaps the beep")
    if nz == 0: reasons.append("STT: no «захар»")
    elif nz > 1: reasons.append(f"STT: {nz} «захар»")
    if other: reasons.append("STT: other words " + " ".join(other))
    rows.append({"clip": c["clip"], "beep_end_s": c["beep_end_s"], "n_segs": len(segs),
                 "a_s": s["a"] if s else "", "b_s": s["b"] if s else "", "dur_s": s["dur"] if s else "",
                 "peak_mult": s["peak_mult"] if s else "", "peak_db": s["peak_db"] if s else "",
                 "stt": text, "verdict": "rejected" if reasons else "kept", "reason": "; ".join(reasons),
                 "split": "", "crop_a_s": "", "crop_b_s": "", "beep_clamped": ""})

kept = [r for r in rows if r["verdict"] == "kept"]
order = sorted(kept, key=lambda r: hashlib.md5(r["clip"].encode()).hexdigest())
n_ho = max(1, len(kept) // 3)
for n, r in enumerate(order):
    r["split"] = "heldout" if n < n_ho else "train"

tr_dir = os.path.join(out, "studio2_pos_train"); os.makedirs(tr_dir, exist_ok=True)
for r in rows:
    if r["split"] != "train": continue
    raw, sr = sf.read(os.path.join(raw_dir, r["clip"]), dtype="int16")
    assert sr == SR and raw.ndim == 1
    ca = max(r["a_s"] - 0.12, r["beep_end_s"]); cb = min(r["b_s"] + 0.20, len(raw) / SR)
    r["crop_a_s"], r["crop_b_s"], r["beep_clamped"] = round(ca, 3), round(cb, 3), int(r["a_s"] - 0.12 < r["beep_end_s"])
    i0, i1 = int(ca * SR), int(cb * SR)
    w = np.clip(raw[i0:i1].astype(np.int32) * 4, -32768, 32767).astype(np.int16)
    sf.write(os.path.join(tr_dir, f"beep_{r['clip']}"), w, SR, subtype="PCM_16")

cols = list(rows[0].keys())
with open(os.path.join(out, "sort_manifest.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
with open(os.path.join(out, "heldout_clips.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["path", "beep_end_s", "a_s", "b_s"])
    for r in rows:
        if r["split"] == "heldout":
            wr.writerow([os.path.join(os.path.abspath(raw_dir), r["clip"]), r["beep_end_s"], r["a_s"], r["b_s"]])

for r in rows:
    print(f"{r['clip']} {r['verdict']:8s} {r['split']:7s} seg={r['a_s']}-{r['b_s']} stt={r['stt']!r} {r['reason']}"
          f"{' crop=' + str(r['crop_a_s']) + '-' + str(r['crop_b_s']) if r['crop_a_s'] != '' else ''}"
          f"{' (start clamped to beep end)' if r['beep_clamped'] == 1 else ''}")
nk = len(kept); nh = sum(r["split"] == "heldout" for r in rows); nt = sum(r["split"] == "train" for r in rows)
print(f"SORT: clips={len(rows)} kept={nk} rejected={len(rows) - nk} heldout={nh} train={nt} -> {tr_dir}")
print("SORT_DONE")
