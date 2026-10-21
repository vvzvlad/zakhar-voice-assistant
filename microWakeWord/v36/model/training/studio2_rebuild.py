#!/usr/bin/env python3
"""Rebuild the v35 beep-cued studio prep (recipe/sort_split_prep.py) without its lost STT answers.

The verdicts (kept/rejected), the split and the STT text are read from the v35 run's own output
(logs_recovered/v35_sort_split_prep.out, recovered from the v35 agent's transcript). Everything energy-based
is recomputed from <sort_dir>/segments.json (recipe/segment_clips.py on the same raw clips) and must match
the recorded run: the word segment of every clip and the crop of every training clip. Any mismatch aborts.
Then, exactly as sort_split_prep.py: training crops [max(onset - 0.12 s, beep end), end + 0.20 s] of the RAW
clip with the device's saturating x4 -> <out>/studio2_pos_train/beep_<clip>; <out>/sort_manifest.csv and
<out>/heldout_clips.csv in the original columns.
Usage: studio2_rebuild.py <raw_dir> <sort_dir> <recorded_sort_out> <out_dir>"""
import sys, os, re, csv, json, hashlib, numpy as np, soundfile as sf

SR = 16000
raw_dir, sort_dir, rec_out, out = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
pat = re.compile(r"^(\d{4}\.wav) (kept|rejected) +(heldout|train)? *seg=(\S+) stt=('.*?'|\".*?\")(.*)$")
rec = {}
for line in open(rec_out):
    m = pat.match(line.rstrip("\n"))
    if not m: continue
    clip, verdict, split, seg, stt, rest = m.groups()
    crop = re.search(r"crop=([\d.]+)-([\d.]+)", rest)
    reason = re.sub(r"\s*crop=.*$", "", rest).strip()
    rec[clip] = {"verdict": verdict, "split": split or "", "seg": seg, "stt": eval(stt), "reason": reason,
                 "crop": (float(crop.group(1)), float(crop.group(2))) if crop else None}
assert len(rec) == 40, f"parsed {len(rec)} recorded rows, expected 40"

rows = []
for c in json.load(open(os.path.join(sort_dir, "segments.json"))):
    r0 = rec[c["clip"]]; segs = c["segs"]; s = segs[0] if len(segs) == 1 else None
    seg = f"{s['a']}-{s['b']}" if s else "-"
    assert seg == r0["seg"], f"{c['clip']}: segment {seg} != recorded {r0['seg']}"
    rows.append({"clip": c["clip"], "beep_end_s": c["beep_end_s"], "n_segs": len(segs),
                 "a_s": s["a"] if s else "", "b_s": s["b"] if s else "", "dur_s": s["dur"] if s else "",
                 "peak_mult": s["peak_mult"] if s else "", "peak_db": s["peak_db"] if s else "",
                 "stt": r0["stt"], "verdict": r0["verdict"], "reason": r0["reason"],
                 "split": r0["split"], "crop_a_s": "", "crop_b_s": "", "beep_clamped": ""})

# the recorded split must be the md5 rule of sort_split_prep.py
kept = [r for r in rows if r["verdict"] == "kept"]
order = sorted(kept, key=lambda r: hashlib.md5(r["clip"].encode()).hexdigest())
n_ho = max(1, len(kept) // 3)
for n, r in enumerate(order):
    assert r["split"] == ("heldout" if n < n_ho else "train"), f"{r['clip']}: split differs from the md5 rule"

tr_dir = os.path.join(out, "studio2_pos_train"); os.makedirs(tr_dir, exist_ok=True)
for r in rows:
    if r["split"] != "train": continue
    raw, sr = sf.read(os.path.join(raw_dir, r["clip"]), dtype="int16")
    assert sr == SR and raw.ndim == 1
    ca = max(r["a_s"] - 0.12, r["beep_end_s"]); cb = min(r["b_s"] + 0.20, len(raw) / SR)
    r["crop_a_s"], r["crop_b_s"], r["beep_clamped"] = round(ca, 3), round(cb, 3), int(r["a_s"] - 0.12 < r["beep_end_s"])
    want = rec[r["clip"]]["crop"]
    assert want and abs(want[0] - r["crop_a_s"]) < 1e-6 and abs(want[1] - r["crop_b_s"]) < 1e-6, \
        f"{r['clip']}: crop {r['crop_a_s']}-{r['crop_b_s']} != recorded {want}"
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
nk = len(kept); nh = sum(r["split"] == "heldout" for r in rows); nt = sum(r["split"] == "train" for r in rows)
print(f"STUDIO2_REBUILD: clips={len(rows)} kept={nk} heldout={nh} train={nt}; segments and crops match the v35 run")
