#!/usr/bin/env python3
"""Fused «захар + next word» positives and fused confusable negatives for v36, from the forced alignment.

Voice split: every voice of the two Fish pools (coart «захар + команда», confus «сахар…» phrases), ordered by
md5(voice_id); the first third is held out of training entirely (both blocks), the rest trains.
Training crops: [0, target end + d] of the 16 kHz clip, d uniform in 100-300 ms, drawn per file from
md5(file name) so the crop does not depend on run order; capped at the clip's end. The crop keeps the
onset of the following word, the case the speaker misses («захаркоторый час»).
Eval lists (whole, uncropped clips of the held-out voices):
  fused_ho.txt          every coart clip;  fused_ho_nospace.txt  the texts written fused («захарвключи свет»);
  isolated_ho.txt       the same voices' isolated «захар» (v29/pos_short_fish_raw_keep_trainsafe_2608,
                        one clip per voice, matched on the 8-hex voice prefix in the file name);
  confus_ho.txt         the confusable phrases with a following word.
Usage: prep_fused.py <coart_align.tsv> <confus_align.tsv> <coart_dir> <confus_dir> <isolated_dir> <W>"""
import sys, os, csv, glob, hashlib, numpy as np, soundfile as sf

CA, FA, CDIR, FDIR, IDIR, W = sys.argv[1:7]
SR = 16000
read = lambda p: list(csv.DictReader(open(p), delimiter="\t"))
coart, confus = read(CA), read(FA)
h = lambda s: hashlib.md5(s.encode()).hexdigest()
voices = sorted({r["voice_id"] for r in coart} | {r["voice_id"] for r in confus}, key=h)
n_ho = len(voices) // 3
held = set(voices[:n_ho])
print(f"voices: {len(voices)} total, {n_ho} held out, {len(voices) - n_ho} train")

def delta(f):  # uniform 0.100-0.300 s from md5(file)
    return 0.1 + 0.2 * int(h(f)[:8], 16) / 0xFFFFFFFF

def crop_set(rows, src, out, tag):
    os.makedirs(out, exist_ok=True)
    assert not glob.glob(os.path.join(out, "*.wav")), f"REFUSE: {out} not empty"
    log = []; bad = 0
    for r in rows:
        if r["voice_id"] in held: continue
        te, dur = float(r["t_end"]), float(r["dur"])
        ns = float(r["next_start"]) if r["next_start"] else dur
        if not (0.15 < te <= ns <= dur): bad += 1; print(f"  {tag} odd alignment, kept as is: {r['file']} {te} {ns} {dur}")
        x, sr = sf.read(os.path.join(src, r["file"]), dtype="int16"); assert sr == SR and x.ndim == 1
        d = delta(r["file"]); cb = min(len(x), int(round((te + d) * SR)))
        sf.write(os.path.join(out, r["file"]), x[:cb], SR, subtype="PCM_16")
        log.append([r["file"], r["voice_id"], r["text"], r["target"], te, r["next_start"], round(d, 3), round(cb / SR, 3), dur])
    with open(out.rstrip("/") + ".crops.tsv", "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["file", "voice_id", "text", "target", "t_end", "next_start", "delta", "crop_end", "dur"]); wr.writerows(log)
    cut = np.array([l[7] for l in log]); tail = np.array([l[7] - l[4] for l in log])
    print(f"{tag}: {len(log)} train crops -> {out}; crop end s p10/p50/p90 {np.percentile(cut, 10):.2f}/"
          f"{np.median(cut):.2f}/{np.percentile(cut, 90):.2f}; kept after the target p50 {np.median(tail):.3f} s; "
          f"capped at clip end {sum(1 for l in log if l[7] >= l[8] - 1e-3)}; odd {bad}")

crop_set(coart, CDIR, f"{W}/prep/coart_fused_train", "coart")
crop_set(confus, FDIR, f"{W}/prep/confus_fused_train", "confus")

os.makedirs(f"{W}/eval_inputs", exist_ok=True)
def write(name, paths):
    with open(f"{W}/eval_inputs/{name}", "w") as fh: fh.write("".join(p + "\n" for p in paths))
    print(f"{name}: {len(paths)} clips")
ho_c = [r for r in coart if r["voice_id"] in held]
write("fused_ho.txt", [os.path.join(CDIR, r["file"]) for r in ho_c])
write("fused_ho_nospace.txt", [os.path.join(CDIR, r["file"]) for r in ho_c if not r["text"].startswith("захар ")])
v8 = {v[:8] for v in held}
iso = sorted(p for p in glob.glob(os.path.join(IDIR, "*.wav")) if os.path.basename(p).split("_")[1] in v8)
write("isolated_ho.txt", iso)
write("confus_ho.txt", [os.path.join(FDIR, r["file"]) for r in confus if r["voice_id"] in held])
print(f"held-out voices with an isolated clip: {len({os.path.basename(p).split('_')[1] for p in iso})}")
with open(f"{W}/prep/fused_voice_split.tsv", "w") as fh:
    fh.write("voice_id\tsplit\n" + "".join(f"{v}\t{'heldout' if v in held else 'train'}\n" for v in voices))
