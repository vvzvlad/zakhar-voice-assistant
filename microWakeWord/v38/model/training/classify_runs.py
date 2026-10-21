#!/usr/bin/env python3
"""Channel 1, gain and class of every rejected/empty wake of the speaker from the run log.
  channel 1: the side picked by prodrej_levels.py (the quieter by >= 15 dB on voice); ambiguous runs, and runs
             whose picked channel 1 has a floor above the other channel's by > 3 dB, are left out;
  gain:      runs up to LAST_GAIN1_RUN came from the firmware that streamed channel 1 at gain 1 (the floor and
             level step in prep/levels.tsv lies between runs 3758 and 3788, around commit 4b76c1d) and get the
             device's saturating x4; later runs are already x4 and stay as they are;
  class:     GigaAM v3 (onnx-asr gigaam-v3-rnnt, int8, as src/plugins/stt/gigaam.py calls it) on the whole
             recording, channel 1 (after the gain) and channel 0 separately; a word starting «захар» in either
             text -> pos, otherwise -> neg.
Writes <out_dir>/runs_ch1/run_<id>.wav (channel 1, x4 domain, whole recording) and <out_dir>/classes.tsv.
Usage: classify_runs.py <levels.tsv> <audio_dir> <gigaam_dir> <out_dir>"""
import sys, os, csv, numpy as np, soundfile as sf, onnx_asr
LEV, AUD, MOD, OUT = sys.argv[1:5]
LAST_GAIN1_RUN = 3758
os.makedirs(os.path.join(OUT, "runs_ch1"), exist_ok=True)
asr = onnx_asr.load_model("gigaam-v3-rnnt", MOD, quantization="int8")
text = lambda x: asr.recognize(x.astype(np.float32) / 32768.0, sample_rate=16000).strip()
has = lambda t: any(w.startswith("захар") for w in t.lower().split())
rows, skipped = [], []
for r in csv.DictReader(open(LEV), delimiter="\t"):
    side = r["ch1"]
    other_floor = float(r["L_floor" if side == "R" else "R_floor"]) if side in "LR" else None
    if side == "?" or float(r["ch1_floor"]) > other_floor + 3:
        skipped.append((r["run"], r["time"], r["L-R_voice"])); continue
    x, sr = sf.read(os.path.join(AUD, f"run_{r['run']}.wav"), dtype="int16")
    c1, c0 = (x[:, 1], x[:, 0]) if side == "R" else (x[:, 0], x[:, 1])
    g = 4 if int(r["run"]) <= LAST_GAIN1_RUN else 1
    if g == 4: c1 = np.clip(c1.astype(np.int32) * 4, -32768, 32767).astype(np.int16)
    sf.write(os.path.join(OUT, "runs_ch1", f"run_{r['run']}.wav"), c1, 16000, subtype="PCM_16")
    t1, t0 = text(c1), text(c0)
    heard = ("ch1" if has(t1) else "") + ("+" if has(t1) and has(t0) else "") + ("ch0" if has(t0) else "")
    rows.append([r["run"], r["time"], r["result"], r["reason"], r["dur"], "right" if side == "R" else "left", g,
                 t1, t0, "pos" if heard else "neg", heard or "-"])
with open(os.path.join(OUT, "classes.tsv"), "w", newline="") as fh:
    wr = csv.writer(fh, delimiter="\t")
    wr.writerow(["run", "time", "result", "reason", "dur", "ch1_side", "gain_applied", "text_ch1", "text_ch0",
                 "class", "zakhar_heard_in"]); wr.writerows(rows)
from collections import Counter
print(f"classified {len(rows)}, left out {len(skipped)} (ambiguous channel): {skipped}")
print("class:", dict(Counter(r[9] for r in rows)), " heard in:", dict(Counter(r[10] for r in rows)),
      " gain x4 applied:", sum(1 for r in rows if r[6] == 4))
for r in rows:
    if r[9] == "pos" or r[7] or r[8]: print("\t".join(map(str, [r[0], r[1], r[5], r[6], r[9], r[10], r[7], "|", r[8]])))
print("CLASSIFY_DONE")
