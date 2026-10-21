#!/usr/bin/env python3
"""Training clips from the speaker's rejected/empty wakes of the run log (channel 1, x4 domain, prepared by
classify_runs.py) and their LABELS.tsv:
  pos  [0, «захар» end + d] of the recording, d uniform in 100-300 ms from md5(file name) (the v36 Fish rule),
       «захар» end from align_runs.py;
  neg  the speaker's firing window: the first 3.2 s of the recording (the window of the studio background
       block); a shorter recording is taken whole.
Left out of both: runs whose channel 1 could not be told by level (classify_runs.py), and UNCERTAIN: runs whose
class GigaAM cannot settle (below). Every run gets a row in runs_classes.tsv with its class or the reason it
was left out.
Usage: prep_runs.py <classes.tsv> <levels.tsv> <runs_align.tsv> <runs_ch1_dir> <out_root>"""
import sys, os, csv, glob, hashlib, numpy as np, soundfile as sf
CL, LEV, ALI, D, OUT = sys.argv[1:6]
SR, WIN = 16000, int(3.2 * 16000)
# A wake followed at once by a command to the assistant, with no «захар» heard: the owner most likely said
# «захар» and GigaAM missed it, as it heard «сбер» for «захар» on channel 1 in runs 3839 and 3841; and 3960,
# where channel 1 heard «захар» and channel 0 a different phrase («я знаю что»).
UNCERTAIN = {"3721": "command without «захар» right at the wake («включи свет»)",
             "3730": "command without «захар» right at the wake («узнай у агента какие глазури заканчивают»)",
             "3851": "command without «захар» right at the wake (ch0 «ха расскажи анекдот»)",
             "4074": "command without «захар» right at the wake («включи на столько свет»)",
             "3960": "channels disagree: ch1 «захар», ch0 «я знаю что»"}
h = lambda s: hashlib.md5(s.encode()).hexdigest()
delta = lambda f: 0.1 + 0.2 * int(h(f)[:8], 16) / 0xFFFFFFFF
cls = {r["run"]: r for r in csv.DictReader(open(CL), delimiter="\t")}
lev = {r["run"]: r for r in csv.DictReader(open(LEV), delimiter="\t")}
ali = {r["run"]: r for r in csv.DictReader(open(ALI), delimiter="\t")}

def out_dir(name):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    assert not glob.glob(os.path.join(d, "*.wav")), f"REFUSE: {d} not empty"; return d

P, N = out_dir("runs_pos_train"), out_dir("runs_neg_train")
hdr = ["file", "run", "time", "result", "reason", "text", "text_ch1", "text_ch0", "class", "zakhar_heard_in",
       "ch1_side", "gain_applied", "zakhar_end_s", "clip_end_s"]
lab = {"pos": [], "neg": []}; table = []
for run in sorted(lev, key=int):
    if run not in cls:
        table.append([run, lev[run]["time"], "", "", "", "left out", "channel 1 ambiguous by level"]); continue
    r = cls[run]
    if run in UNCERTAIN:
        table.append([run, r["time"], r["text_ch1"], r["text_ch0"], r["class"], "left out", UNCERTAIN[run]]); continue
    x, sr = sf.read(os.path.join(D, f"run_{run}.wav"), dtype="int16"); assert sr == SR and x.ndim == 1
    f = f"run_{run}.wav"
    if r["class"] == "pos":
        a = ali[run]; te = float(a["t_end"]); cb = min(len(x), int(round((te + delta(f)) * SR)))
        sf.write(os.path.join(P, f), x[:cb], SR, subtype="PCM_16"); text, end = a["text"], round(cb / SR, 3)
    else:
        cb = min(len(x), WIN); sf.write(os.path.join(N, f), x[:cb], SR, subtype="PCM_16")
        text, te, end = r["text_ch1"] or r["text_ch0"], "", round(cb / SR, 3)
    lab[r["class"]].append([f, run, r["time"], r["result"], r["reason"], text, r["text_ch1"], r["text_ch0"], r["class"],
                            r["zakhar_heard_in"], r["ch1_side"], r["gain_applied"], te, end])
    table.append([run, r["time"], r["text_ch1"], r["text_ch0"], r["class"], r["class"], ""])
for k, d in (("pos", P), ("neg", N)):
    with open(os.path.join(d, "LABELS.tsv"), "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t"); wr.writerow(hdr); wr.writerows(lab[k])
with open(os.path.join(OUT, "runs_classes.tsv"), "w", newline="") as fh:
    wr = csv.writer(fh, delimiter="\t")
    wr.writerow(["run", "time", "text_ch1", "text_ch0", "gigaam_class", "used_as", "left_out_reason"]); wr.writerows(table)
from collections import Counter
print("used as:", dict(Counter(t[5] for t in table)), " of", len(table), "runs")
kp = np.array([float(l[13]) - float(l[12]) for l in lab["pos"]])
print(f"pos clips {len(lab['pos'])}: kept after «захар» s min/p50/max {kp.min():.3f}/{np.median(kp):.3f}/{kp.max():.3f}; "
      f"neg clips {len(lab['neg'])}: length s min/p50/max {min(float(l[13]) for l in lab['neg']):.2f}/"
      f"{np.median([float(l[13]) for l in lab['neg']]):.2f}/{max(float(l[13]) for l in lab['neg']):.2f}; "
      f"gain x4 applied pos {sum(1 for l in lab['pos'] if l[11] == '4')} neg {sum(1 for l in lab['neg'] if l[11] == '4')}")
