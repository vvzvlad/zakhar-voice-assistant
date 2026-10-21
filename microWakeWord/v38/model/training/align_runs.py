#!/usr/bin/env python3
"""«захар» end in the positive run recordings (channel 1, x4 domain, whole recording) by CTC forced alignment
(torchaudio MMS_FA, romanized text) on the GigaAM text: channel 1's text when it holds a word starting «захар»,
else channel 0's. The target is the first such word. The speaker fired about 2 s into the recording (the
pre-roll), so «захар» should end shortly before that.
Usage: align_runs.py <classes.tsv> <runs_ch1_dir> <exclude_ids,comma> <out.tsv>"""
import sys, os, re, csv
import numpy as np, soundfile as sf, torch
from torchaudio.pipelines import MMS_FA as bundle
CL, D, EXC, OUT = sys.argv[1], sys.argv[2], set(sys.argv[3].split(",")), sys.argv[4]
TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
               "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "iu", "ia"]))
roman = lambda w: "".join(TR.get(c, "") for c in w)
words = lambda t: [w for w in re.sub(r"[^а-яё ]", " ", t.lower()).split() if roman(w)]
torch.set_num_threads(16)
model = bundle.get_model(with_star=False).eval(); tok = bundle.get_tokenizer(); ali = bundle.get_aligner()
res = []
for r in csv.DictReader(open(CL), delimiter="\t"):
    if r["class"] != "pos" or r["run"] in EXC: continue
    text = r["text_ch1"] if any(w.startswith("захар") for w in words(r["text_ch1"])) else r["text_ch0"]
    w = words(text); t = next(i for i, x in enumerate(w) if x.startswith("захар"))
    x, sr = sf.read(os.path.join(D, f"run_{r['run']}.wav"), dtype="float32"); assert sr == 16000 and x.ndim == 1
    with torch.inference_mode():
        em, _ = model(torch.from_numpy(x)[None])
    spans = ali(em[0], tok([roman(v) for v in w])); k = len(x) / em.shape[1] / sr
    s = spans[t]; nxt = round(spans[t + 1][0].start * k, 3) if t + 1 < len(w) else ""
    res.append([f"run_{r['run']}.wav", r["run"], " ".join(w), w[t], round(s[0].start * k, 3), round(s[-1].end * k, 3),
                nxt, round(len(x) / sr, 3), round(float(np.mean([v.score for v in s])), 3)])
with open(OUT, "w", newline="") as fh:
    wr = csv.writer(fh, delimiter="\t")
    wr.writerow(["file", "run", "text", "target", "t_start", "t_end", "next_start", "dur", "score"]); wr.writerows(res)
for x in res: print("\t".join(map(str, x)))
e = np.array([x[5] for x in res])
print(f"ALIGNED {len(res)}: «захар» end s min/p50/max {e.min():.2f}/{np.median(e):.2f}/{e.max():.2f}")
