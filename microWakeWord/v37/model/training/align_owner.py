#!/usr/bin/env python3
"""Word timestamps of the owner's read clips by CTC forced alignment with the script text (torchaudio MMS_FA,
romanized input), as v36/align_fa.py does for the Fish clips, with the target word named per phrase:
  pos  «захар + команда»: the target is the first word «захар»;
  neg  the 22 confusable phrases: the target is the word that sounds like «захар» (TARGET below; for
       «за хлебом / за час / за хорошую» the second word, so the crop keeps the whole «за х…» unit);
       the 8 commands without one have no target and are not aligned.
Writes file, text, target, its start/end (s), the next word's start (s), the clip duration (s) and the mean
frame posterior of the target's tokens.
Usage: align_owner.py <LABELS.tsv> <pos_dir> <neg_dir> <out.tsv>"""
import sys, os, csv
import numpy as np, soundfile as sf, torch
from torchaudio.pipelines import MMS_FA as bundle

LAB, PDIR, NDIR, OUT = sys.argv[1:5]
TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
               "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "iu", "ia"]))
TARGET = {"сахар кончился": "сахар", "сахар в шкафу": "сахар", "добавь сахар в чай": "сахар",
          "купи сахар и молоко": "сахар", "загар уже сошёл": "загар", "заказ пришёл": "заказ",
          "оформи заказ на завтра": "заказ", "закат сегодня красивый": "закат", "захват держит плохо": "захват",
          "закрой захват покрепче": "захват", "заходи давай": "заходи", "захотел чаю": "захотел",
          "на базаре дешевле": "базаре", "комар кусается": "комар", "самовар закипел": "самовар",
          "гончар лепит чашку": "гончар", "товар уже на складе": "товар", "захлопни дверь": "захлопни",
          "за хлебом сходи": "хлебом", "за час успеем": "час", "за хорошую работу": "хорошую",
          "захожу в магазин": "захожу"}
roman = lambda w: "".join(TR.get(c, "") for c in w.lower())

jobs = []
for r in csv.DictReader(open(LAB), delimiter="\t"):
    w = r["text"].split()
    if r["kind"] == "pos":
        assert w[0] == "захар", r; jobs.append((r["clip"], PDIR, r["text"], w, 0))
    elif r["text"] in TARGET:
        jobs.append((r["clip"], NDIR, r["text"], w, w.index(TARGET[r["text"]])))
assert len(jobs) == 62, len(jobs)
torch.set_num_threads(16)
model = bundle.get_model(with_star=False).eval(); tok = bundle.get_tokenizer(); ali = bundle.get_aligner()
res = []
for f, d, text, w, t in jobs:
    x, sr = sf.read(os.path.join(d, f), dtype="float32"); assert sr == 16000 and x.ndim == 1
    with torch.inference_mode():
        em, _ = model(torch.from_numpy(x)[None])
    spans = ali(em[0], tok([roman(v) for v in w])); r = len(x) / em.shape[1] / sr
    s = spans[t]; nxt = round(spans[t + 1][0].start * r, 3) if t + 1 < len(w) else ""
    res.append([f, text, w[t], round(s[0].start * r, 3), round(s[-1].end * r, 3), nxt, round(len(x) / sr, 3),
                round(float(np.mean([v.score for v in s])), 3)])
with open(OUT, "w", newline="") as fh:
    wr = csv.writer(fh, delimiter="\t")
    wr.writerow(["file", "text", "target", "t_start", "t_end", "next_start", "dur", "score"]); wr.writerows(res)
for kind in ("pos", "neg"):
    rr = [x for x in res if x[0].startswith(kind)]
    st = np.array([x[3] for x in rr]); ln = np.array([x[4] - x[3] for x in rr]); sc = np.array([x[7] for x in rr])
    print(f"{kind}: {len(rr)} aligned; target start s p10/p50/p90 {np.percentile(st, 10):.2f}/{np.median(st):.2f}/"
          f"{np.percentile(st, 90):.2f}; length s p10/p50/p90 {np.percentile(ln, 10):.2f}/{np.median(ln):.2f}/"
          f"{np.percentile(ln, 90):.2f}; score p10/p50 {np.percentile(sc, 10):.3f}/{np.median(sc):.3f}")
print("ALIGN_DONE", OUT)
