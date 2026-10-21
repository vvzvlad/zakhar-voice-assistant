#!/usr/bin/env python3
"""Word timestamps by CTC forced alignment with the known synthesis text (torchaudio MMS_FA, the
multilingual wav2vec2 aligner; input text romanized, as the bundle expects).

coart:  every clip of <audio_dir> (Fish «захар + команда»); the target is the first word «захар». The
        texts written fused («захарвключи свет») are split after «захар» for the aligner: the text says
        where the word boundary is, the aligner places it in the audio.
confus: only the phrases where a confusable («сахар», «гончар», «закат», ...) is followed by another word;
        the confusables are the pool's one-word texts plus every word starting with «сахар».
For each clip writes: file, voice_id, text, target word, its start/end (s), the next word's start (s),
the clip duration (s) and the mean frame posterior of the target's tokens (alignment confidence).
Usage: align_fa.py <coart|confus> <manifest.tsv> <audio_dir> <out.tsv> [workers]"""
import sys, os, csv, glob
import numpy as np, soundfile as sf

MODE, MAN, AUD, OUT = sys.argv[1:5]
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
               "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "iu", "ia"]))

def roman(w):
    return "".join(TR.get(c, "") for c in w.lower())

man = {r["file"]: r for r in csv.DictReader(open(MAN), delimiter="\t")}
files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUD, "*.wav")))
missing = [f for f in files if f not in man]
assert not missing, f"{len(missing)} clips without a manifest row, e.g. {missing[:3]}"

def words_target(text):
    w = text.split()
    if MODE == "coart":
        assert w[0].startswith("захар"), text
        if w[0] != "захар": w = ["захар", w[0][len("захар"):]] + w[1:]
        return w, 0
    single = {r["text"] for r in man.values() if len(r["text"].split()) == 1}
    for i, x in enumerate(w):
        if x in single or x.startswith("сахар"):
            return w, i
    return w, None

jobs = []
for f in files:
    w, t = words_target(man[f]["text"])
    if MODE == "confus" and (t is None or t == len(w) - 1 or len(w) == 1):
        continue
    jobs.append((f, w, t))
print(f"{MODE}: {len(files)} clips in {AUD}, {len(jobs)} to align", flush=True)

def work(chunk):
    import torch
    from torchaudio.pipelines import MMS_FA as bundle
    torch.set_num_threads(max(1, 64 // WORKERS))
    model = bundle.get_model(with_star=False).eval()
    tok = bundle.get_tokenizer(); ali = bundle.get_aligner()
    out = []
    for f, w, t in chunk:
        x, sr = sf.read(os.path.join(AUD, f), dtype="float32")
        assert sr == 16000 and x.ndim == 1, (f, sr, x.shape)
        with torch.inference_mode():
            em, _ = model(torch.from_numpy(x)[None])
        spans = ali(em[0], tok([roman(v) for v in w]))
        r = len(x) / em.shape[1] / sr                                   # seconds per emission frame
        s = spans[t]; nxt = spans[t + 1][0].start * r if t + 1 < len(w) else ""
        out.append([f, man[f]["voice_id"], man[f]["text"], w[t], round(s[0].start * r, 3), round(s[-1].end * r, 3),
                    round(nxt, 3) if nxt != "" else "", round(len(x) / sr, 3),
                    round(float(np.mean([v.score for v in s])), 3)])
    return out

if __name__ == "__main__":
    import multiprocessing as mp
    from torchaudio.pipelines import MMS_FA
    MMS_FA.get_model(with_star=False)                                   # download once, before the workers
    chunks = [jobs[i::WORKERS] for i in range(WORKERS)]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        res = [r for part in pool.map(work, chunks) for r in part]
    res.sort()
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["file", "voice_id", "text", "target", "t_start", "t_end", "next_start", "dur", "score"])
        wr.writerows(res)
    e = np.array([r[5] for r in res]); sc = np.array([r[8] for r in res])
    print(f"ALIGNED {len(res)}: target end s p10/p50/p90 {np.percentile(e, 10):.2f}/{np.median(e):.2f}/"
          f"{np.percentile(e, 90):.2f}; score p10/p50 {np.percentile(sc, 10):.3f}/{np.median(sc):.3f}")
