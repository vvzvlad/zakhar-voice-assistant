#!/usr/bin/env python3
"""Why a per-clip replay disagrees with the continuous one: replay every phrase of a session with the speaker's
rule (session_check.dev_run) from different starting states and compare the phrase's peak window mean with the
continuous replay's.
  clip1    fresh frontend and fresh model from start-1 s (what a pos_NN.wav clip gives);
  feat1    the frontend run over the whole session (its features sliced out), fresh model from start-1 s;
  clip10   fresh frontend and fresh model from start-10 s.
A replay "flips" a phrase when it lands on the other side of the 0.80 cutoff than the continuous replay.
Usage: warmup_check.py <session.wav> <LABELS.tsv> <vad.tflite> <tflite> <session.json from session_check> <label>"""
import sys, json, numpy as np
WAV, LAB, VAD, TFL, SJ, LABEL = sys.argv[1:7]
sys.argv = ["session_check.py", WAV, LAB, "-", VAD, "/dev/null"]
import session_check as sc   # module-level code runs with no models: it only parses the labels and the wav

x = sc.x; feats = sc.frontend_v(x); cont = {p["id"]: p["dev_cont_peak"] for p in json.load(open(SJ))["models"][LABEL]["phrases"]}
def peak(q, t0, s, e):
    _, _, tr = sc.dev_run(TFL, q)
    v = [m for t, m in tr if s - sc.PRE <= t + t0 <= e + sc.POST]
    return max(v) / (sc.WIN * 256) if v else 0.0
rows = []
for r in sc.phr:
    s, e = r["s"], r["e"]; a = int(max(0, s - 1) * 16000); b = int(min(len(x), (e + 1.2) * 16000)); a10 = int(max(0, s - 10) * 16000)
    clip1 = peak(sc.dev_q(sc.frontend_v(x[a:b])), a / 16000, s, e)
    k0 = max(0, int(round((a / 16000 - 0.03) / 0.01))); k1 = int(round((b / 16000 - 0.03) / 0.01))
    feat1 = peak(sc.dev_q(feats[k0:k1]), 0.03 + 0.01 * k0 - 0.03, s, e)
    clip10 = peak(sc.dev_q(sc.frontend_v(x[a10:b])), a10 / 16000, s, e)
    rows.append((r["id"], cont[r["id"]], clip1, feat1, clip10))
c = np.array([r[1] for r in rows])
for i, name in ((2, "clip1"), (3, "feat1"), (4, "clip10")):
    v = np.array([r[i] for r in rows])
    flips = sum((a > 0.797) != (b > 0.797) for a, b in zip(c, v))
    print(f"{name:7s} |peak - continuous| median {np.median(np.abs(v - c)):.3f} p90 {np.percentile(np.abs(v - c), 90):.3f}"
          f"  flips vs continuous {flips}/{len(rows)}")
print("id  cont  clip1  feat1  clip10")
for r in rows: print(f"{r[0]} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f}")
