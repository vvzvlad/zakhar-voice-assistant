#!/usr/bin/env python3
"""The owner's sessions in the gate, from session_check.py outputs, by the speaker's rule on the continuous
stream (dev_cont, the replay that tracks the speaker):
  free session (test only): hits on the 32 phrases with a Wilson 95 % CI, detections outside the phrases,
            and pairwise against each reference: phrases caught only by the reference / only by the model,
            exact two-sided McNemar p, the phrase ids;
  read negatives: phrases that fired, for the models that did not train on them (TRAINED are marked).
Usage: summarize_owner.py <free.json> <read.json> <ref,ref> <trained,...> <label> ..."""
import sys, json
from math import comb, sqrt
FREE, READ, REFS, TRAINED = sys.argv[1], sys.argv[2], sys.argv[3].split(","), set(sys.argv[4].split(","))
LABS = sys.argv[5:]
F, R = json.load(open(FREE)), json.load(open(READ))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f}% [{100 * (c - h):.1f}, {100 * (c + h):.1f}]"

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

hits = lambda J, lab, kind=None: {p["id"]: bool(p["dev_cont"]) for p in J["models"][lab]["phrases"]
                                  if kind is None or p["kind"] == kind}
print(f"== free session {F['wav']}: {F['n_phrases']} phrases, speaker's rule on the continuous stream @0.80")
for lab in LABS:
    h = hits(F, lab); m = F["models"][lab]
    print(f"  {lab:12s} hits {wilson(sum(h.values()), len(h))}   outside the phrases {len(m['dev_cont_outside'])}"
          f" {m['dev_cont_outside']}   VAD-blocked {len(m['vad_blocked'])}")
print(f"  speaker live (v36): {sum(1 for p in F['models'][LABS[0]]['phrases'] if p['live'])} hits, outside {F['device_live_outside']}")
for ref in REFS:
    print(f"== free session pairwise vs {ref}")
    r = hits(F, ref)
    for lab in LABS:
        if lab == ref: continue
        m = hits(F, lab); a = sorted(k for k in r if r[k] and not m[k]); b = sorted(k for k in r if m[k] and not r[k])
        print(f"  {lab:12s} only {ref}: {len(a)} {a}  only {lab}: {len(b)} {b}  McNemar p={mcnemar(len(a), len(b)):.3g}")
print("== read negatives (30 phrases of the read session), fires @0.80, speaker's rule on the continuous stream")
for lab in LABS:
    if lab in TRAINED:
        print(f"  {lab:12s} trained on them: not counted"); continue
    h = hits(R, lab, "neg"); fired = sorted(k for k in h if h[k])
    txt = {p["id"]: p["text"] for p in R["models"][lab]["phrases"]}
    print(f"  {lab:12s} {len(fired)}/{len(h)} fired: " + (", ".join(f"{k} «{txt[k]}»" for k in fired) or "-"))
print("SUMMARY_DONE")
