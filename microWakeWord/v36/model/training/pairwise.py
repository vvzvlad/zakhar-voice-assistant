#!/usr/bin/env python3
"""Paired comparison against a reference model @0.80 on the same clips, from run_gate.sh outputs:
the 18 home takes (device_faithful_stream, x4), the beep-free held-out studio clips streamed one by one
(heldout_clips_eval), and (a) the held-out voices' fused phrases (device_faithful_stream, x1).
For each model: clips caught only by the reference, only by the model, the exact two-sided McNemar p
(binomial on the discordant clips), and for the small sets the clip names with both models' max window.
Usage: pairwise.py <eval_dir> <ref_label> <label> ..."""
import sys, os, json
from math import comb
E, REF, LABS = sys.argv[1], sys.argv[2], sys.argv[3:]

def dfj(lab, kind):
    r = json.load(open(os.path.join(E, f"{lab.upper()}_{kind}.dfjson")))
    return {x["f"]: (x["det"]["0.80"] > 0, x["maxwin"]) for x in r["rows"]}

def ho(lab, name):
    r = json.load(open(os.path.join(E, f"{lab.upper()}.{name}")))
    return {x["clip"]: (bool(x["hit"]), x["maxwin_word"]) for x in r["rows"]}

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

checks = [("18 home takes (x4)", lambda l: dfj(l, "spk18"), True),
          ("no beep: held-out, per clip", lambda l: ho(l, "heldout_new_nobeep.json"), True),
          ("(a) fused phrases, whole (x1)", lambda l: dfj(l, "fused_ho"), False)]
for name, get, names in checks:
    ref = get(REF)
    print(f"== {name}: {REF} catches {sum(v[0] for v in ref.values())}/{len(ref)}")
    for lab in LABS:
        m = get(lab); assert m.keys() == ref.keys(), (name, lab)
        only_ref = sorted(k for k in ref if ref[k][0] and not m[k][0])
        only_m = sorted(k for k in ref if m[k][0] and not ref[k][0])
        line = (f"  {lab:12s} {sum(v[0] for v in m.values())}/{len(m)}  only {REF}: {len(only_ref)}  "
                f"only {lab}: {len(only_m)}  McNemar p={mcnemar(len(only_ref), len(only_m)):.3g}")
        if names:
            fmt = lambda ks: ", ".join(f"{k} {ref[k][1]:.3f}->{m[k][1]:.3f}" for k in ks) or "-"
            line += f"\n      only {REF}: {fmt(only_ref)}\n      only {lab}: {fmt(only_m)}"
        print(line)
