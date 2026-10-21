#!/usr/bin/env python3
"""Agreement of each replay of session_check.py with the speaker's live detections, for one model (the one the
speaker ran, v36): per phrase hit/miss in each replay, the 2x2 counts against the live column, and the time
offset replay - live on the phrases both caught.
Usage: session_agree.py <session.json> <label> [kind]   (kind: pos / neg / free; default: every phrase)"""
import sys, json, numpy as np
R = json.load(open(sys.argv[1])); LAB = sys.argv[2]; KIND = sys.argv[3] if len(sys.argv) > 3 else None
ph = [p for p in R["models"][LAB]["phrases"] if KIND is None or p["kind"] == KIND]
live = [bool(p["live"]) for p in ph]
print(f"{R['wav']} {LAB} kind={KIND or 'all'}: {len(ph)} phrases, live hits {sum(live)}")
variants = [("dev_cont", lambda p: bool(p["dev_cont"])), ("py_cont", lambda p: bool(p["py_cont"]))]
if "dev_clip" in ph[0]:
    variants += [("dev_clip", lambda p: p["dev_clip"]), ("py_clip", lambda p: p["py_clip"])]
for name, f in variants:
    rep = [f(p) for p in ph]
    both = sum(a and b for a, b in zip(live, rep)); ol = sum(a and not b for a, b in zip(live, rep))
    orp = sum(b and not a for a, b in zip(live, rep))
    line = f"  {name:9s} hits {sum(rep):2d}  both {both:2d}  only live {ol:2d}  only replay {orp:2d}  disagree {ol + orp:2d}/{len(ph)}"
    if name.endswith("cont"):
        dt = [p[name][0] - p["live"][0] for p in ph if p["live"] and p[name]]
        if dt: line += f"  dt replay-live median {np.median(dt):+.2f} s [{min(dt):+.2f}, {max(dt):+.2f}]"
    print(line)
m = R["models"][LAB]
print(f"  outside phrases: live {R['device_live_outside']}  dev_cont {m['dev_cont_outside']}  py_cont {m['py_cont_outside']}"
      f"  VAD-blocked {m['vad_blocked']}")
print("  id        live    dev_cont(peak)    py_cont(peak)" + ("    dev_clip(peak)  py_clip(peak)" if "dev_clip" in ph[0] else "") + "  text")
for p in ph:
    s = (f"  {p['id']:9s} {(str(p['live'][0]) if p['live'] else '-'):7s} {(str(p['dev_cont'][0]) if p['dev_cont'] else '-'):7s}({p['dev_cont_peak']:.3f})"
         f"   {(str(p['py_cont'][0]) if p['py_cont'] else '-'):7s}({p['py_cont_peak']:.3f})")
    if "dev_clip" in p:
        s += f"   {'HIT' if p['dev_clip'] else '-':4s}({p['dev_clip_peak']:.3f})   {'HIT' if p['py_clip'] else '-':4s}({p['py_clip_peak']:.3f})"
    print(s + f"  {p['text']}")
