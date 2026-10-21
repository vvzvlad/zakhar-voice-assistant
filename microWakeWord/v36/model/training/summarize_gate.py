#!/usr/bin/env python3
"""One table @0.80 from run_gate.sh outputs: recipe/summarize_v35.py with the models as arguments, one file
per model and check, and the v36 rows when their files exist (gate_fused.sh). Recall with a 95% Wilson CI;
false accepts per hour with the Poisson CI of room_bg_faph2.py; «сахар» clips firing.
The export's streaming ROC is not read here: the ROC files of v34/v35 were lost with /tmp (see the report).
Usage: summarize_gate.py <eval_dir> <label:tflite> ..."""
import sys, os, json, math, re
E = sys.argv[1]; CUT = "0.80"; LABS = [a.split(":", 1)[0] for a in sys.argv[2:]]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def rec(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"

def js(lab, name):
    p = os.path.join(E, f"{lab.upper()}.{name}")
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else None

def panel(lab):
    for line in open(os.path.join(E, f"{lab.upper()}.panel.out")):
        f = line.split()
        if f and f[0] == "0.80": return round(float(f[1].rstrip("%")) * 19 / 100)

def dfjson(lab, kind):
    p = os.path.join(E, f"{lab.upper()}_{kind}.dfjson")
    if not (os.path.exists(p) and os.path.getsize(p)): return None
    r = json.load(open(p))
    fired = sum(1 for x in r["rows"] if x["det"][CUT] > 0)
    return fired, r["n"], max(x["maxwin"] for x in r["rows"])

def faph(lab, name):
    p = os.path.join(E, f"{lab.upper()}_{name}.json")
    if not (os.path.exists(p) and os.path.getsize(p)): return None
    r = json.load(open(p)); s = next(x for x in r["sweep"] if abs(x["cut"] - 0.8) < 1e-9)
    return f"{s['det']} FA in {r['hours'] * 60:.0f} min = {s['faph']}/h [{s['ci'][0]}, {s['ci'][1]}]"

rows = [("new held-out, per clip (x4)", lambda l: rec(js(l, "heldout_new.json")["hits"], js(l, "heldout_new.json")["n"])),
        ("new held-out, session (x4)", lambda l: rec(*js(l, "session.json")["heldout"])),
        ("new train-side, session (x4)", lambda l: rec(*js(l, "session.json")["train"])),
        ("no beep: held-out, per clip", lambda l: rec(js(l, "heldout_new_nobeep.json")["hits"], js(l, "heldout_new_nobeep.json")["n"])),
        ("no beep: held-out, session", lambda l: rec(*js(l, "session_nobeep.json")["heldout"])),
        ("no beep: train-side, session", lambda l: rec(*js(l, "session_nobeep.json")["train"])),
        ("studio held-out words (x4)", lambda l: rec(*js(l, "studio.json")["heldout"])),
        ("studio all segments (x4)", lambda l: rec(*js(l, "studio.json")["all"])),
        ("speaker 18 home takes (x4)", lambda l: rec(*dfjson(l, "spk18")[:2])),
        ("panel held-out words (x1)", lambda l: rec(panel(l), 19)),
        ("room held-out FAPH (x4)", lambda l: faph(l, "roomheld")),
        ("«сахар» false accepts (x4)", lambda l: "%d/%d fire, maxwin %.3f" % dfjson(l, "sahar"))]
# v36 checks (gate_fused.sh); a row is printed only when every model has it
fused = [("(a) fused phrases, whole (x1)", "fused_ho"), ("(a') of them written fused (x1)", "fused_ho_nospace"),
         ("(b) same voices, isolated (x1)", "isolated_ho"), ("(d) fused «сахар…», whole (x1)", "confus_ho")]
for name, kind in fused:
    if all(dfjson(l, kind) for l in LABS):
        if kind == "confus_ho":
            rows.append((name, lambda l, k=kind: "%d/%d fire, maxwin %.3f" % dfjson(l, k)))
        else:
            rows.append((name, lambda l, k=kind: rec(*dfjson(l, k)[:2])))
if all(faph(l, "studiobg_ho") for l in LABS):
    rows.append(("(c) studio bg held-out (x1)", lambda l: faph(l, "studiobg_ho")))

W = 34
print(f"{'@0.80':<34}" + "".join(f"{l:<{W}}" for l in LABS))
for name, fn in rows:
    print(f"{name:<34}" + "".join(f"{fn(l):<{W}}" for l in LABS))
print("studio detections total @0.80: " + ", ".join(f"{l} {js(l, 'studio.json')['dets']}" for l in LABS))
print("session detections total @0.80 (28 kept words in 120 s): " + ", ".join(f"{l} {js(l, 'session.json')['dets']}" for l in LABS))
print("new held-out clips, detections outside the word window @0.80: " +
      ", ".join(f"{l} {js(l, 'heldout_new.json')['outside']}" for l in LABS))
