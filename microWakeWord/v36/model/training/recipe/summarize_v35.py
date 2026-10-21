#!/usr/bin/env python3
"""Collect the v35 ship-gate outputs @0.80 into one table (v34 summarize.py plus two rows): recall with a
95% Wilson CI, room FAPH with the Poisson CI from room_bg_faph2.py, «сахар» clips firing, the beep-cued
clips (held out per clip via heldout_clips_eval.py; held out and train-side in the rebuilt session take via
studio_eval.py), and the export's built-in streaming ROC (FRR at the FA/h points
the three ROC files share; at any FA/h a model's FRR is the one of its ROC point with the largest
FA/h not above it).
Usage: summarize_v35.py <eval_dir> <label:tflite_streaming_roc.txt> ..."""
import sys, os, json, math, re
E = sys.argv[1]; CUT = "0.80"; LABS = ["v33", "v34", "v35"]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def rec(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"

studio = {}
for line in open(os.path.join(E, "studio.jsonl")):
    r = json.loads(line); studio[r["label"]] = r
newho = {}
for line in open(os.path.join(E, "heldout_new.jsonl")):
    r = json.loads(line); newho[r["label"]] = r
def by_label(name):
    out = {}
    for line in open(os.path.join(E, name)):
        r = json.loads(line); out[r["label"]] = r
    return out
session = by_label("session.jsonl")
newho_nb = by_label("heldout_new_nobeep.jsonl"); session_nb = by_label("session_nobeep.jsonl")

# panel.out: '=== vNN ===' header, then rows 'cut HELD-OUT% all% events% hardFA totaldets'
panel, cur = {}, None
for line in open(os.path.join(E, "panel.out")):
    m = re.match(r"=== (v\d+) ===", line.strip())
    if m: cur = m.group(1); continue
    f = line.split()
    if cur and f and f[0] == "0.80":
        panel[cur] = round(float(f[1].rstrip("%")) * 19 / 100)

def dfjson(lab, kind):
    r = json.load(open(os.path.join(E, f"{lab.upper()}_{kind}.dfjson")))
    fired = sum(1 for x in r["rows"] if x["det"][CUT] > 0)
    return fired, r["n"], max(x["maxwin"] for x in r["rows"])

rows = [("new held-out, per clip (x4)", lambda l: rec(newho[l]["hits"], newho[l]["n"])),
        ("new held-out, session (x4)", lambda l: rec(*session[l]["heldout"])),
        ("new train-side, session (x4)", lambda l: rec(*session[l]["train"])),
        ("no beep: held-out, per clip", lambda l: rec(newho_nb[l]["hits"], newho_nb[l]["n"])),
        ("no beep: held-out, session", lambda l: rec(*session_nb[l]["heldout"])),
        ("no beep: train-side, session", lambda l: rec(*session_nb[l]["train"])),
        ("studio held-out words (x4)", lambda l: rec(*studio[l]["heldout"])),
        ("studio all segments (x4)", lambda l: rec(*studio[l]["all"])),
        ("speaker 18 home takes (x4)", lambda l: rec(*dfjson(l, "spk18")[:2])),
        ("panel held-out words (x1)", lambda l: rec(panel[l], 19))]
print(f"{'@0.80':<30}" + "".join(f"{l:<30}" for l in LABS))
for name, fn in rows:
    print(f"{name:<30}" + "".join(f"{fn(l):<30}" for l in LABS))
faph = []
for l in LABS:
    r = json.load(open(os.path.join(E, f"{l.upper()}_roomheld.json")))
    s = next(x for x in r["sweep"] if abs(x["cut"] - 0.8) < 1e-9)
    faph.append(f"{s['det']} FA in {r['hours'] * 60:.0f} min = {s['faph']}/h [{s['ci'][0]}, {s['ci'][1]}]")
print(f"{'room held-out FAPH (x4)':<30}" + "".join(f"{x:<30}" for x in faph))
sah = []
for l in LABS:
    k, n, mw = dfjson(l, "sahar"); sah.append(f"{k}/{n} fire, maxwin {mw:.3f}")
print(f"{'«сахар» false accepts (x4)':<30}" + "".join(f"{x:<30}" for x in sah))
print("studio detections total @0.80: " + ", ".join(f"{l} {studio[l]['dets']}" for l in LABS))
print("session detections total @0.80 (28 kept words in 120 s): " + ", ".join(f"{l} {session[l]['dets']}" for l in LABS))
print("new held-out clips, detections outside the word window @0.80: " +
      ", ".join(f"{l} {newho[l]['outside']}" for l in LABS))

# export's built-in streaming ROC
roc = {}
for spec in sys.argv[2:]:
    lab, path = spec.split(":", 1); pts = []
    for line in open(path):
        m = re.match(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)", line.strip())
        if m: pts.append((float(m.group(3)), float(m.group(2)), float(m.group(1))))
    roc[lab] = pts
def frr_at(pts, x):
    ok = [p for p in pts if p[0] <= x + 1e-9]
    best = max(ok, key=lambda p: (p[0], -p[1]))
    return f"{100 * best[1]:.1f}% (cut {best[2]:.2f}, {best[0]:.3f}/h)"
grid = sorted({p[0] for pts in roc.values() for p in pts if 0 < p[0] <= 2.0})
print("\nexport streaming ROC: FRR at FA/h <= x (each model's own cfg test set)")
print(f"{'FA/h':<10}" + "".join(f"{l:<32}" for l in roc))
for x in grid:
    print(f"{x:<10.3f}" + "".join(f"{frr_at(roc[l], x):<32}" for l in roc))
