#!/usr/bin/env python3
"""Real-room FAPH on the operator's 20 min ch1 work-call background (device-faithful: int16 -> clamp(4x)
-> generate_features_for_clip, the ESP32 path; fresh state per file; win5-mean; 25-step/750ms lockout).
Same path as the positives, so pos & neg are processed identically. Per cut: FAPH + Poisson-exact CI.
Lists every false accept: file, mm:ss offset (to seek & listen), absolute wall-clock, peak window value.
Filenames encode the span END time (HHMMSS); start = end - 300 s. Report numbers only — no train/flash/deploy.
Usage: room_bg_faph.py <tflite> <label> [cuts]   Prints JSON + a readable FA table."""
import sys, os, glob, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
try:
    from scipy.stats import chi2
    def pois_ci(k, hrs, a=0.05):
        if hrs <= 0: return (0.0, 0.0)
        lo = chi2.ppf(a/2, 2*k)/2 if k > 0 else 0.0
        hi = chi2.ppf(1-a/2, 2*k+2)/2
        return (lo/hrs, hi/hrs)
except Exception:
    def pois_ci(k, hrs, a=0.05):
        if hrs <= 0: return (0.0, 0.0)
        import math; s = math.sqrt(k); return (max(k-1.96*s, 0)/hrs, (k+1.96*s)/hrs)
ROOT = "/home/claude/zakhar-mww"; STRIDE = 3; W = 5; LOCK = 25
import os as _os
D = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/device_captures/room_bg_ch1_20261016"
TFL, LAB = sys.argv[1], sys.argv[2]
CUTS = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else
        ["0.5","0.6","0.7","0.8","0.85","0.9","0.95"])]

def start_secs(fn):  # filename ...HHMMSS.wav = END of a 300 s span
    t = os.path.basename(fn).split("_")[-1].split(".")[0]
    h, m, s = int(t[:2]), int(t[2:4]), int(t[4:6])
    return h*3600 + m*60 + s - 300
def hhmmss(sec):
    sec = int(round(sec)); return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"
def load_i16(p):
    d, sr = sf.read(p, dtype="int16"); assert sr == 16000
    return (d[:, 0] if d.ndim > 1 else d).astype(np.int16)
def gain4(x): return np.clip(x.astype(np.int32)*4, -32768, 32767).astype(np.int16)

def det_times(ma, cut, sec_per_step):
    """lockout-gated detections; return list of (time_in_file_s, peak_window_value)."""
    out = []; cd = 0; i = 0; n = len(ma)
    while i < n:
        if cd > 0: cd -= 1; i += 1; continue
        if ma[i] >= cut:
            j = i
            while j < n and ma[j] >= cut: j += 1        # peak over the firing run
            pk = float(ma[i:j].max())
            out.append((i*sec_per_step, pk)); cd = LOCK
        i += 1
    return out

files = sorted(glob.glob(f"{D}/*.wav"))
assert files, f"no files in {D}"
per_file = []; total_h = 0.0
traces = []
for p in files:
    x = gain4(load_i16(p)); dur = len(x)/16000.0
    m = Model(TFL, stride=STRIDE)
    pr = np.asarray(m.predict_clip(x, step_ms=10), dtype=np.float32)
    ma = sliding_window_view(pr, W).mean(-1) if len(pr) >= W else np.array([0.0])
    sps = dur/len(ma) if len(ma) else 0.03
    traces.append((os.path.basename(p), start_secs(p), ma, sps, dur))
    total_h += dur/3600.0
rows = []; fa_list_at = {}
for cut in CUTS:
    allfa = []
    for name, st, ma, sps, dur in traces:
        for tin, pk in det_times(ma, cut, sps):
            allfa.append({"file": name, "offset_s": round(tin, 1), "mmss": f"{int(tin)//60:02d}:{tin%60:04.1f}",
                          "abs": hhmmss(st+tin), "peak": round(pk, 3)})
    k = len(allfa); lo, hi = pois_ci(k, total_h)
    rows.append({"cut": cut, "det": k, "faph": round(k/total_h, 2) if total_h else 0.0,
                 "ci": [round(lo, 2), round(hi, 2)]})
    fa_list_at[cut] = allfa
print("ROOMFAPH " + json.dumps({"label": LAB, "gain": 4, "win": W, "hours": round(total_h, 4),
      "files": [{"f": t[0], "start": hhmmss(t[1]), "dur_s": round(t[4], 1)} for t in traces],
      "sweep": rows, "fa_at": {f"{c:.2f}": fa_list_at[c] for c in CUTS}}, ensure_ascii=False))
# readable
print(f"\n=== {LAB}  device-faithful x4  {total_h*60:.1f} min ===")
for r in rows:
    print(f"cut {r['cut']:.2f}: {r['det']:3d} FA  = {r['faph']:6.2f}/h  [95% Poisson {r['ci'][0]:.2f},{r['ci'][1]:.2f}]")
print("\nFalse accepts @0.80 (file  mm:ss  abs-clock  peakwin):")
for fa in fa_list_at.get(0.8, []):
    print(f"  {fa['file']}  {fa['mmss']}  {fa['abs']}  pk={fa['peak']}")
if not fa_list_at.get(0.8): print("  (none)")
