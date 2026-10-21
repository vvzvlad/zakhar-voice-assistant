#!/usr/bin/env python3
"""Energy segmentation of the 40 beep-cued studio clips (raw ch1, gain 1), stage 1 of the v35 sort.

Per clip:
  - 30 ms RMS frames in dBFS and the timeline.py activity threshold: p10 + 15 dB over the clip;
  - beep: every clip starts inside the speaker's chime tail (dominant 1320 Hz). The beep ends at the
    first frame >= 3 whose level drops below the threshold. If that dip comes later than BEEP_MAX_S,
    the word ran into the beep and the two cannot be told apart by energy -> flagged;
  - word segments: the timeline.py finder (>= 6 frames, 8-frame gap bridging) run on the frames after
    the beep only, so the beep never joins a word segment;
  - word-like filter as studio_prep.py (v34): 0.18 <= dur <= 0.90 s AND peak frame RMS >= 3x noise
    (noise = median frame RMS of the clip, linear);
  - edge flags: a segment that starts on the beep's last frame or ends on the clip's last frame.
Writes <out>/segments.json and, for the STT check, <out>/stt_in/<clip>.wav: the clip after the beep,
peak-normalised to 0.9 (so the quiet word is not dwarfed by the beep), 16 kHz PCM16.
Usage: segment_clips.py <raw_dir> <out_dir>"""
import sys, os, glob, json, numpy as np, soundfile as sf

SR = 16000; FR = 480; STEP = 0.03
BEEP_MAX_S = 0.39          # the beep tail decays below threshold by 0.24-0.33 s in clips without overlap


def frame_db(raw):
    f = len(raw) // FR
    return 20 * np.log10(np.sqrt(((raw[:f * FR].astype(np.float32) / 32768).reshape(f, FR) ** 2).mean(1)) + 1e-9)


def find_segs(act, start):
    """timeline.py finder over act[start:], indices absolute."""
    f = len(act); segs, i = [], start
    while i < f:
        if act[i]:
            j = i
            while j < f and act[j:j + 8].any(): j += 1
            if j - i >= 6: segs.append((i, j))
            i = j
        else: i += 1
    return segs


src, out = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(os.path.join(out, "stt_in"), exist_ok=True)
res = []
for p in sorted(glob.glob(os.path.join(src, "*.wav"))):
    raw, sr = sf.read(p, dtype="int16")
    assert sr == SR and raw.ndim == 1, (p, sr, raw.shape)
    r = frame_db(raw); f = len(r)
    rms = np.sqrt(((raw[:f * FR].astype(np.float64)) .reshape(f, FR) ** 2).mean(1))
    noise = float(np.median(rms))
    thr = float(np.percentile(r, 10) + 15)
    act = r > thr
    beep_end = next((k for k in range(3, f) if not act[k]), f)
    segs = find_segs(act, beep_end)
    rows = []
    for i0, i1 in segs:
        dur = round((i1 - i0) * STEP, 2); pm = round(float(rms[i0:i1].max() / noise), 1)
        rows.append({"a": round(i0 * STEP, 2), "b": round(i1 * STEP, 2), "dur": dur, "peak_mult": pm,
                     "peak_db": round(float(r[i0:i1].max()), 1),
                     "wordlike": bool(0.18 <= dur <= 0.90 and pm >= 3.0),
                     "touches_beep": i0 == beep_end, "touches_end": i1 >= f})
    tail = raw[beep_end * FR:].astype(np.float64)
    pk = np.abs(tail).max() or 1.0
    sf.write(os.path.join(out, "stt_in", os.path.basename(p)), (tail / pk * 0.9 * 32767).astype(np.int16), SR,
             subtype="PCM_16")
    res.append({"clip": os.path.basename(p), "noise_db": round(float(np.percentile(r, 10)), 1), "thr_db": round(thr, 1),
                "beep_end_s": round(beep_end * STEP, 2), "beep_overlap": beep_end * STEP > BEEP_MAX_S,
                "segs": rows, "db": [round(float(v)) for v in r]})
    w = [s for s in rows if s["wordlike"]]
    print(f"{res[-1]['clip']} thr={thr:.0f} beep_end={beep_end * STEP:.2f}{' OVERLAP' if res[-1]['beep_overlap'] else ''} "
          f"segs={len(rows)} wordlike={len(w)} " +
          " ".join(f"[{s['a']:.2f}-{s['b']:.2f} d{s['dur']:.2f} x{s['peak_mult']} {s['peak_db']}dB"
                   f"{' WL' if s['wordlike'] else ''}{' @beep' if s['touches_beep'] else ''}{' @end' if s['touches_end'] else ''}]"
                   for s in rows))
json.dump(res, open(os.path.join(out, "segments.json"), "w"), indent=1)
print("SEGMENT_DONE", len(res))
