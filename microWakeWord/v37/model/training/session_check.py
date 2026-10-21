#!/usr/bin/env python3
"""Replay the owner's continuous ch1 sessions (already x4: the exact wake-word input, gain 1) and match the
detections to the spoken phrases; compare with the speaker's own live detections from LABELS.tsv.

Four replays per model, so a disagreement with the speaker can be traced to its cause:
  dev_cont  the speaker's rule (firmware/components/wake_word/wake_word.cpp) on the whole session in one
            stream: features quantized as next_slice() does (((v*256)+333)/666-128, clamped), the wake and
            VAD models fed slice by slice, a check on every fresh wake probability, detection when the
            5-window uint8 sum > cutoff*5 and the VAD window sum > vad_cutoff*5, then the window zeroed and
            100 slices of cool-off counted only on slices whose newest probability is below the cutoff;
            cutoffs are percent -> uint8 as zl_cutoff_u8() (80 -> 204, 30 -> 76);
  py_cont   device_faithful_stream.py's rule (microwakeword Model.predict_clip features, win5 float mean
            >= cut, lockout 25 inferences, no VAD) on the whole session in one stream;
  dev_clip  dev_cont's rule, fresh state per clip (pos_NN.wav / neg_NN.wav, 1 s pad each side);
  py_clip   device_faithful_stream.py itself: fresh state per clip (the per-clip replay).
A detection at time t (end of its last feature slice) belongs to phrase j when start_j - 0.2 <= t <=
end_j + 1.0 and t < start_{j+1} - 0.2; any other detection is outside the phrases.
Usage: session_check.py <session.wav> <LABELS.tsv> <clips_dir|-> <vad.tflite> <out.json> <label:tflite> ...
LABELS.tsv: the read session (clip, kind, line, text, start_s, end_s, ...) or the free one (start_s, end_s,
source, ...); both carry device_v36_detect_s."""
import sys, os, csv, json, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
import tensorflow as tf
from pymicro_features import MicroFrontend
from microwakeword.inference import Model

WAV, LAB, CLIPS, VAD, OUT = sys.argv[1:6]
MODELS = [m.split(":", 1) for m in sys.argv[6:]]
WAKE_CUT, VAD_CUT, CUT = 80, 30, 0.80
WIN, COOL, LOCK, STRIDE = 5, 100, 25, 3
PRE, POST, GAP = 0.2, 1.0, 0.2

def u8(percent):  # zl_cutoff_u8
    return int(percent / 100 * 255)

def frontend_v(x):
    """Raw frontend outputs (uint16 domain) of one continuous int16 stream, one row per 10 ms slice."""
    fe, b, i, out = MicroFrontend(), x.tobytes(), 0, []
    while i + 320 <= len(b):
        r = fe.process_samples(b[i:i + 320]); i += r.samples_read * 2
        if r.features: out.append(r.features)
    return np.rint(np.asarray(out, dtype=np.float64) * 25.6).astype(np.int64)

def dev_q(v):  # next_slice(): ((v*256)+333)/666 - 128, clamped to int8
    return np.clip((v * 256 + 333) // 666 - 128, -128, 127).astype(np.int8)

class Streamer:
    def __init__(self, path, cutoff):
        self.it = tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=
                                      tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.stride = int(self.inp["shape"][1]); self.buf = np.zeros(self.inp["shape"], np.int8)
        self.step = 0; self.idx = 0; self.probs = np.zeros(WIN, np.int64); self.fresh = False
        self.ignore = 0; self.cut = cutoff

    def feed(self, q):  # model_feed()
        self.step %= self.stride
        self.buf[0, self.step] = q; self.step += 1
        if self.step >= self.stride:
            self.it.set_tensor(self.inp["index"], self.buf); self.it.invoke()
            self.idx = (self.idx + 1) % WIN
            self.probs[self.idx] = int(self.it.get_tensor(self.out["index"])[0][0]); self.fresh = True
        if self.probs[self.idx] < self.cut: self.ignore = min(self.ignore + 1, 0)

    def above(self):
        return int(self.probs.sum()) > self.cut * WIN

def dev_run(wake_path, q):
    """The speaker's rule over one stream of quantized slices -> (detections, VAD-blocked, peak window mean)."""
    w, v = Streamer(wake_path, u8(WAKE_CUT)), Streamer(VAD, u8(VAD_CUT))
    det, blk, peak = [], [], []
    for k in range(len(q)):
        w.feed(q[k]); v.feed(q[k])
        t = 0.03 + 0.01 * k
        peak.append((t, int(w.probs.sum())))
        if not w.fresh or w.ignore < 0: continue
        w.fresh = False
        if not w.above(): continue
        if not v.above(): blk.append(round(t, 2)); continue
        det.append(round(t, 2)); w.probs[:] = 0; w.ignore = -COOL
    return det, blk, peak

def py_run(wake_path, x):
    """device_faithful_stream.py's rule over one stream -> (detections, window-mean trace)."""
    pr = np.asarray(Model(wake_path, stride=STRIDE).predict_clip(x, step_ms=10), dtype=np.float32)
    det, cd, tr = [], 0, []
    for i in range(WIN - 1, len(pr)):
        m = float(pr[i - WIN + 1:i + 1].mean()); t = 0.03 + 0.01 * (STRIDE * i + STRIDE - 1)
        tr.append((t, m)); cd = max(cd - 1, 0)
        if cd == 0 and m >= CUT: det.append(round(t, 2)); cd = LOCK
    return det, tr

def load(p):
    x, sr = sf.read(p, dtype="int16"); assert sr == 16000 and x.ndim == 1, (p, sr, x.shape); return x

rows = list(csv.DictReader(open(LAB), delimiter="\t"))
phr = [r for r in rows if r.get("start_s")]
for j, r in enumerate(phr):
    r["s"], r["e"] = float(r["start_s"]), float(r["end_s"])
    r["id"] = r.get("clip") or f"seg{j + 1:02d}"
dev_live = sorted(float(r["device_v36_detect_s"]) for r in rows if r.get("device_v36_detect_s"))

def assign(dets):
    """Detections -> {phrase id: [t]} and the list outside every phrase."""
    hit, out = {r["id"]: [] for r in phr}, []
    for t in dets:
        for j, r in enumerate(phr):
            nxt = phr[j + 1]["s"] - GAP if j + 1 < len(phr) else 1e9
            if r["s"] - PRE <= t <= r["e"] + POST and t < nxt: hit[r["id"]].append(t); break
        else: out.append(t)
    return hit, out

def peak_in(trace, r, j):
    nxt = phr[j + 1]["s"] - GAP if j + 1 < len(phr) else 1e9
    vals = [m for t, m in trace if r["s"] - PRE <= t <= min(r["e"] + POST, nxt)]
    return round(max(vals), 3) if vals else 0.0

x = load(WAV)
qdev = dev_q(frontend_v(x))
live_hit, live_out = assign(dev_live)
res = {"wav": os.path.basename(WAV), "n_phrases": len(phr), "device_live": dev_live,
       "device_live_outside": live_out, "models": {}}
for lab, path in MODELS:
    d_det, d_blk, d_tr = dev_run(path, qdev)
    p_det, p_tr = py_run(path, x)
    d_hit, d_out = assign(d_det); p_hit, p_out = assign(p_det); b_hit, _ = assign(d_blk)
    per = []
    for j, r in enumerate(phr):
        row = {"id": r["id"], "text": r.get("text") or r.get("vosk_text", ""), "kind": r.get("kind", "free"),
               "s": r["s"], "e": r["e"], "live": live_hit[r["id"]], "dev_cont": d_hit[r["id"]],
               "py_cont": p_hit[r["id"]], "vad_blocked": b_hit[r["id"]],
               "dev_cont_peak": round(peak_in(d_tr, r, j) / (WIN * 256), 3), "py_cont_peak": peak_in(p_tr, r, j)}
        if CLIPS != "-" and r.get("clip"):
            xc = load(os.path.join(CLIPS, r["clip"]))
            cd, _, ctr = dev_run(path, dev_q(frontend_v(xc)))
            pd, ptr = py_run(path, xc)
            row.update(dev_clip=len(cd) > 0, py_clip=len(pd) > 0,
                       dev_clip_peak=round(max(m for _, m in ctr) / (WIN * 256), 3),
                       py_clip_peak=round(max(m for _, m in ptr), 3))
        per.append(row)
    res["models"][lab] = {"dev_cont_det": d_det, "dev_cont_outside": d_out, "vad_blocked": d_blk,
                          "py_cont_det": p_det, "py_cont_outside": p_out, "phrases": per}
    print(f"{lab}: dev_cont {len(d_det)} det ({len(d_out)} outside), VAD-blocked {len(d_blk)}; "
          f"py_cont {len(p_det)} det ({len(p_out)} outside)", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print("SESSION_DONE", OUT)
