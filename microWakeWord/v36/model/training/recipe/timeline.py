# Per-model detection timeline on one long ch1 capture, device-faithful (x4 gain, win5 mean, 750 ms lockout).
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
D = "/tmp/zstudio"; STEP = 0.03; W = 5; LOCK = 25
raw, sr = sf.read(f"{D}/take.wav", dtype="int16")
x4 = np.clip(raw.astype(np.int32) * 4, -32768, 32767).astype(np.int16)
f = len(raw) // 480
r = 20 * np.log10(np.sqrt(((raw[:f * 480].astype(np.float32) / 32768).reshape(f, 480) ** 2).mean(1)) + 1e-9)
act = r > np.percentile(r, 10) + 15
segs, i = [], 0
while i < f:
    if act[i]:
        j = i
        while j < f and act[j:j + 8].any(): j += 1
        if j - i >= 6: segs.append((i * 0.03, j * 0.03))
        i = j
    else: i += 1
print(f"words (energy segments): {len(segs)}")
for v in ["v33", "v32", "v31", "v28", "v19"]:
    pr = np.asarray(Model(f"{D}/{v}.tflite", stride=3).predict_clip(x4, step_ms=10), dtype=np.float32)
    ma = sliding_window_view(pr, W).mean(-1)
    out = []
    for cut in (0.80, 0.60, 0.40):
        dets, cd = [], 0
        for k, val in enumerate(ma):
            cd = max(cd - 1, 0)
            if cd == 0 and val >= cut: dets.append((k + W - 1) * STEP); cd = LOCK
        hit = sum(1 for a, b in segs if any(a - 0.3 <= t <= b + 1.0 for t in dets))
        out.append(f"@{cut:.2f}: {len(dets):2d} det, {hit:2d}/{len(segs)} words")
    seg_max = [ma[max(0, int((a - 0.3) / STEP) - W + 1):int((b + 1.0) / STEP)].max() for a, b in segs]
    print(f"{v}: " + " | ".join(out) + f" | per-word max win: median {np.median(seg_max):.2f}")
