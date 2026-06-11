# On-device 2-stage detector — DEPLOY SPEC (verifier)

Goal: ship the angular detector's recall (FRR ~3%) with its false-accepts cut by a per-trigger
verifier, as a real on-device 2-stage. The firmware fork already exists, so a small component
change is acceptable.

## Form: ONE dual-OUTPUT tflite (NOT a per-frame product)
The model emits TWO scalars per inference: `kws_prob` and `verify_prob`. Built via mixednet
`--angular_head 4`: backbone → Flatten → UnitNormalization(L2) → {cos head → kws_prob} and
{verify Dense(1) → verify_prob}; returns `[kws_prob, verify_prob]`. Converts cleanly to INT8
streaming (81 KB, two uint8 outputs). The earlier BAKED product (`output=kws·verify`, one output)
FAILED because the windowed-mean of a per-frame product ≠ the verifier's per-trigger decision —
so DO NOT bake; keep two outputs and AND them in firmware AT THE TRIGGER.

## I/O contract
- **Input:** unchanged — 40-band micro_speech spectrogram, feature_step_size 10 ms, stride 3,
  window = `spectrogram_length` (same tensor the current mWW component feeds).
- **Outputs (per inference, each scalar):**
  - `kws_prob`  = sigmoid(s·cosθ)  — the wake probability (use EXACTLY as today).
  - `verify_prob` = sigmoid(w·ê + b) — the verifier (logreg on the L2-normalised 1280-d bottleneck,
    baked into the graph so the firmware needs NO embedding export or extra math).
  - Output index order is assigned by the converter. In our build: **kws = output[1],
    verify = output[0]**. Pin by tensor name or calibrate once (kws has the higher max on a clear
    positive). Document the mapping in the component.
- **Firmware decision (per-TRIGGER):**
  1. Maintain the sliding-window mean of `kws_prob` over `sliding_window_size` (5) — exactly as now.
  2. A candidate trigger = windowed `kws_prob` ≥ `probability_cutoff` (0.9) with the usual cooldown.
  3. At that trigger frame, ALSO require `verify_prob` ≥ `verify_threshold` (0.5).
  4. Fire the wake iff BOTH hold. The verify check is taken ONCE at the trigger (per-trigger), not
     averaged — this is what makes it match the offline verifier.

## Parameters / cost
- Detector: ~32 k backbone params (unchanged). Verifier: 1281 params (1280 weights + bias), baked
  into the tflite as a Dense head → computed inside the normal forward pass.
- Extra compute vs current model: ~1280 MAC/frame for the verify Dense → negligible on ESP32-S3.
- Latency: unchanged (no second model, no embedding export). Flash: ~81 KB tflite (vs ~78 KB).
- Tunable: `verify_threshold` trades recall vs FAPH (see tuning below).

## Tuned operating point (offline, device-eval, more real FA data)
Verifier @ vthr 0.5: keep-TP 99.2%, reject silence 97% / music 100% / speech 100%.
Combined v19 + verifier ≈ **FRR ~3.8%**, silence-FAPH **~0.85/h**, music/speech **0** (no-VAD) —
i.e. the angular recall (≈7× better than the 21% single-model floor) with FAPH at the v11 level.
(Streaming/int8 dual-output validation: `nr2/dual_eval.py` numbers — see report.)

## Host reference
`nr2/dual_eval.py` runs the dual tflite frame-by-frame and applies the exact firmware logic above
(windowed kws + per-trigger verify), reporting FRR + per-class FAPH. Use it as the golden reference
when implementing the component change.

## Threshold tuning knob
- Lower `verify_threshold` → fewer true wakes lost, more FAs slip (toward angular-alone).
- Higher → more FAs rejected, more true wakes lost.
- 0.5 is the keep-TP≈0.99 / reject≈0.97+ point; sweep on the live panel against real FAPH.

## ⚠ CRITICAL FINDING — verifier must be trained on STREAMING-INT8 embeddings
The dual-output tflite converts and the per-trigger AND is the right firmware logic, BUT a direct
on-device check exposed a pitfall: the `verify_prob` channel is HIGH (0.65–0.99) on silence
false-accepts in streaming/int8, so it does NOT reject them on-device — even though the offline
verifier rejected 97%. Root cause: the verifier (`nr2/v19_verifier.npz`) was trained on the
OFFLINE embedding distribution (non-streaming float, taken at the argmax window of a front-padded
clip). On-device the embedding is STREAMING + INT8 + per-frame — a different distribution. The
offline numbers (FRR 3.8% / silence 0.85/h) are therefore OPTIMISTIC; the verifier does not
transfer as-is. (This is the same mechanism that sank the baked product head.)

**Required for deployment:** retrain the verifier logreg on STREAMING-INT8 per-frame embeddings
collected at real triggers (expose the L2-norm bottleneck as a temporary 3rd tflite output, run
the streaming model over the real neg/pos sets, harvest embeddings at trigger frames, fit the
logreg on THOSE, then bake into `verify_logit`). Until then, the 2-stage gain is unproven on-device.
The dual-output graph, the per-trigger firmware contract, the host reference, and the threshold-
sweep tooling are all ready — only the verifier weights need the streaming-domain refit.
