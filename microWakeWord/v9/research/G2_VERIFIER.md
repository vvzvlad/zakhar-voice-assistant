# G2 — Second-stage verifier (no-VAD vocal-music false-accept cutter)

A tiny classifier that runs **only after the KWS triggers**, reading the KWS model's
own bottleneck embedding, to reject music/media false-accepts in **no-VAD** configs
(where the VAD pre-gate isn't used to save latency/RAM). Research-backed two-stage
pattern (Amazon ICASSP 2018; negative-mining Hou 2020).

## Method
- KWS = v8 student. Bottleneck = the pre-Dense feature. The Flatten output is 1280-d
  (≈20 time-steps × 64 channels). Time-pool to a **64-d channel embedding** (mean over
  time) → feed a classifier.
- Train: positives = 100 real device wakes + 600 clean synth; negatives = **music windows
  v8 itself falsely fires on** (mined with `v5/mine_false.py` from fma music + a 40-min
  ambient-music stream, thr 0.35–0.8). Eval is **source-disjoint**: 53 held-out device
  wakes + 24 music false-accepts from held-out music tracks (fma_eval + ambient).

## Result (held-out)
| verifier | params | keep true-wakes | reject music-FA | margin (minTP−maxFA) |
|----------|-------:|----------------:|----------------:|---------------------:|
| MLP 16-hidden on 1280-d | 20,513 | 0.887 (unstable) | 1.000 | ≈0 (overfits) |
| **logreg on 64-d pooled** | **65** | **1.000** | **1.000** | **0.793** |

→ The **65-param logistic regression on the time-pooled 64-d bottleneck** is both
cheaper and far more robust than the bigger MLP (which overfits the 1280-d vector).
On the held-out set it keeps **100% of true wakes** and rejects **100% of music
false-accepts**, with a large 0.79 confidence gap.

**Caveat (honest):** the eval music-FA set is small (N=24) — because v8 already rarely
false-fires on music (no-VAD music-FAPH ≈ 4.67/h per v8 report). The 0.79 margin gives
confidence the mechanism is real, but the exact FAPH-cut % should be re-confirmed once
real long-tail vocal-music captures land. Expectation: cuts the residual no-VAD
music-FAPH toward ~0 at ~0 true-wake cost.

## Cost on device (ESP32-S3)
- **Compute:** 64 multiply-adds + sigmoid, **once per trigger** (not per frame). The 64-d
  embedding is a mean-pool of activations the KWS already computes → effectively free
  (<1 µs, no extra model inference).
- **Flash/RAM:** 65 float32 = **260 B** weights. Negligible vs the 78 KB KWS model.
- **Latency:** ~0; runs in the trigger handler, before the wake is dispatched.

## ESPHome integration (two options)
1. **Dual-head tflite (recommended):** bake the pool+dense(1) as a second output head in
   the same `zakhar.tflite`. The mWW ESPHome component reads two outputs and fires only
   if `kws_prob ≥ cutoff AND verify_prob ≥ vthr`. Needs a small patch to the component to
   read/AND a second output tensor. Zero extra model load.
2. **Interim (no firmware change):** the existing **VAD pre-gate already drives music-FAPH
   to 0**, so the verifier's marginal value today is for *no-VAD / lowest-latency* setups.
   Ship the verifier weights (`v9/verifier/logreg64.npz`) + this doc; enable via option 1
   when the component patch lands.

## Artifacts
`v9/train_verifier.py`, `v9/extract_embeddings.py` (reusable 64-d bottleneck extractor,
also used by the G10 field loop), `v9/verifier/logreg64.npz` (deployable weights),
`v9/verifier/verifier_chosen.json` (metrics).
