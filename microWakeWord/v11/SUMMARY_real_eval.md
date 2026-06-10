# SUMMARY — all versions on the REAL device-tract eval (the night program)

Held-out real eval: 72 min real silence + 6 min real music (leakage-safe split) + FRR on
dev_heldout(153) & real_people(107, real human voices). Deploy point: cutoff 0.9 / window 5.

## Master table (no-VAD = model robustness; with-VAD = Silero gate)

| version | silence-FAPH no-VAD | music-FAPH no-VAD | with-VAD (sil/mus) | FRR dev@0.9 | FRR real_people@0.9 | ship |
|---------|--------------------:|------------------:|:------------------:|------------:|--------------------:|------|
| **v8** (incumbent) | 12.5/h | 9.3/h | 0 / 0 | 8.5% [5.0–14.0] | 4.7% [2.0–10.5] | live |
| **v11** = v8 + REAL sil/mus hard-negs | **0.8/h** | **0/h** | 0 / 0 | 12.4% [8.1–18.6] | **0.9% [0.2–5.1]** | **output_v11/** (single-model) |
| v11b = v11 + yandex pos | 0/h | 9.3/h@0.9 | 0 / 0 | 15.7% | 6.5% | reject (FRR worse) |
| v12 = angular + REAL negs | 12.5/h | 18.5/h | 2.5 / 0 | **2.0% [0.7–5.6]** | **0.0% [0–3.5]** | not alone (FAPH) |
| **v13** = angular + verifier(real) | **~0/h** | **~0/h** | ~0 / 0 | **~5%** | **~3%** | **output_v13/** (2-stage, best metrics) |

## The story (real data finally made the bug visible)
- v8 false-fires **12.5/h in real idle-room silence** (synthetic gave 0) — VAD masks it to 0, so
  the field bug is "VAD does all the work; any gap leaks". This was invisible until real negatives.
- **REAL silence/music hard-negs are the fix synthetics never could be**: v11 cuts silence
  12.5→0.8/h, music 9.3→0/h, and *improves* real-human FRR (4.7→0.9%). Drop-in single model.
- The **angular head** (v12) gives spectacular recall (FRR 2%/0%) but is too confident — real negs
  don't fix its FAPH; a **2nd-stage verifier** (v13) cuts it to ~0 while keeping 96.9% TP.
- **Analyses (all decisive negatives):** a firmware energy/RMS gate would HURT (silence FAs are
  AGC-loud −28 dBFS, louder than the quiet wakes −37..−72); SNR-adaptive thresholding can't fix
  high-confidence FAs. The model-level real-negative fix is the only thing that works.

## Recommendation
- **Ship v11 now** as the single-model upgrade: drop-in replacement, fixes the field bug
  (silence/music false-fires), best real-human FRR (0.9%), simple deploy at 0.9/win5. **Beats v8.**
- **v13 is the best overall** (0 silence/music FAPH + dev-FRR ~5% < v8's 8.5%) if the 2-stage
  verifier is integrated in ESPHome (dual-head). Validate its verifier on more held-out FAs first
  (current held-out FA N is small: 13 silence, 2 music).
- v1/v4/v5/v8 untouched. Manifests valid (website/author).

## Best version: **v11** for immediate ship (single model, beats v8 decisively on the field bug);
**v13** for the best achievable metrics once the verifier stage is deployed.

## v15 (STRETCH) — feature-KD from angular teacher → standard student: REJECT
| cfg | silence-FAPH @0.9/win5 | FRR dev | FRR real_people |
|-----|----------------------:|--------:|----------------:|
| v11 (best single) | 0.8/h | 12.4% | 0.9% |
| v15a (λ=0.5) | 4.2/h | 10.5% | 9.3% |
| v15b (λ=2.0) | 4.2/h | 11.1% | 6.5% |
Pulling the student's bottleneck toward v12's angular embedding (MSE) made it WORSE on both
silence-FAPH (inherits the angular head's silence-confusion) and real-people FRR (the teacher
embedding over-constrains the student's own task). Angular recall is NOT transferable to a
standard head via embedding-KD. **v11 stays the best single model; v13 the best 2-stage.**

## FINAL VERDICT (unchanged): ship **v11** (single-model, beats v8 on the field bug); **v13**
for best metrics if the verifier stage is integrated. BC-ResNet A/B not pursued (multi-hour port,
low EV vs the real-negative lever that decisively won).

## UPDATE — v13 deployable (baked single-output) does NOT beat v11
The 2-stage v13 numbers (silence/music ~0) were from the OFFLINE verifier on a tiny held-out FA
set (13 silence, 2 music). Baked the verifier into one tflite (output = P(wake)·P(verify), one
output, converts clean, NO firmware change) and evaluated in streaming/int8 on the full real eval:
| | silence-FAPH @0.9/win5 | music-FAPH | FRR real_people@0.9 |
|--|----------------------:|-----------:|--------------------:|
| **v11** | **0.8/h** | **0/h** | 0.9% |
| v13 baked | 6.7/h | 18.5/h | 0.9% |
The verifier (trained on only ~6 real music FAs) does NOT generalise to music in streaming/int8.
**v11 wins decisively on the field bug at the SAME real-user FRR.**

# FINAL ANSWER: ship **output_v11/** — single model, drop-in v8 replacement, no firmware/fork
change. Fixes the silence/music false-fire field bug (12.5→0.8/h, 9.3→0/h) at real-people FRR 0.9%.
Angular+verifier (v12/v13) revisit only later if more real music data + extra recall is wanted.
