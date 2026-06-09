# TRAINING REPORT v9 (DRAFT — autonomous breadth program over v8)

Scope: a model-/engine-side breadth program run while awaiting real device long-tail
captures. v1/v4/v5/v8 frozen and untouched. **Ship rule: a v9 candidate ships to
output_v9/ ONLY if it beats v8 on the leakage-safe DEVICE-EVAL (FRR at equal/lower
device-FAPH).** Manifest always `website=https://github.com/vvzvlad/zakhar-voice-assistant`,
`author=vvzvlad`.

## Incumbent to beat (v8)
- Device-eval FRR @0.9 = **0.059** (recall 0.941), device-FAPH(VAD)=0.
- Worst stratified modes @0.9: music_snr5 **0.216**, reverb+music10 **0.176**,
  muffled_lp3k 0.157, babble_snr10 0.124.

## Candidates (all on v9 multi-condition data; gated identically)
| candidate | lever | device-FRR@0.9 | music_snr5 | reverb+music10 | FAPH(VAD) | verdict |
|-----------|-------|---------------:|-----------:|---------------:|----------:|---------|
| v8 (incumbent) | — | 0.059 | 0.216 | 0.176 | 0 | baseline |
| v9-multicond (G5/G6) | far-field SLR28 + music/TV mix (aggressive) | 0.098 | 0.307 | 0.176 | 20/h noVAD | **REGRESSION — over-augmented** |
| v9-margin (G4) | margin-BCE (AM-softmax approx, m=0.35) | 0.111 | 0.314 | 0.196 | (noisy) | **REGRESSION — reject** |
| v9-res (G8) | residuals + multicond | 0.085 | 0.288 | 0.131 | ~10/h | mixed: **best reverb (0.033 vs 0.078)** & reverb+music, worse clean/music — reject but residuals help |
| v9-distill (G1) | teacher→student KD (α=0.6, 30k) | 0.170 | 0.451 | 0.366 | ~10/h | **REGRESSION — KD hurt at our data scale** |
| v9b (G5/G6+G8) | residuals + v8-intensity aug + SLR28 banks | 0.098 | 0.333 | 0.170 | ~10/h | **REGRESSION — SLR28 RIR bank hurts even at v8 intensity** |
| v9r8 (G8) | residuals on v8's EXACT data (no SLR28) | 0.137 | — | — | ~10/h | **REGRESSION — residuals HURT (0.065→0.137 on identical data)** |
| _control_ v8 last_weights | (rule out best-vs-last confound) | 0.065 | — | — | ~10/h | ≈ v8 best (0.059) — confirms pipeline sound |

## Per-goal findings (see individual docs)
- G2 verifier — `G2_VERIFIER.md` (65-param logreg, keeps 100% TP / rejects 100% music-FA, 0.79 margin).
- G3 QAT — `G3_QAT.md` (PTQ lossless; QAT not warranted).
- G7 stratified eval — `strat_eval.py`, v8 worst-case profile.
- G9 version history — `VERSION_HISTORY_SUMMARY.md`.
- G10 production loop — `PRODUCTION_LOOP.md`.
- G11 research — `RESEARCH_NOTES.md`.
- G8 arch — `G8_ARCH_SWEEP.md`.

## Decision
_To be filled once candidates evaluated. If a candidate beats v8 → promote to output_v9/
with manifest + this report's final table. If none beat v8 → keep v8, report the negative
result honestly (the breadth work still de-risks the next real-data iteration)._
