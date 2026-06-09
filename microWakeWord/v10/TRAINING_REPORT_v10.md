# V10 — TRUE angular-margin head (AM-Softmax/CosFace) results

Bar to beat = v8: device-FRR 0.046@0.8 / 0.059@0.9; worst-case strat music_snr5 @0.9 = 0.216,
reverb+music10 0.176, muffled 0.157; music-FAPH (40min) @0.95/win5 = 0.0/h.
GATE: ship to output_v10 ONLY if worst-case device-FRR ≥3pp better AND device-FAPH not higher.

## Convertibility & quantization (key wins)
- Angular head (UnitNormalization L2 + cosine Dense) converts cleanly through the mWW int8
  streaming path (Lambda failed clone_model; UnitNormalization works). tflite ~79–98 KB.
- **quant-gap ≈ 0** (int8 == float on device-eval, mean prob identical). The literature INT8
  angular-head warning did NOT materialise here (tiny embedding + per-channel int8).

## Sweep s∈{15,30} × m∈{0.20,0.35} (clean v8 data, 30k steps)
| cfg | dev-FRR@0.9 | worst strat@0.9 (music_snr5) | music-FAPH 40min @0.95/w5 | verdict |
|-----|-----------:|-----------------------------:|--------------------------:|---------|
| v8  | 0.059 | 0.216 | 0.0 | baseline |
| s15 m20 | 0.000 | 0.085 | **21.0** | reject: FAPH regress |
| s15 m35 | 0.013 | **0.052** | (dev-neg 30/h) | reject: FAPH regress |
| s30 m20 | 0.013 | 0.124 | (dev-neg 30/h) | reject: FAPH regress |
| s30 m35 | 0.007 | 0.065 | (dev-neg 50/h) | reject: FAPH regress |

**Finding:** the angular head massively improves recall & worst-case FRR (−13 to −16pp) with
zero quant loss, BUT the scale `s` saturates the sigmoid → negatives with modest cosine become
confident false-accepts → large music-FAPH regression. Bimodal output, not threshold-fixable.
Silence/idle FAPH = 0 for all (not the issue). → s∈{15,30} REJECTED on the FAPH gate.

**Fix under test:** lower s ∈ {5,8,10} (m=0.2) — keeps the margin-trained discriminative
embedding but restores graded outputs / FAPH control. Results appended below.

## Low-s sweep s∈{5,8,10}, m=0.20 (the FAPH-fix attempt)
| cfg | dev-FRR@0.9 | worst strat@0.9 | music-FAPH 40min @0.9/w5 (v8≈1.5) | verdict |
|-----|-----------:|----------------:|----------------------------------:|---------|
| s05 m20 | 0.007 | 0.183 | 7.5 (and breaks at cut 0.95: rejects all) | reject |
| s08 m20 | 0.026 | 0.157 (−5.9pp) | 6.0 | reject: FAPH regress |
| s10 m20 | 0.020 | 0.092 (−12.4pp) | 18–22 | reject: FAPH regress |

**Conclusion:** the TRUE angular head is a strong RECALL lever (worst-case FRR −6 to −16pp,
quant-gap ≈0) but at EVERY scale s it raises **music false-accepts** — the exact field bug.
Lower s trades away FRR gain without curing music-FAPH. None pass the "≥3pp FRR better WITHOUT
FAPH rise" gate → v8 stays. Direct fix to try: angular head + mined MUSIC hard-negatives
(suppress the music direction while keeping the margin recall gain). Combines mine_hardneg.py
(1:2 hard:random) with the angular trainer.

## Angular + mined MUSIC hard-negatives (1:2 hard:random, penalty 8)
| cfg | dev-FRR@0.9 | worst strat@0.9 | music-FAPH ambient @0.95/w5 (v8=0.0) | verdict |
|-----|-----------:|----------------:|-------------------------------------:|---------|
| v8 | 0.059 | 0.216 | 0.0 | baseline |
| s10 (no hn) | 0.020 | 0.092 | 18.0 | reject |
| **s10 + music-hn** | 0.020 | **0.078** (−13.8pp) | **9.0** (halved from 18) | closer, still FAPH>v8 |
Music hard-negs HALVE the angular head's music-FAPH while keeping the big FRR gain, but
synthetic fma negs don't fully generalise to held-out ambient music (9 vs v8's 0). Trying s=8+hn.

## Final: s08 + music-hn, and the conclusion
| cfg | worst strat@0.9 | music-FAPH ambient @0.95/w5 (v8=0.0) |
|-----|----------------:|-------------------------------------:|
| s10 + music-hn | 0.078 | 9.0 (best FAPH of the angular set) |
| s08 + music-hn | **0.059** (best FRR, −15.7pp) | 19.5 (hn made it more triggery) |

**VERDICT (v10): DO NOT SHIP.** Across 8 configs (s∈{5,8,10,15,30}, ±music-hard-negs) the TRUE
angular head consistently delivers a large worst-case-FRR improvement (−13 to −16pp, quant-gap
≈0) but ALWAYS raises music-FAPH (best achievable 9/h vs v8's 0/h). The margin makes the model
more confident/triggery → higher recall AND higher music false-accepts; the two move together.
Synthetic fma hard-negs halve the music-FAPH but don't reach v8 (don't generalise to held-out
ambient music). Per the gate (≥3pp FRR better WITHOUT FAPH rise) → v8 stays.

### This is a STRONG, actionable result (not a dead end)
The angular head is by far the best **recall** lever found (v9 had none). Two forward paths make
it shippable:
1. **Real music negatives** (incoming device data): real-tract music captures will suppress the
   music direction far better than synthetic fma — re-run the angular head (s≈10, m=0.2, +music
   hard-negs from REAL captures). This is the top model-side lever for the next iteration.
2. **Angular head + 2nd-stage verifier (G2)**: the angular tflite supplies the recall; the
   65-param verifier (keeps 100% TP, rejects music FA) cuts the residual 9/h music-FAPH toward 0
   at ~0 cost. A deployment combination worth A/B-ing once a verifier is trained on the angular
   model's embeddings.

## Path-2 VALIDATED: angular head + 2nd-stage verifier (shippable COMBINATION)
The 65-param logreg verifier on the angular model's L2-normalised bottleneck **rejects 100% of
held-out ambient-music false-accepts** (the windows causing the 9/h) while **keeping 98.1% of
true wakes** (margin: minTP 0.193 > maxFA 0.126). So the combination achieves:
- worst-case device-FRR **0.078** (−13.8pp vs v8 0.216), device-FRR@0.9 ~0.02
- music-FAPH driven to **~0** by the verifier (vs the model-alone 9/h)
- cost: ~2% true-wake (verifier), 260 B + ~0 latency (post-trigger).

This MEETS the v10 goal (worst-case FRR ≫3pp better, FAPH not raised) — but as a **2-stage
deployment** (tflite + post-trigger verifier), not a single model, and on a small held-out FA
set (N=8). Recommendation: validate on more music FAs / real data and integrate the verifier in
ESPHome (dual-head per output_v9/research/G2_VERIFIER.md) before swapping production. v8 stays
the live single-model default until then. Artifacts: candidates/angular_s10_hardneg.tflite +
candidates/angular_verifier_logreg.npz + research/verify_angular.py.
