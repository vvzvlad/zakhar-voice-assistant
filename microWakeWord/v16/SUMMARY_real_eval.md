# SUMMARY — device-eval round 2 (REAL device positives + real device negatives)

Eval: held-out device «захааар» positives (362, leakage-safe by source key) + real device
negatives held-out: silence 72min / music 21min / speech 61min / vacuum 16min. @0.9/win5.

## #0 RE-BASELINE (no-VAD) — existing models on the NEW real device-eval
| model | FRR (recall) | silence | music | speech | vacuum |
|-------|-------------:|--------:|------:|-------:|-------:|
| v8 | 21.3% [17.4–25.8] | 12.5/h | 14.5/h | 0 | 0 |
| v11 | 22.1% [18.1–26.7] | **0.8/h** | 5.8/h | 0 | 0 |
| v12 angular | **9.7% [7.0–13.1]** | 12.5/h | 26.2/h | 3.9/h | 0 |
| v13 baked | 21.5% | 6.7/h | 14.5/h | 1.0/h | 0 |

**Key:** on REAL drawn-out device «захааар», standard models (v8/v11/v13) miss ~21% — the recall
gap the TZ flagged. The **angular head (v12) is far better at recall (9.7%)** but worst FAPH.
v11 keeps the best FAPH. → #1 (add real positives to fix recall) and #4 (angular + real negs to
keep recall without FAPH) are the right moves.

## #1 (v16) and #4 (v17 angular, annealed) on device-eval (@0.9/win5)
| model | FRR | silence noVAD/VAD | music noVAD/VAD | speech noVAD/VAD |
|-------|----:|------------------:|----------------:|-----------------:|
| v16 = v8 + REAL positives + 4 real neg classes | 21.0% [17.1–25.5] | 0.0 / 0.0 | 5.8 / 0.0 | 0 / 0 |
| v17 = angular (s10,m0.2, **annealed**) + real negs | **6.4% [4.3–9.4]** | 27.5 / 5.8 | 29.1 / 0.0 | 6.9 / 4.9 |

- **#1 negative:** real positives did NOT fix recall (21%, same as v8/v11). The recall lever is the
  ANGULAR head (margin → generalises the onset embedding), not more positives. Matches the
  duration verdict (onset-spectral reliance; BCE memorises training sources, margin generalises).
- **#4 positive:** the operator's annealing fix (softmax warmup + margin ramp) lifted angular recall
  to **6.4%** (best of all models) — confirms angular is the recall lever and that v12's 9.7% was
  partly training instability. But FAPH stays high (angular over-confident; real negs don't suppress
  it at the model level). → next: #5 angular + verifier (now with far more real neg data).

## #5 (v18 = angular v17 + verifier, baked) — verifier doesn't bake into streaming
Offline verifier on v17 (much more real FA data): @0.4 keepTP 98.6%, reject silence 94%/music 100%/
speech 100% → offline-combined would be FRR ~7.8%, silence ~1.7/h, music/speech 0. BUT baked into
one tflite (output=kws·verify) in streaming/int8: FRR 6.9% but silence 22.5/h, music 23.3/h — the
per-frame product + windowed-mean does NOT replicate the per-trigger argmax-window verifier decision
(same failure as v13baked last round). The verifier only works as a true 2-stage (firmware), not baked.

## v19 = angular + HEAVY neg penalty (15/15/12) — FAPH unfixable at model level
FRR **3.0%** [1.7–5.4] (best recall yet) but FAPH WORSE not better: silence 28.4/h, music 37.8,
speech 10.8 (no-VAD); VAD: silence 5.0, music 0, speech 7.8. Heavier penalty just made the angular
head MORE aggressive (recall up, FAPH up). → **angular FAPH cannot be suppressed in-model** (penalty
8 and 15 both fail; cutoff fails — saturated; baked verifier fails — per-frame). Only the offline
per-trigger verifier (firmware 2-stage) cuts it. Angular models also false-fire on real TV speech
(7–11/h) which standard models (v11/v16) don't.

## STANDING (device-eval, deploy = VAD on)
| model | FRR | silence | music | speech | deploy |
|-------|----:|--------:|------:|-------:|--------|
| v8 (prod now) | 21.3% | 0(VAD) | 0 | 0 | — |
| **v16** = v8 + real positives + 4 real neg classes | 21.0% | **0** | **0** | **0** | single-model drop-in |
| v11 | 22.1% | 0 | 0 | 0 | (≈v16) |
| v17 angular | 6.4% | 5.8 | 0 | 4.9 | needs verifier |
| v19 angular (heavy pen) | **3.0%** | 5.0 | 0 | 7.8 | needs verifier |

## v20 (angular s=6) and #3 v0.3 ablation
- **v20 s=6: degenerate** — no usable operating point (FRR 0.003 @cut0.5 firing constantly; FRR 1.0
  @cut≥0.7). Low scale compresses cosine scores into a narrow band → no graded curve. s=10 is the
  angular operating sweet spot. Lower-s does NOT let real negs suppress FAPH (it kills the curve).
- **#3 v0.3 ablation: KEEP v0.3.** v16 without quiet v0.3 positives → FRR 25.7% (worse than v16's
  21.0%) + silence-FAPH 1.7 (vs 0). Quiet/far positives HELP generalisation; removing them hurts.

# ============ FINAL VERDICT ============
The recall↔FAPH split is fundamental this round:
- **SINGLE-MODEL ship → v16** (= v8 recipe + REAL device positives + 4 real neg classes, keep v0.3):
  FRR 21.0%, ALL-class FAPH 0 with VAD (silence/music/speech/vacuum). Strictly better than v8 (which
  has 12.5/h silence FAPH at the same 21% recall). Drop-in INT8 32k, no firmware change.
- **Best recall → v19 angular** (annealed, FRR 3.0%) — 7× better recall, BUT its FAPH (silence 5/h,
  speech 8/h even with VAD) is unfixable in-model; needs the OFFLINE per-trigger verifier (firmware
  2-stage; verifier rejects 94–100% of real FAs offline but can't be baked into one streaming output).
- The 21% standard-model recall floor is **fundamental** (duration eval: onset-spectral proxy, not
  duration). Real fix = short-«захар» hard-negatives (duration-awareness) → next round (recording pending).

RECOMMEND: ship **v16** now (single-model, fixes FAPH, same recall as prod v8). Pursue v19+verifier
(firmware 2-stage) OR short-«захар» negs (single-model duration-aware) for the recall jump next round.
