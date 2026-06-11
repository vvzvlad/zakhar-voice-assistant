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

## #A PROBE — do models fire on SHORT «захар» (compressed «захааар»)? YES (confirms duration verdict)
Firing rate (score≥0.9) on held-out positives compressed to target duration (RMS-normalised):
| targ | v16 | v17 | v19 |
|-----:|----:|----:|----:|
| 5.0s (drawn-out) | 0.75 | 0.91 | 0.98 |
| 1.3s | 0.15 | 0.70 | 0.85 |
| 1.0s (SHORT захар) | 0.65 | 0.68 | **0.93** |
| 0.8s | 0.60 | 0.72 | 0.72 |
All models fire substantially on SHORT «захар» → they don't use vowel duration (confirms the
DURATION_CAUSALITY verdict). v19 (best recall) is the WORST at distinguishing (93% fires on 1.0s).
A short «захар» (common word) would falsely trigger — a false-accept class the FAPH eval misses.

## #A PROTOTYPE — short-«захар» hard-negs instill DURATION-AWARENESS (standard model)
| model | drawn-out FRR | short-«захар» firing @1.0s | @0.8s |
|-------|--------------:|---------------------------:|------:|
| v16 (baseline) | 21.0% | 0.65 | 0.60 |
| **v21 = v16 + short-negs** | 26.0% | **0.175** | 0.235 |
| v19 angular | 3.0% | 0.93 | 0.72 |
| v22 = angular + short-negs | 5.2% | 0.93 | 0.94 |

**Result (PoC success):** the STANDARD model (v21) learns duration discrimination from synthetic
compress-shorts — firing on a 1.0 s short «захар» drops 65%→17% while drawn-out recall barely moves
(FRR 21%→26%, small cost). So short-«захар» as hard-negatives WORKS, and synthetic compression is a
valid de-risk for recording REAL short «захар». The ANGULAR model (v22) does NOT learn it (fires 93%
on short still — the margin overpowers the short-negs, same as it resists FAPH suppression). →
**duration-awareness is a STANDARD-model lever; pair with real short «захар» negs next round.**
The v21 recall cost (21→26%) is tunable down via lower short-neg weight.

## #B detection-aware loss (noisy-OR window pool) — regressed (impl issue)
v23 = per-frame detection head + noisy-OR soft-pool loss (instead of per-frame BCE). Per-frame head
converts to int8 streaming (feasibility ✓, 77 KB). But FRR 45% @0.9 (subset) vs v16's 21% — REGRESS.
Cause: noisy-OR over the FULL 20-frame window inflates NEGATIVES too (20 frames × small p → OR≈0.9),
forcing the model to over-suppress every frame → recall collapses. The Apple gain needs a MAX-pool or
a calibrated narrow soft-pool, not full-window noisy-OR. Documented as next-step; this variant rejected.

# ============ FINAL (round: real positives + the 4 follow-ups) ============
SHIP UNCHANGED: **output_v16/ (=output_best/)** single-model — FRR 21%, all-class FAPH 0 (VAD),
strictly better than v8. The 4 follow-ups produced INSIGHTS, not a better single model:
- **#A (headline):** models fire on SHORT «захар» (duration verdict confirmed); synthetic short-«захар»
  hard-negs INSTILL duration-awareness in the STANDARD model (short-firing 65%→17%, small recall cost
  21→26%); ANGULAR can't learn it. → next round: standard model + REAL short «захар» negs (de-risked).
- **#C:** DET curves — v16 flat ~21% recall floor (all-FAPH 0); v19 angular tunable (4.4%/5.8h → 10.8%/3.3h).
- **#D:** on-device 2-stage = dual-OUTPUT tflite [kws,verify] + per-trigger AND (spec+host-ref ready),
  BUT the verifier needs a STREAMING-INT8-embedding refit to actually work on-device (offline 3.8%/0.85
  doesn't transfer — key de-risk finding).
- **#B:** detection-aware feasible but the noisy-OR variant regressed; max-pool variant is the next-step.
Best recall path remains angular (FRR 3%) + a streaming-refit verifier (firmware 2-stage), OR the
duration-aware standard model with REAL short negs. v1/v4/v5/v8 untouched. Not pushed to repo.
