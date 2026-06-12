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

## #1 STREAMING-REFIT VERIFIER — VERDICT: on-device 2-stage does NOT transfer (data-limited)
Exposed the bottleneck as a 3rd tflite output (`--angular_head 5`), harvested STREAMING-INT8
embeddings at trigger frames (real negs FAs + positives), refit the verifier logreg on those,
baked into the dual-output (`--angular_head 4`). Domain mismatch FIXED (a qualitative check shows
silence FAs now get verify≈0.01–0.06 vs 0.65–0.99 with the offline verifier). BUT on HELD-OUT
streaming FAs the verifier does NOT generalise:
| vthr | keepTP | FRR_comb | silence | music | speech |  (angular-alone: silence 28.4, music 37.8) |
|-----:|-------:|---------:|--------:|------:|-------:|
| 0.5 (1280-d) | 0.983 | 0.039 | 21.3 | 28.3 | 1.7 |
| 0.5 (64-d pooled) | 0.983 | 0.039 | 24.8 | 22.1 | 5.8 |
At any vthr that keeps recall, held-out silence/music FAPH stays ~22–28/h (≈ angular-alone) — only
~25% of FAs rejected, not the 97% seen on the TRAINING FAs. **Root cause: the verifier OVERFITS the
small real-FA training set (~98 negs, even at 64-d / strong regularisation); held-out streaming-int8
embeddings of silence/music FAs overlap true-wake embeddings too much for a linear verifier to
separate generally.** The offline 3.8%/0.85 was optimistic (domain mismatch + overfit).
**VERDICT: do NOT build the firmware 2-stage yet — it won't help on held-out with current FA data.**
It needs MUCH more real FA data (e.g. the incoming SOVA radio negs) to train a generalising verifier.
The recall path that DOES work on current data = duration-aware standard model + REAL short «захар» negs (#A).

## #2 max-pool detection-loss — also regresses (detection-head under-powered here)
v24 = per-frame head + MAX-pool window loss (Apple OWM, fixing the noisy-OR inflation). Operating
points (subset): @0.9 FRR 43%/sil 1.7/mus 5.8; @0.7 FRR 34%/sil 6.7/mus 14.5; @0.5 FRR 28%/sil 7.5/
mus 20.4. WORSE than v16 (21%/0/5.8) on BOTH axes at every point. Max-pool fixed the neg-inflation
(FAPH lower than noisy-OR's) but recall still regresses — the per-frame 1×1-conv detection head is
less expressive than v8's Flatten+Dense, and window-pool training doesn't recover it. The Apple OWM
gain does not materialise in this 32k mWW arch. Both detection-aware variants (noisy-OR, max-pool)
rejected; v16 head stays. (A wider detection head or longer training might help — low priority.)

## #E YANDEX CLONES — voice-safe split + positive ablation (held-out VOICES eval)
Voice-safe split: yandex grouped by VOICE (yx_<voice>), section by uuid; train voices (14) disjoint
from eval voices (5: ermil/filipp/jane/marina/omazh). One v16 recipe, vary ONLY positives:
| variant | FRR | silence | music | speech | vacuum |
|---------|----:|--------:|------:|-------:|-------:|
| (a) real section only (204) | 34.8% [29.6–40.3] | 15.0 | 55.3 | 11.8 | 41.1 |
| **(b) real + FULL yandex (986)** | **21.7% [17.4–26.8]** | 3.3 | 5.8 | 1.0 | 0.0 |
| (c) real + DEDUP yandex (256) | 32.4% [27.4–37.9] | 10.8 | 14.5 | 2.0 | 0.0 |

**Verdict 1 — NO speaker-leakage inflation:** (b) on the voice-safe (held-out-voices) eval = FRR
21.7% ≈ the source-split v16's 21.0%. Recall generalises to UNSEEN voices at the same level → the
clones did NOT leak across the source-split; prior ~21% recall numbers are HONEST, not optimistic.
**Verdict 2 — yandex clones HELP:** full yandex (21.7%) ≫ real-only (34.8%, also catastrophic FAPH:
music 55, vacuum 41 — 204 positives is too few) > dedup (32.4%). The clone VOLUME provides positive
coverage real-only lacks; dedup throws most of it away. **This REVISES v11b** (yandex "hurt" real-FRR):
that was on the non-device real_people eval; on the honest DEVICE-tract eval, device-recorded yandex
clones HELP both recall and FAPH. → KEEP full yandex in positives (v16 already does). Reports in nr2/.

## #F HONEST DENSE-SPEECH FAPH (real radio, 3h held-out, leakage-safe contiguous-block split)
| model | speech-FAPH @0.9/win5 | @0.95 | @0.99 |
|-------|---------------------:|------:|------:|
| **v16** | **1.3/h** | 1.3 | 0.7 |
| v19 angular | **21.5/h** | 16.2 | 7.0 |
**v16 is ROBUST on dense Russian speech (1.3/h); the angular "best recall" model false-fires ~21/h**
— the research suspicion confirmed, and the old small speech-eval (speech-FAPH 0) completely hid it.
The angular recall advantage comes with catastrophic dense-speech FAPH (even 7/h at cut 0.99). →
strongly reinforces v16 as the robust single-model ship; the angular line needs its dense-speech FAPH
cut (by training negs #H or the verifier #G) before its recall is usable.
(Radio: 1231 clips, 32 clipped dropped, 1199 kept; train 7h / eval 3h, contiguous-block session-safe.)

## #G RETRY 2-STAGE VERIFIER with LARGE radio FA pool — VERDICT: still not viable
Harvested streaming-INT8 trigger embeddings from radio-FA (195) + old classes (98) = 293 real FA
(vs the previous 98), refit the streaming-domain verifier, tested on HELD-OUT FAs (incl 64 radio).
Reject rates @keepTP 97.7% (vthr 0.5): silence 44% / music 25–33% / speech 69–77% / **radio 53%**.
More FA helped (held-out reject ~25% → ~44–53%) but the verifier STILL passes ~half the held-out FAs.
Combined v19+verifier ≈ FRR ~5%, but silence ~16/h, music ~25/h, radio ~10/h (no-VAD) — still far
worse than v16 (0 / 5.8 / 1.3). **The angular embedding does not LINEARLY separate held-out FAs from
true wakes well enough; the on-device 2-stage does NOT become viable even with a large real-FA pool.**
→ Firmware 2-stage (option B) NOT recommended. (A nonlinear verifier might push further but the spec
is logreg; the fundamental embedding overlap caps a linear gate at ~50% held-out reject.)

## #H RADIO AS TRAINING NEGATIVE — neither beats v16
| model | FRR | silence | music | speech | radio-FAPH |
|-------|----:|--------:|------:|-------:|-----------:|
| v16 (baseline) | 21.0% | 0 | 5.8 | 0 | 1.3 |
| v25 = v16 + radio negs | 33.7% | 2.5 | 5.8 | 0 | **0.3** |
| v26 = angular + radio negs | **2.2%** | 33.4 | 49.5 | 5.9 | **3.6** |
- **v25:** radio negs cut radio-FAPH 1.3→0.3 BUT cost recall 21→34% — not worth it (v16 already robust
  on dense speech at 1.3/h; the dense-speech penalty over-conservatised the boundary).
- **v26:** radio negs DID cut the angular dense-speech FAPH 21.5→3.6 — **model-level suppression WORKS
  for SPEECH** (unlike silence/music, which stay 33/50). New finding: the angular FAPH is class-specific
  — speech is suppressible by negs, silence/music are not. But silence/music keep the angular non-viable.

# ============ FINAL (round: radio dense-speech negatives) ============
SHIP UNCHANGED: **output_v16/** single-model — and #F now PROVES it's robust where the angular line
isn't: dense-speech FAPH 1.3/h (angular 21.5/h). Round verdicts:
- #F: v16 robust on dense Russian speech (1.3/h); angular "best recall" falls apart (21.5/h) — old eval hid it.
- #G: on-device 2-stage NOT viable even with the large radio FA pool (held-out reject caps ~50%, FAPH stays
  ~10–25/h) → firmware option B not recommended.
- #H: radio-in-training cuts the angular SPEECH-FAPH (21.5→3.6, model-suppressible) but not silence/music;
  for the standard model it costs recall (21→34%) for a FAPH gain v16 doesn't need. Neither beats v16.
The recall jump remains blocked: angular recall is un-deployable (silence/music FAPH un-suppressible by
model/cutoff/verifier); standard-model recall is duration-proxy-limited. Next levers: real SHORT «захар»
negs (#A path, de-risked) for the standard model. v1/v4/v5/v8 untouched. Not pushed to repo.

# ============ ROUND #I/#J/#K (synthetic short-захар negs · RepCNN · final DET) ============

## #I SYNTHETIC SHORT-«ЗАХАР» NEGATIVES — a single model that BEATS bare-v16
Recipe = v16 (real device pos + 4 real neg classes) + SYNTHETIC short-«захар» negatives made by
time-compressing the vowel of TRAIN positives (disjoint from eval), to teach duration-awareness.
Per operator correction (#H): **NO radio by default** (radio in v16 costs recall 21→34% for a FAPH
gain v16 doesn't need). Swept the short-neg weight (sampling/penalty):
| model | short-neg wt | FRR (recall) | short-fire@1.0s | silence | music | speech | radio |
|-------|:-----------:|------------:|----------------:|--------:|------:|-------:|------:|
| v16 (baseline)        | —    | 21.0% | **0.65** | 0   | 5.8 | 0 | 1.3 |
| **v27 = sc_lo (SHIP)**| 4/5  | **19.3%** [15.6–23.7] | **0.233** | 0.8 | 2.9 | 0 | 2.3 |
| sc_mid                | 6/8  | 34.5% | — | — | — | — | — |
| sc_hi                 | 8/10 | 29.8% | — | — | — | — | — |
- **sc_lo (v27) is the new best single-model:** it FIXES the production short-«захар» false-trigger
  (firing@1.0s 0.65→0.233, a 64% cut) WHILE slightly IMPROVING drawn-out recall (FRR 21.0→19.3%) and
  staying robust on every FAPH class incl. dense speech (radio 2.3/h, vs v16 1.3/h — both fine).
- Heavier short weights (sc_mid/sc_hi) over-penalise and cost recall (34/30%) — light (4/5) is the sweet spot.
- This is the first single model to beat bare-v16 on the short-захар bug at no recall cost, AND it tunes
  the recipe for tomorrow's REAL short-«захар» recordings (swap synthetic→real, keep light weight).
- Artifact: **output_v27/** zakhar.tflite 77848 B, manifest cutoff 0.9 / win5 (same firmware config as v16).

## #J RepCNN REPARAMETRIZATION (Apple 2406.02652) — exact fold, but NO recall gain
Trained a multi-branch over-parametrised pointwise (2× Conv1x1+BN + identity-BN, summed) `vrep`
(857KB h5, val recall 98.7%), then FOLDED to a single Conv1x1+bias (`nr2/repcnn_fold.py`):
- **Fold is numerically exact:** max|multi−single| = 9.0e-10 on random input; branch counts [2,3,3,3,3]
  (block1 no identity since in_ch 32≠64; blocks2-5 with identity). Folded single-branch tflite 77928 B —
  **same size/runtime as v16, fully deployable** (the RepCNN promise holds: zero inference cost).
- **But recall REGRESSES:** device FRR 27.9% [23.5–32.7] vs v16 21.0% (FAPH comparable: silence 1.7,
  music 2.9, speech 0, vacuum 0). Identical val recall (98.7%) yet worse on honest device-eval.
- **Verdict:** over-parametrised training does NOT break the ~21% duration-causality floor — the floor is
  the onset-proxy limitation (model keys on spectral onset «заха», not vowel duration), not optimiser
  capacity. RepCNN is a sound, deployable technique here but offers no recall lever for this task. Negative
  result, documented; not shipped.

## #K FINAL DET CURVES + OPERATING POINT (FRR vs aggregate FAPH over ALL classes incl. radio)
Full streaming DET swept over cutoff for v16 / v25 / v27(=sc_lo, #I) / vrepfold(=RepCNN, #J), both
no-VAD and VAD (device-speech-gated; radio is speech so VAD passes it). FAPH_all = total false-fires /
total hours over {silence, music, speech, vacuum, radio}. (`nr2/det_final.py`, `nr2/det_<m>.json`.)

**Pareto frontier (lower-left = better), no-VAD, selected points:**
| model | cut | FRR | FAPH_all | music | radio |
|-------|----:|----:|---------:|------:|------:|
| **v27** | 0.70 | 13.5% | 3.60 | 5.84 | 5.65 |
| **v27** | 0.85 | 16.3% | 1.89 | 2.92 | 2.66 |
| **v27** | 0.90 | 19.3% | 1.54 | 2.92 | 2.33 |
| **v27** | 0.95 | 22.7% | 0.86 | 0.00 | 1.33 |
| v16 | 0.50 | 13.5% | 4.98 | 20.44 | 5.32 |
| v16 | 0.85 | 20.2% | 1.03 | 5.84 | 1.33 |
| v16 | 0.90 | 21.0% | 1.03 | 5.84 | 1.33 |
| v25 | 0.90 | 33.7% | 1.03 | 5.84 | 0.33 |
| vrepfold | 0.80 | 22.1% | 2.23 | 14.60 | 1.99 |

**Findings:**
- **v27 and v16 are the two frontier models; v25 & vrepfold are Pareto-dominated** (every v25/vrepfold
  point has a v16 or v27 point with both lower FRR and lower FAPH). Drop them.
- **High-recall regime (FRR ≤ 17%): v27 STRICTLY dominates v16** — at FRR 13.5% v27 FAPH 3.60 vs v16 4.98;
  v27 keeps music/radio far lower across the board. If you want recall < 20%, v27 is unambiguously better.
- **At the ~21% knee they tie on aggregate FAPH** (v16 cut0.90 = 1.03 vs v27 cut0.92 = 1.37): v27 has much
  better MUSIC (2.92 vs 5.84) but slightly worse radio/silence; nets ~even. The tiebreaker is the
  short-«захар» bug: **v27 fires@1.0s 0.23 vs v16 0.65** — invisible in DET but the real field complaint.
- **VAD ~halves FAPH** for both, wiping silence/music/vacuum (non-speech); the residual is radio-type dense
  speech (VAD passes it). v27@0.90+VAD: FRR 19.3%, FAPH 1.20 (radio 2.33). VAD strongly recommended.
- **Cutoff sweep rejects the "0.95-optimal-for-v16" hypothesis:** v16's FAPH PLATEAUS at 1.03 from 0.85→0.95
  (music 5.84 is stuck until 0.97), so raising cutoff above 0.85 only costs recall (20.2→24.9%) with NO FAPH
  gain. v16's knee is **0.85–0.90**. v27 by contrast keeps lowering FAPH past 0.90 (0.90:1.54 → 0.95:0.86),
  so it has real cutoff headroom.

### RECOMMENDATION (ship)
- **Model: output_v27 (= v16 recipe + synthetic short-«захар» negs, light 4/5).** It matches v16 on the
  drawn-out-recall/FAPH frontier, strictly beats it in the high-recall regime, and — decisively — fixes the
  production short-«захар» false-trigger (65%→23%) that v16 ships with. Same size (77848 B) & firmware config.
- **Cutoff: 0.90 / win5** (= current firmware default): FRR 19.3%, FAPH ~1.5/h no-VAD, ~1.2/h with VAD.
  - Want fewer false-accepts: **0.95** → FRR 22.7%, FAPH 0.86/h (no-VAD) / 0.69/h (VAD).
  - Want max recall: **0.85** → FRR 16.3%, FAPH 1.9/h.
- **Enable VAD** on the device — it halves FAPH at zero recall cost; only dense-speech (radio/TV) remains,
  which no model/cutoff fully removes (the un-suppressible residual established in #F/#H).
- **RepCNN (#J)** is a validated, exact, zero-cost reparametrization but gives no recall gain here — keep on
  the shelf, not for this ship. **v25/radio-in-training** rejected (Pareto-dominated). v1/v4/v5/v8 untouched.

# ============ OVERALL SHIP DECISION ============
**Ship output_v27 at cutoff 0.90/win5, VAD on.** It is the first single model to beat bare-v16: same recall
& FAPH frontier PLUS it fixes the short-«захар» false-trigger. The synthetic-short-neg recipe (light 4/5
weight) is now tuned and ready to swap synthetic→REAL short-«захар» recordings when they arrive (expected to
push the short-fire rate further down at the same recall). NOT pushed to repo — operator collects output_v27/.
