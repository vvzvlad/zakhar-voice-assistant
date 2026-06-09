# TRAINING REPORT v9 — autonomous model/engine breadth program over v8

A ~10-hour autonomous program of model- and engine-side experiments run **while
awaiting real device long-tail captures**. Goal: improve the «захааар» wake word
beyond v8, shipping a new version to `output_v9/` **only if it beats v8 on the
leakage-safe DEVICE-EVAL** (FRR at equal/lower device-FAPH). v1/v4/v5/v8 frozen and
untouched.

## VERDICT

**No model-side candidate beat v8. v8 remains the production model — we do NOT ship a
regression.** This is a well-controlled negative result: v8 is a strong, well-tuned
local optimum for the current data, and the highest-value lever now is the **real
field data arriving next**, not architecture/loss/augmentation tweaks. The program
delivered: a reusable eval/verifier/field-loop toolchain, a clean QAT finding, a
deployable 2nd-stage verifier (the one genuine enhancement that complements v8 without
regressing it), and an honest map of which levers help/hurt — all de-risking the next
real-data iteration.

## Incumbent (v8) — the bar
- Device-eval FRR **0.046 @0.8 / 0.059 @0.9** (recall 0.941), device-FAPH(VAD)=0.
- Control: v8 `last_weights` = 0.065 @0.9 ≈ best_weights → no best/last confound; our
  retrain pipeline is sound and reproduces v8.

## Candidates — all gated identically (device-eval + stratified by-mode)
| candidate | lever | dev-FRR@0.9 | music_snr5 | reverb+music10 | verdict |
|-----------|-------|------------:|-----------:|---------------:|---------|
| **v8 (incumbent)** | mit_rirs aug, no residual | **0.059** | 0.216 | 0.176 | **ship (unbeaten)** |
| v9-multicond (G5/G6) | SLR28 far-field + music/TV, aggressive | 0.098 | 0.307 | 0.176 | regression (over-aug) |
| v9-margin (G4) | margin-BCE (AM-softmax approx) | 0.111 | 0.314 | 0.196 | regression |
| v9-focal (G4) | focal loss γ=2 on v8 data | 0.144 | — | — | regression |
| v9-res (G8) | residuals + multicond | 0.085 | 0.288 | 0.131 | regression (best reverb though) |
| v9b (G5/6+G8) | residuals + v8-intensity aug + SLR28 | 0.098 | 0.333 | 0.170 | regression |
| v9r8 (G8) | residuals on v8's EXACT data | 0.137 | — | — | regression |
| v9-distill (G1) | teacher→student KD | 0.170 | 0.451 | 0.366 | regression |

## What we learned (clean attributions)
1. **Residuals HURT here**: on v8's identical data, adding `residual_connection 0,0,1,1,1`
   moved FRR 0.065 → 0.137 (2×). The earlier "residuals help" read was noise inside
   aug-degraded models. (G8)
2. **The SLR28 RIR bank hurts** even at v8's augmentation intensity — its rooms are too
   reverberant for this near-field-dominant device; v8's `mit_rirs` was better tuned. (G5)
3. **Aggressive multi-condition augmentation over-corrupts positives** → recall drops
   across all modes. Confirms the operator's "augmentation plateau ~20×" warning. (G6)
4. **KD gives no gain at our data scale** (~10k synth + ~900 real): the distilled student
   regressed badly despite a 99.2%-recall teacher. Matches the literature (KD needs
   >400k positives). (G1)
5. **Custom-loop loss variants both regress**: margin-BCE (AM-softmax approx) 0.111 and
   focal γ=2 0.144, vs v8 0.059. The framework's plain weighted-BCE training is best here;
   a faithful angular head (L2-norm + additive margin) — not the logit-shift approx — is the
   proper port if the loss lever is revisited on real data. (G4)
6. **QAT is unwarranted**: INT8 PTQ is lossless vs float on device-eval (gap ≤0.7 pp). (G3)
7. v8 is best in **every** stratified mode; worst modes for all versions: `music_snr5`,
   `reverb+music10`, `muffled_lp3k`. These are the targets for the real-data iteration. (G7/G9)

## Genuine shippable enhancement (complements v8, no regression)
**2nd-stage verifier** (`research/G2_VERIFIER.md`, `verifier/`): a 65-param logreg on the
64-d time-pooled KWS bottleneck, run only after a trigger. On held-out data it keeps 100%
of true wakes and rejects 100% of mined no-VAD music false-accepts (0.79 margin) for ~260 B
and ~0 latency. Deployable as a dual-head tflite; most valuable for no-VAD/low-latency setups.

## Forward path (for the real device long-tail data)
- Use the **production loop** (`research/PRODUCTION_LOOP.md`): log miss embeddings → review →
  hard-neg mine → retrain, gated on this same device-eval + `strat_eval`.
- Target the known weak modes (`music_snr5`, `reverb+music10`, `muffled`) with **real**
  captures of those conditions (real-tract data closed the gap v5→v8; it will again).
- Re-try far-field with **milder, real** RIRs (or real far-field captures) — SLR28 was too
  reverberant; mit_rirs-level intensity is the ceiling for synthetic.
- If revisiting loss: implement a **true** angular-margin head, not the logit-shift approx.

## Deliverables in output_v9/
- `research/` — G2 verifier, G3 QAT, G7 strat-eval (+ `strat_eval.py`), G8 arch sweep,
  G9 version history, G10 production loop, G11 research notes; reusable scripts
  (`strat_eval.py`, `extract_embeddings.py`, `train_verifier.py`, `quant_gap.py`,
  `evaluate_device.py`).
- `verifier/` — `logreg64.npz` (deployable weights) + metrics.
- **Production model stays `output_v8/`** (unbeaten). Manifest there already valid
  (`website=https://github.com/vvzvlad/zakhar-voice-assistant`, `author=vvzvlad`).
