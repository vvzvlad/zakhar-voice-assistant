# output_v10 — TRUE angular-margin head (AM-Softmax/CosFace) — investigation

**No model ships from v10 — v8 remains production.** The TRUE angular head is the strongest
RECALL lever found (worst-case device-FRR −13 to −16pp, INT8 quant-lossless) but at every scale
it raises **music false-accepts** (the live field bug); mined music hard-negs halve it (18→9/h)
but synthetic fma negs don't reach v8's 0/h. Fails the strict gate (≥3pp FRR better WITHOUT
FAPH rise) → not shipped. See TRAINING_REPORT_v10.md for the full 8-config sweep + verdict.

## What's here
- `TRAINING_REPORT_v10.md` — full results, all configs, the verdict, and the 2 forward paths.
- `research/` — `RESEARCH_angular.md` (cosine-margin choice + INT8 risk), `am_train.py`
  (the true AM-Softmax trainer: L2-norm embedding via UnitNormalization, weight projected to
  norm s, additive cosine margin in the loss), `gate_am.sh` (convert+quant-gap+device+strat+FAPH),
  `mine_hardneg.py`/`log_misses.py` (production-loop prototype, hard:random 1:2 anti-collapse),
  `silence_eval*.py` (the silence/idle + music FAPH gate).
- `candidates/` — the two best angular tflites for reference (NOT for deployment; music-FAPH
  regresses): `angular_s10_hardneg.tflite` (best FAPH 9/h, worst-FRR 0.078),
  `angular_s08_hardneg.tflite` (best worst-FRR 0.059, FAPH 19/h).

## mixednet change (flag-guarded, v8 untouched)
`--angular_head 1` inserts `Flatten → UnitNormalization(L2) → Dense(1,no-bias) → sigmoid`.
(Lambda(l2_normalize) breaks the mWW streaming clone_model; `UnitNormalization` works and
int8-converts cleanly.) Default 0 → v1/v4/v5/v8 reproduce exactly.

## Forward (top model-side lever for the real data)
Re-run the angular head (s≈10, m=0.2) with **real device music negatives** (incoming captures),
and/or pair it with the **2nd-stage verifier** (output_v9/verifier) to cut the residual music
FAPH. The recall gain is large and real — it just needs real music negatives to be deployable.
