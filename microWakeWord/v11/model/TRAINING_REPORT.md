# v11 — v8 recipe + REAL device-tract silence+music hard-negatives

**Ships: v11 fixes the field bug.** Adding REAL device idle/silence (166 min) + music (38 min)
hard-negatives (no augmentation, kept real texture; sampling 10/8, penalty 8) to the v8 recipe.

## Real held-out eval (no-VAD), vs v8 baseline
| metric | v8 | **v11** |
|--------|---:|--------:|
| silence-FAPH @0.9/win5 | 12.5/h | **0.8/h** (−94%) |
| music-FAPH @0.9/win5 | 9.3/h | **0.0/h** |
| FRR real_people(107) @0.9 | 4.7% [2.0–10.5] | **0.9% [0.2–5.1]** |
| FRR dev_heldout(153) @0.9 | 8.5% [5.0–14.0] | 12.4% [8.1–18.6] |
With Silero VAD both are ~0 FAPH; v11's win is being robust **without** VAD.

**Verdict:** v11 eliminates the real silence/music false-fires that synthetic data never
reproduced, and IMPROVES FRR on real human voices (the representative future users). The only
cost is a modest FRR uptick on the old dev_heldout device captures (overlapping CIs). Net:
strong win on the operator's primary pain (false-fires) → ship at cutoff 0.9 / window 5.
Follow-up to recover dev_heldout FRR: add the new yandex/real positives to training (v11b).

## Recipe
v8 exactly + two real-negative mmap sets (`features_v11_sil`, `features_v11_mus`), built no-aug
from `samples_real/negative_{silence,music}` (leakage-safe 70/30 split, held-out 30% = the eval).
