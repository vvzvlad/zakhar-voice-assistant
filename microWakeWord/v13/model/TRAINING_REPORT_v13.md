# v13 — angular head (v12) + 2nd-stage verifier on REAL negatives

**Best metrics of the program (2-stage deployment).** v12 = TRUE angular head (s=10,m=0.2) +
real silence/music negatives → spectacular recall (FRR dev 2.0%, real_people 0.0%) but the
angular head is too confident, so real negs did NOT fix its FAPH (silence 12.5/h, music 18.5/h).
v13 adds a 65-param logreg verifier on the angular bottleneck, trained on v12's REAL false-fires.

## Real held-out eval
| | silence-FAPH @0.9/win5 | music-FAPH | FRR dev | FRR real_people |
|--|----------------------:|-----------:|--------:|----------------:|
| v8 | 12.5/h | 9.3/h | 8.5% | 4.7% |
| v11 (single model) | 0.8/h | 0/h | 12.4% | 0.9% |
| v12 (angular, no verifier) | 12.5/h | 18.5/h | **2.0%** | **0.0%** |
| **v13 (angular + verifier @0.5)** | **~0/h** | **~0/h** | **~5%** | **~3%** |

Verifier @thr 0.5: rejects **100% of held-out silence FAs (13) + 100% music FAs (2)** while
keeping **96.9% of true positives**. Combined FRR = 1−(1−FRR_v12)·keepTP ≈ 5% (dev) / 3% (real).

**Verdict:** v13 has the best FRR↔FAPH frontier — 0 silence/music false-fires AND lower dev-FRR
than v8/v11. **Caveats:** (1) 2-stage deployment (tflite + post-trigger verifier) needs ESPHome
dual-head integration (see output_v9/research/G2_VERIFIER.md); (2) held-out FA eval N is small
(13 silence, 2 music) — the 100% rejection is indicative; validate on more captures. v8 stays the
live single-model default; v11 is the recommended single-model upgrade; v13 is the best if the
2-stage path is taken. Artifacts: zakhar.tflite (angular) + verifier_logreg.npz + verifier.json.

## UPDATE: deployable (baked single-output) verifier does not beat v11
Baked the verifier into one tflite (output = P(wake)·P(verify), no firmware change; converts to
81 KB int8 with a Multiply op). Real-eval streaming/int8: silence-FAPH 6.7/h, music 18.5/h (vs
v11's 0.8 / 0). The verifier (≤6 real music FAs) doesn't generalise to music in streaming/int8.
**Recommendation downgraded: ship v11 (single model) — it fixes the bug better at equal real-user
FRR. The baked-verifier path (v13/build_baked.py, mixednet `--angular_head 2`) is validated and
ready to revisit once more real music negatives exist.**
