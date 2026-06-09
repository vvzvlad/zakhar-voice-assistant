# G9 — Leakage-safe re-eval of all versions (v1 / v4 / v5 / v8)

All numbers on the SAME leakage-safe device-eval (153 held-out device positives,
uuid/source-disjoint) + the stratified programmatic eval (those 153 positives
degraded into 9 acoustic modes). One honest scale across the whole history.
v1=output/, v4=output_v4/, v5=output_v5/, v8=output_v8/ (all frozen).

## Device-eval (real tract), cutoff 0.9, window 3

| version | device FRR | recall | device-FAPH (VAD) |
|--------:|-----------:|-------:|------------------:|
| v1 | 0.346 | 0.654 | 0.0 |
| v4 | 0.163 | 0.837 | 0.0 |
| v5 | 0.183 | 0.817 | 0.0 |
| v8 | **0.059** | **0.941** | 0.0 |

→ v5 was a **slight regression** vs v4 on real device audio (0.183 vs 0.163);
v8 fixed contamination + added real device captures and roughly **tripled** the
margin (FRR 5.9%). FAPH with VAD is 0 for all.

## Stratified FRR @0.9 by acoustic mode (worst-case profiling)

| mode | v1 | v4 | v5 | v8 |
|------|----:|----:|----:|----:|
| clean | 0.346 | 0.163 | 0.183 | **0.059** |
| reverb | 0.431 | 0.183 | 0.235 | **0.078** |
| music_snr10 | 0.516 | 0.294 | 0.366 | **0.111** |
| music_snr5 | 0.641 | 0.444 | 0.490 | **0.216** |
| babble_snr10 | 0.484 | 0.294 | 0.327 | **0.124** |
| quiet_-18dB | 0.307 | 0.170 | 0.183 | **0.059** |
| muffled_lp3k | 0.536 | 0.261 | 0.307 | **0.157** |
| lombard_+6dB | 0.399 | 0.150 | 0.203 | **0.065** |
| reverb+music10 | 0.614 | 0.340 | 0.366 | **0.176** |
| **mean adverse (excl clean)** | **0.491** | **0.267** | **0.310** | **0.123** |

→ v8 is best in **every** mode. The hardest mode for every version is
`music_snr5` (loud music overlap, SNR 5 dB) and `reverb+music10`. This is the
target for v9 (multi-condition + far-field training) and the 2nd-stage verifier.

## Takeaways

1. v8 is a strong incumbent; any v9 must beat **device FRR 0.059** (and not
   regress FAPH) to ship — and ideally cut the `music_snr5`/`reverb+music10`
   worst-case (v8: 0.216 / 0.176).
2. The full history on one honest scale: v1 → v4 big gain → v5 minor regression →
   v8 large gain. (The old per-version reports each used their own split; this is
   the unified, leakage-safe view.)

Data: `v9/device_eval_all.json`, `v9/strat_{v1,v4,v5,v8}.json`, harness
`v9/strat_eval.py` + `v8/evaluate_device.py`.
