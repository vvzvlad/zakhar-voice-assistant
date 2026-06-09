# G3 — QAT vs PTQ

**Question:** does quantization-aware training (QAT) recover device accuracy lost to
INT8 post-training quantization (PTQ)? Literature suggests +4–6% rel on some KWS models.

**Method:** isolate the quantization effect by comparing the SAME v8 student on the SAME
streaming path — a **float streaming tflite** vs the shipped **INT8 streaming tflite** —
on the leakage-safe device-eval (153 held-out positives) + device-noise FAPH. (A first
attempt compared float *non-streaming* vs int8 *streaming* and showed a huge "gap", but
that was a front-padding artifact of the non-streaming path, not quantization — discarded.)

## Result — PTQ is effectively lossless

| cutoff | float FRR | int8 FRR | gap (int8−float) |
|-------:|----------:|---------:|-----------------:|
| 0.70 | 0.046 | 0.046 | +0.000 |
| 0.80 | 0.052 | 0.046 | −0.007 |
| 0.90 | 0.059 | 0.059 | +0.000 |
| 0.95 | 0.078 | 0.085 | +0.007 |

- mean positive prob: **float 0.962 = int8 0.962** (identical).
- device-noise FAPH@0.9 (no-VAD): float 10.0 = int8 10.0 (identical).

## Verdict: QAT NOT warranted for this model

The INT8 PTQ model is statistically indistinguishable from the float model on device-eval
(|gap| ≤ 0.7 pp, within noise of N=153; mean prob identical). There is **no quantization
loss to recover**, so QAT's expected +4–6% cannot materialize here. Why: the 40-band
micro_speech spectrogram is already INT8/uint16, the mixednet weights are small and
well-conditioned, and the mWW per-channel INT8 converter captures them faithfully.

**Decision:** keep PTQ (the existing convert path). Spend the compute budget on the levers
that actually move device-FRR — multi-condition data (G5/G6) and loss (G4) — not QAT.
If a future larger/wider arch (G8) shows a real float→int8 gap, revisit QAT then; the
measurement script (`v9/quant_gap.py` + the streaming float-vs-int8 compare) is the gate.
