# zakhar microWakeWord — v27 (RECOMMENDED SHIP)

**v27 = v16 recipe + SYNTHETIC short-«захар» negatives (light weight 4/5).**
Single INT8 streaming model, micro_speech 40-band, feature_step 10, ~32k params, 77848 B (≤45KB arena).
Manifest: cutoff 0.9 / sliding_window 5 (same firmware config as v16), website/author = vvzvlad.

## Why v27 over v16
v16 fires on ~65% of SHORT «захар» utterances (the production false-trigger complaint). v27 adds
synthetic short-«захар» negatives — TRAIN positives whose vowel is time-compressed (disjoint from eval) —
at a light sampling/penalty weight (4/5) to instil duration-awareness without over-penalising.

| metric | v16 (old ship) | **v27 (this)** |
|--------|---------------:|---------------:|
| FRR, drawn-out held-out (recall) @0.90 | 21.0% | **19.3%** [15.6–23.7] |
| short-«захар» firing @1.0s | **0.65** | **0.233** |
| silence-FAPH | 0 | 0.8/h |
| music-FAPH | 5.8/h | **2.9/h** |
| speech-FAPH | 0 | 0 |
| radio dense-speech FAPH | 1.3/h | 2.3/h |
| size | 77848 B | 77848 B |

v27 fixes the short-«захар» bug (64% fewer false triggers) while slightly *improving* drawn-out recall and
staying robust on every FAPH class. On the full DET it sits on the v16 frontier and strictly dominates it in
the high-recall regime (FRR < 17%). See `SUMMARY_real_eval.md` §#I and §#K.

## Operating point
- **Default: cutoff 0.90 / win 5** — FRR 19.3%, FAPH ~1.5/h (no-VAD) / ~1.2/h (VAD).
- Lower false-accepts: **0.95** → FRR 22.7%, FAPH 0.86/h (no-VAD) / 0.69/h (VAD).
- Max recall: **0.85** → FRR 16.3%, FAPH 1.9/h.
- **Enable device VAD** — halves FAPH at no recall cost; only dense-speech (radio/TV) remains (un-suppressible).

## Notes
- The synthetic-short-neg recipe is tuned to swap synthetic→REAL short-«захар» recordings when available
  (expected to push short-firing lower at the same recall).
- RepCNN reparametrization (#J) was validated (exact fold, zero runtime cost) but gave no recall gain — not
  used here. Radio-in-training (v25) was Pareto-dominated — rejected.
- ESPHome snippet identical to output_v16/ (same cutoff/win); only the .tflite + .json change.
