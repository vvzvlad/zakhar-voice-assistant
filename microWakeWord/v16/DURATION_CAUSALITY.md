# Duration-causality eval — what does «захааар» detection actually key on?

Question (operator/RESEARCH_6): does the model distinguish «захааар» from «захар» by VOWEL
DURATION, or by a proxy (loudness / spectral fingerprint / base-rate)? If proxy → field recall
is fragile (exactly the ~21% FRR we see on real held-out positives). No KWS-literature precedent.
Tested on the best-recall model (v12 angular, 9.7% FRR) and v8 (`nr2/duration_causality*.py`).

## Findings
1. **Energy-invariant — NOT a loudness proxy.** Score vs GAIN (duration fixed, RMS then ×0.25…×4):
   v12 0.977→0.968, v8 0.832→0.842 — **flat across 16× loudness**. The model does not key on energy.
2. **Time-axis saliency peaks MID-CLIP on the «заха»/vowel-ONSET, not the sustained vowel.**
   Occlusion (0.25 s mask) score-drop by clip decile: peak at **decile 4** (~40–50%, the «х→аа»
   onset) for BOTH v8 and v12; the drawn-out tail (deciles 6–9) has near-zero saliency.
3. **The drawn-out tail contributes ~nothing.** Real «захааар» positives are ALL long (≥4.6 s,
   median 5.3 s). Full clip vs its first-4.6 s crop: v12 0.841 vs 0.874, v8 0.599 vs 0.613 — the
   crop is *slightly higher*, i.e. the sustained vowel beyond ~4.6 s is NOT used (if anything the
   tail dilutes the windowed score).
4. (Whole-clip stretch score *declined* with length, but that's the receptive-window confound —
   stretching a 5 s clip to 9 s pushes it past the ~4.9 s window; discarded as a duration signal.)

## VERDICT: the model relies on the spectral ONSET pattern («заха»), NOT on vowel DURATION.
It does not integrate the drawn-out vowel as the discriminative cue — so the «захааар»-vs-«захар»
distinction is effectively NOT being learned by duration; detection rides on onset-spectral match.
That match is voice/room-specific → **fragile recall in the field** (the ~21% held-out FRR, and
worse off-session). This is the proxy-reliance the operator hypothesised, confirmed empirically.

## Implications for the program
- More real device positives (#1, v16) will improve the onset-spectral coverage → better recall,
  but won't by itself make the model *duration-aware*.
- To actually use duration, the model needs a **duration-discriminative signal in training**:
  explicit SHORT «захар» hard-negatives paired with drawn-out positives (forces the boundary onto
  vowel length), and/or a longer/explicit temporal-integration feature. Short-«захар» negs aren't
  in this round's data (recording pending) → flag for next round; meanwhile #1 helps recall but
  the duration cue remains unaddressed.
- The angular head (v12) has the same saliency profile — its recall gain is better onset coverage,
  not duration awareness.

## Operating-point sweep (supporting) — angular FAPH is cutoff-immune
v17 angular: even at cutoff 0.99 silence-FAPH stays 12.5/h (FRR 16%); v8: 9.2/h @0.99 (FRR 33%).
The silence false-fires are HIGH-CONFIDENCE (AGC-loud, saturated outputs) → raising the cutoff
kills recall before it dents FAPH. Only MODEL-LEVEL real-negative training (v11/v16 → silence 0)
suppresses them. → confirms FAPH must be fixed in training (v19 heavy-penalty), not by threshold.
