# v16 — SINGLE-MODEL SHIP (= output_best). v8 recipe + REAL device positives + 4 real neg classes.
Device-eval @0.9/win5: FRR 21.0% [17.1–25.5]; FAPH (VAD) silence/music/speech/vacuum all 0; no-VAD
silence 0 / music 5.8 / speech 0 / vacuum 0. Strictly better than v8 (12.5/h silence FAPH at the same
~21% recall). Keep v0.3 quiet positives (ablation: removing them → FRR 25.7%). INT8 32k, drop-in.
Recall floor (21%) is fundamental this round (duration eval: onset-proxy); short-«захар» negs next round.
FIELD operating point: probability_cutoff **0.95** — at 0.80 v16 false-fired in real silence (heard the word
in a quiet room); v16's DET is ~flat so 0.95 holds recall ~21% while removing the silence false-fires.
Dense-speech (SOVA radio) FAPH 1.3/h @0.9 — robust (#F); angular alternatives are catastrophic there (21.5/h).
