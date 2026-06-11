# v16 — SINGLE-MODEL SHIP (= output_best). v8 recipe + REAL device positives + 4 real neg classes.
Device-eval @0.9/win5: FRR 21.0% [17.1–25.5]; FAPH (VAD) silence/music/speech/vacuum all 0; no-VAD
silence 0 / music 5.8 / speech 0 / vacuum 0. Strictly better than v8 (12.5/h silence FAPH at the same
~21% recall). Keep v0.3 quiet positives (ablation: removing them → FRR 25.7%). INT8 32k, drop-in.
Recall floor (21%) is fundamental this round (duration eval: onset-proxy); short-«захар» negs next round.
