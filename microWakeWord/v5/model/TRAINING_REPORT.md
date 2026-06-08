# TRAINING REPORT v5 — «захааар» (drawn-out-centered + adversarial mining + duration lever)

Goal (revised by operator): the wake word is the **drawn-out «захааар»** (long «а»);
the short «захар» is explicitly NOT a target. Primary metric = recall on the
drawn-out form (the 93 real held-out are drawn-out). Use **duration** as a
discriminator. Ship to output_v5/ only if it beats v4 on (drawn-out recall ↔ clean
music-FAPH, with VAD and without). v1/v4 frozen and untouched.

## VERDICT

**v5 is the new best on the primary metric and ships.** With the VAD gate (the
recommended deployment) it **strictly dominates v4**; on the drawn-out recall axis it
beats v4 everywhere; and it cut the «Сахар» near-miss. One honest caveat without VAD
(below).

| metric | v4 | **v5** |
|--------|----|--------|
| Drawn-out held-out recall (per-clip, %≥0.9) | 99% (mean 0.996) | **100% (mean 0.999)** |
| «Сахар» near-miss (speech-neg fires ≥0.9) | 3/90 | **1/90** |
| Clean music-FAPH **with VAD** (all cutoffs) | 0.00 | **0.00** |
| Recall @ cutoff 0.9/win3 (clean-music eval) | 0.882 | **0.946** |
| Clean music-FAPH **without VAD** @ cutoff 0.9/win3 | 2.67 | 2.67 |

- **With VAD (deployment):** music-FAPH is 0 for both → recall decides, and v5 is
  higher at every cutoff (e.g., 0.946 vs 0.882 @0.9; max 0.989 vs 0.968). Strict win.
- **Without VAD:** v5 wins the realistic high-recall region (music-FAPH 2.0–2.67 →
  recall 0.91–0.95 vs v4 0.88). v4 only "wins" an extreme low-FAPH corner
  (FAPH 1.33 at recall ~0.80) that is both lower-recall and moot under VAD. So without
  VAD it is a favourable trade, not strict domination across the whole curve.

## What moved the frontier (with sources)

1. **Adversarial false-trigger mining** (Amazon arXiv 2010.06659; MS ICASSP'20 hard-neg
   mining; HA "falsekw honeypot"). Ran v4 over audio with no «Захар» and harvested every
   high-probability window as a hard-negative:
   - **1313** from **vocal music** (fma_small) — the residual VAD passes.
   - **940** from a с/з-rich synthetic RU speech corpus — the «Сахар» family.
   Folded in at sampling_weight 6 / **penalty_weight 8**.
2. **Drawn-out centering of positives** (operator's reframe). Positives = 528 real
   (drawn-out reference) + **1056 time-stretched reals** (sox tempo 0.7/0.55 = ×1.4–1.8
   longer, pitch preserved) + **7247 elongated synthetic** (Piper+Silero, «захааар/
   захаааар/захаааааар», high length_scale). Short «захар» removed. This is what pushed
   drawn-out recall to 100% and shrank «Сахар» (Сахар is short → loses on duration).
3. **Duration / window lever.** Larger sliding windows require sustained probability;
   the drawn-out word sustains, short false (Сахар, music bursts) do not. clip_duration
   1800 ms. (Window sweep in the eval grid; win3 is the recall-optimal point.)
4. **VAD pre-gate** (carried from v4): instrumental music → 0 false-accepts.

Other research tried/considered: focal loss & AM-softmax/large-margin (class-imbalance /
separation — candidate for v6 if needed), knowledge distillation (~20% miss-rate
reduction, Amazon — Priority D), second-stage verifier (arXiv 2304.03416 — Priority D).

## Chosen operating point (v5)

`probability_cutoff = 0.95`, `sliding_window_size = 3`: drawn-out recall 0.914
(windowed) / 100% per-clip, clean-music-FAPH **0.0 with VAD**, 2.0 without; «Сахар»
1/90. **VAD strongly recommended** (default in the snippet). Manifest v2,
`tensor_arena_size 45000` (generous; verify against the live JSON and lower if desired).

## Next (in progress this session): round-2 mining → v6

Mining v5's OWN residual (v5 over fma_small music + 16,095 Common Voice ru real-speech
clips) → fresh hard-negs → v6, aiming to also close the no-VAD low-FAPH corner. v6 ships
only if it strictly beats v5/v4.

## Files
`zakhar.tflite` (INT8 streaming, 77.8 KB, 5-block 32,321 params), `zakhar.json` (v2),
`esphome_micro_wake_word.yaml` (VAD on), `eval_v5_cleanmusic.json`, and the v5 configs/
scripts (training_parameters_v5.yaml, train/convert, mine_false.py, gen_drawn_*.py,
gen_ru_corpus.py). Run with PYTHONPATH=~/zakhar-mww/micro-wake-word; venvs as in v4.
