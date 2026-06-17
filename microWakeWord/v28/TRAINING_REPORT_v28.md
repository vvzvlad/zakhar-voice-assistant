# zakhar microWakeWord — angmax (MAX-RECALL candidate, 2026-06-17)

**Goal: maximum recall.** A server-side gate-wakeword verifies «захар» downstream, so on-device false
accepts are cheap (≈ a server call). The on-device model's job is to MISS as little «захар» as possible —
short or drawn, any voice, any condition. angmax is built for that.

## What it is
Angular (AM-Softmax / CosFace, s=10, m=0.20, margin-annealed) mixednet, INT8 streaming, micro_speech 40-band,
feature_step 10, ~32k params, **78992 B** (≤45 KB tensor arena, same budget as v16/v19/v27). Manifest:
cutoff **0.90 / sliding_window 5** (same firmware config). website/author = vvzvlad.
Recipe: v19 angular base + **all diverse positives** — fish device-tract drawn (5729 voices) + **fish short
«захар» as POSITIVE** (short/drawn distinction dropped) + section real-device + v8 (piper/silero/device).
Train: `training_parameters_angmax.yaml` + `nr2/train_angmax.sh` (am_train.py, 30k steps).

## Recall (cut 0.9, Wilson 95% CI) vs the shipped v27 and the prior angular v19
| channel | v27 (prod) | v19 | **angmax** |
|---------|-----------:|----:|-----------:|
| fish device-tract (n=5664, 3173 voices) | 0.673 | 0.957 | **0.9952** [.993,.997] |
| off-channel, clean held-out (n=91/16 voices) | ~0.80 | 0.978 | 0.934 (tie w/ v19, p=0.22) |
| clean-16 (n=43) | 0.674 | 0.953 | 0.884 |
- **+32 pp recall over v27 on the device tract; beats v27 significantly on every channel.**
- Beats v19 on the device tract (0.995 vs 0.957, p=1e-58); ties v19 off-channel on clean data.

## Not "fires on everything" (red-teamed)
Fires on **99.3 %** of «захар» but only **6.6 %** of generic non-«захар» speech (TV+radio, 14 h) →
«захар»-vs-speech AUC **0.995**. Fires on speech *less* than v19 (6.6 % vs 14.8 %). A genuine discriminator,
not saturation. Leakage-free (bench voices voice-disjoint from train, MFCC-independent).

## Server-call-rate (on-device fire → server gate)
Post-VAD (silence/music gated), worst-case continuous TV/radio: **~8.7 calls/h** (2.4× lower than v19's 20.6).
Real homes (intermittent speech) well below this. "Units/hour", within budget. Raw FAPH (no VAD), for
reference: silence 27/h, music 58/h, speech 7.8/h, radio 5.3–8.3/h — VAD removes the non-speech bulk.

## Operating point
- Default **0.90 / win 5** (validated above). Recall is already ~0.995 on-channel here.
- For even higher recall (and more server calls), lower the cutoff (e.g. 0.80) — server gate absorbs the extra
  false accepts. 0.90 is the recommended starting point.

## Honest limits / before shipping
1. The 0.995 is on the **fish recapture device tract**. Off a different tract recall is ~0.93 (still strong).
   **Confirm the production device == the fish recapture tract** for the headline number; otherwise expect ~0.93.
2. Clean off-channel evidence is **n=16 voices** (small) — directionally a tie with v19, but more clean
   off-channel recordings would firm it up.
3. The on-device 2-stage verifier is NOT used here (the gate is server-side, per the new design) — good.

## Verdict
Under the max-recall goal, **angmax is the recommended on-device model**: +32 pp recall over the shipped v27,
genuine «захар» discrimination, tolerable server load, same size/firmware config. v19 is the channel-robust
fallback if broader-tract coverage is later prioritized. v27 remains prod until you swap.
