# G11 — Self-research notes (tiny streaming KWS, 2023–2025) + how it maps to our backlog

Sources via tavily; each finding tagged with the goal it informs and whether we TRY it.

## Loss functions — biggest single lever found
- **AM-Softmax / additive-margin** (arXiv 2211.00439, 2022, Table 2): on a streaming KWS task,
  FRR@FAR=10% dropped **15.81% → 8.08%** vs plain softmax (~49% relative); Angular-Prototypical
  even better (7.77%). AAM-Softmax also validated in MTCANet (Springer 2023), weighted with BCE.
  → **G4: PRIORITIZE.** Add an angular-margin head (margin 0.35–0.4, scale s≈30) on the bottleneck
  embedding, weighted with the existing BCE. Expected to be a top contributor.
- Noise-robust loss + embedding centroid training (AAU, IEEE TASLP 2021): improves unseen-noise FRR.
  → G4 secondary: t-SNE shows margin losses tighten intra-class / widen inter-class — exactly what
  helps the «Сахар»/confusable separation.

## Robustness to noise/music/far-field (v8's worst modes per G7)
- **Multi-style training + AGC** (Google, Prabhavalkar et al.): up to **75% rel** improvement in
  far-field by multi-condition training. → confirms G5/G6 are high-value.
- **Curriculum multi-condition** (MISP 2021): progressively harder (distance/noise) **beats vanilla**
  multi-condition. → **G5/G6: try curriculum ordering** (clean → reverb → reverb+media, increasing
  over training steps) rather than uniform-random aug.
- Multichannel/ANC/beamforming (Interspeech 2025) lowers FRR more than doubling model size — but
  needs >1 mic; HA Voice PE is effectively single-stream to mWW → not applicable on-device.

## Architectures (G8)
- **BC-ResNet** (Qualcomm, broadcasted residual learning): 1D temporal conv + broadcasted residual
  expanding temporal→freq-temporal; scales to device budget; strong accuracy/size frontier.
  → G8: port a sub-100k BC-ResNet-style block, compare vs mixednet at same arena on device-eval.
- **LiCo-Net** (linearized conv, depthwise-separable + linear activations, sub-MB): efficient encoder
  used as KD student in APSIPA 2025 zero-shot work. → G8 alt encoder idea.
- res15 (238k) classic ref; KWT transformer (5.3M) too big for MCU.

## Distillation (G1)
- Survey "Advances in Small-Footprint KWS" (arXiv 2506.11169, Jun 2025): KD + NAS + QAT + pruning are
  the canonical on-device levers. Representation-level KD (intermediate features) > logit-only KD in
  APSIPA 2025. → G1: besides logit/soft-target KD, consider an intermediate-feature distill term if
  logit-KD underdelivers.
- Wake-word KD historically needs huge positive corpora (>400k–2.5M utterances) for big gains
  (Ghosh 2022, Amazon). We have ~10k synth + ~900 real/device → KD gains may be modest; manage
  expectations, lean on multi-condition + AM-softmax as primary movers.

## Two-stage / second-stage verification (G2)
- Monophone-based background modeling for **two-stage on-device wake word** (Amazon, ICASSP 2018):
  validated production pattern — cheap first stage triggers, second stage confirms. → confirms G2.
- Negative-sample mining (Hou et al., ICASSP 2020 "Mining effective negative training samples"):
  → confirms G10 hard-neg mining loop; mine from embeddings of false-wakes.

## Net takeaways for our run
1. **AM-softmax (G4)** = likely the highest-ROI model-side change → do early after the multi-cond retrain.
2. **Curriculum multi-condition (G5/G6)** > uniform aug → worth a variant.
3. KD (G1) gains may be modest at our data scale → still run it (teacher already training) but don't
   bet the ship on it; primary movers are multi-cond + AM-softmax.
4. BC-ResNet (G8) is the most promising alt arch to A/B at equal arena.

## ADDENDUM — why synthetic music-mix failed, and what to capture (forward)

Second research pass on the #1 weak mode (`music_snr5`, media overlap):
- **Playback-interference augmentation** (Amazon, arXiv 1808.00563): mixing music + TV at
  various SIRs gives 30–45% rel FA reduction — BUT at realistic SIRs matching the device's
  **post-AEC residual echo**, not full-volume mixes.
- **Device-playback augmentation + AEC sim** (Interspeech 2021, Opatka): improvement is
  consistent **only when ALL device-path factors are simulated** (loudspeaker transfer +
  nonlinearity + AEC residual). Plain music-mixing alone underperforms.
- **Key implication for us:** HA Voice PE runs AEC with the media **loopback reference**, so
  on-device the KWS sees only the *residual* echo after cancellation — NOT full-volume music.
  Our v9 synthetic `music_snr5` mixing (full-volume held-out music onto clean positives) is a
  **harder, wrong distribution** → training on it hurt, and the eval over-states the problem.
  This is the concrete reason the synthetic far-field/media levers regressed.
- **What to capture tomorrow (correct signal):** wake word spoken **while media plays through
  the device's own speaker** (so it passes the real AEC) — that residual-echo distribution is
  what the model must learn, and it can't be faithfully synthesised without the device path.
  Also capture far-field + muffled in the real room. Add these to the device positive set
  (weight 12) per the runbook.
