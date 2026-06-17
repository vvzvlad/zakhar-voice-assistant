# zakhar v28 (angmax) — DEPLOYABLE package (2026-06-17, fixed)

Max-recall on-device model for ESPHome `micro_wake_word` (server-side gate verifies «захар» downstream).

## ⚠️ Fix vs the first (crashing) export
The first angmax.tflite contained DEQUANTIZE/RSQRT/SQUARE/SUM (from the angular head's L2-normalization),
which the on-device MicroMutableOpResolver<20> lacks → boot-loop. **This version folds the L2-normalization
into a constant scale** (x/‖x‖ → x/c, c=31.03 = mean embedding norm), turning the head into a plain
`FullyConnected + Logistic` — **op-set now identical to v27 (verified: no DEQUANTIZE, all ops in the resolver).**

## Files — ~/zakhar-mww/angmax_deploy/
```
~/zakhar-mww/angmax_deploy/zakhar.tflite   (md5 7dc421ea13d854e3539d1319a066c718)
~/zakhar-mww/angmax_deploy/zakhar.json
```

## Deploy config (in zakhar.json)
| field | value |
|-------|-------|
| **probability_cutoff** | **0.80** |
| sliding_window_size | 5 |
| feature_step_size | 10 (micro_speech 40-band) |
| tensor_arena_size | 45000 |
| **model size** | **77728 bytes** |
| minimum_esphome_version | 2024.7.0 |
| op-set | CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, QUANTIZE, RESHAPE, STRIDED_SLICE, CONCATENATION, SPLIT_V, VAR_HANDLE/READ/ASSIGN_VARIABLE, CALL_ONCE — ⊆ resolver ✓ (same as v27) |

## Cutoff: the fold shifted the probability scale → use **0.80** (lower than the original 0.92)
Recall vs radio server-calls (fish bench, radio_eval 3.0 h, fresh-state):
| cutoff | recall (fish) | radio calls/h |
|-------:|--------------:|--------------:|
| 0.78 | 0.9931 | 5.30 |
| **0.80 (recommended)** | **0.9910** | **5.30** |
| 0.85 | 0.9838 | 3.64 |
| 0.90 | 0.9725 | 2.98 |
- 0.80 = recall-max at your ~5/h radio target. Lower → more recall + calls; higher → fewer calls.

## Recall (cut 0.80) vs original angular angmax and shipped v27
| channel | v27 | angmax (un-deployable) | **this deployable** |
|---------|----:|-----------------------:|--------------------:|
| fish device tract (n=5664) | 0.673 | 0.995 | **0.991** |
| off-channel clean (n=91/16 voices) | ~0.80 | 0.93 | 0.846 |
| clean-16 (n=43) | 0.674 | 0.884 | 0.721 |
- **On the production (fish) tract the fold costs almost nothing: 0.991 vs 0.995.** It loses some OFF-channel
  recall (the per-sample L2-norm mattered most there) but still beats v27 on every channel.
- Genuine «захар» discriminator (not saturation), leakage-free — RED-TEAM confirmed on the angular source; the
  fold only changes the head normalization, backbone unchanged.

## Note
0.991 is on the fish recapture device tract; a different tract gives ~0.85 (still > v27's 0.67). Firmware config
unchanged from v16/v19/v27 — only .tflite + .json change.

**This is FINAL.** A BCE fine-tune of the deployable head was tried to claw back off-channel recall — it gave NO
improvement (fish 0.9905, off-channel 0.846, identical). The off-channel 0.846 is the deployable ceiling: the
angular model's off-channel edge came from the per-sample L2-norm, which needs RSQRT (not in the resolver) and
cannot be reproduced by a fold or a standard head. To get the full 0.93 off-channel you would need the un-
deployable angular tflite + adding DEQUANTIZE/RSQRT to a forked on-device resolver (firmware change). For the
stated goal (max recall on the production tract, server gate downstream) this deployable model — 0.991 on-tract,
≫ v27 everywhere — is the recommendation.
