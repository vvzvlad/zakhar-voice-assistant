# Real-data iteration runbook (run when the device long-tail captures land)

The v9 program found that **synthetic** model-side levers don't beat v8 — the gap closes
with **real device-tract data** (it's what took v5→v8 from 18%→6% device-FRR). This runbook
turns new captures into the next gated candidate using the **proven v8 recipe** + the v9
gate harness. Target the known weak modes: `music_snr5`, `reverb+music10`, `muffled_lp3k`.

## 0. Inputs expected
- New **positive** captures (someone saying «Захар» through the device), ideally including
  the weak conditions: with music/TV playing, far-field, quiet/muffled. → `incoming/pos/`
- New **negative** streams (device room noise, music, TV, conversations; NO wake word).
  → `incoming/neg/`
- Keep the original filenames/uuids so the leakage-safe split stays honest.

## 1. Leakage-safe split (NEVER let eval sources leak into train)
Reuse `v8/process_device.py` (uuid/source-grouped split). Add the new captures to its
source pools; hold out the same uuid family used by `v8/dev_heldout_pos/` so device-eval
stays comparable across versions. Output: train positives, held-out positives, neg train,
held-out neg stream.

## 2. Build features (PROVEN v8 augmentation — do NOT use SLR28/aggressive aug)
Use `build_features_v4.py` settings (mit_rirs, RIR 0.6, AddBackgroundNoise 0.75,
SNR −10..12). v9 showed SLR28 + heavier aug regress. Build:
- new real positives → `features_vNEXT_device` (sampling_weight 12, the gold real-tract set)
- keep v8's synth + real positive sets
- new reviewed negatives → a hard-neg set (penalty_weight 6–8), like `features_v5_mined`.

## 3. Train (v8 arch exactly — NO residuals; v9 proved residuals hurt: 0.065→0.137)
```
mixednet --pointwise_filters 64,64,64,64,64 --repeat_in_block 1,1,1,1,1 \
  --mixconv_kernel_sizes [5],[7,11],[9,15],[17,23],[29] --residual_connection 0,0,0,0,0 \
  --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3
```
Keep the ambient eval sets (sampling_weight 0) IN the yaml so `best_weights` is selected by
FAPH — unless RAM-constrained with concurrent jobs (then train solo to keep them).

## 4. Convert (PTQ int8 — QAT not needed, G3)
`--train 0 --test_tflite_streaming_quantized 1 --use_weights best_weights` (same arch flags).

## 5. GATE (ship only if it beats v8)
```
python v8/evaluate_device.py  <new>.tflite           # device FRR/recall + FAPH (vs v8 0.046@0.8 / 0.059@0.9)
python v9/strat_eval.py       <new>.tflite out.json   # per-mode FRR; check music_snr5/reverb+music/muffled improve
```
Ship to `output_v10/` ONLY if device-FRR ≤ v8 at equal/lower device-FAPH AND no mode
regresses materially. Manifest: copy `output_v8/zakhar.json`, keep
`website=https://github.com/vvzvlad/zakhar-voice-assistant`, `author=vvzvlad`; set
`tensor_arena_size` from the converter's estimate.

## 6. Optional enhancements (independent of the retrain)
- **2nd-stage verifier** (`verifier/`, `research/G2_VERIFIER.md`): retrain its 65-param logreg
  on real false-wakes once you have reviewed ones; deploy as dual-head for no-VAD configs.
- **Faithful AM-softmax** (not the v9 logit-shift approx): if you want the loss lever, add a
  real L2-normalised angular-margin head (m≈0.35, s≈30) — the literature's ~49% rel FRR cut
  needs the true head, which the v9 approximation didn't capture.

## Reusable harness (all in research/)
`process_device.py` (split) · `strat_eval.py` (mode profiling) · `evaluate_device.py` (gate)
· `extract_embeddings.py` + `train_verifier.py` (verifier/field loop) · `quant_gap.py` (QAT
check if arch changes). Design of the full logging→mine→retrain loop: `PRODUCTION_LOOP.md`.
