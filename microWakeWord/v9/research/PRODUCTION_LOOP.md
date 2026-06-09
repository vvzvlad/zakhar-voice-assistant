# G10 — Production data-loop scaffold (field → retrain)

Goal: a concrete, low-bandwidth loop that turns real field misfires into the next
training set, with a defined "when to retrain" trigger. Grounded in the research
(Hou 2020 negative-mining; Amazon 2018 two-stage; survey arXiv 2506.11169): log
**embeddings, not raw audio**; only **human-reviewed** false-wakes become negatives.

## 1. On-device capture (firmware side — design, needs an ESPHome/mWW patch)

mWW already emits a per-window probability stream. Add two shadow bands around the
deploy cutoff `c` (e.g. c=0.8):

- **Near-miss-positive** band `[c-0.15, c)`: the model *almost* fired. If the user
  then repeats / the voice assistant did NOT wake, this window is a likely **false
  reject** (missed wake). Capture it.
- **Shadow-accept** band `[c, 1]` that the **2nd-stage verifier (G2) rejected**, or
  that the user immediately cancelled: likely **false accept**. Capture it.

What to store per event (ring buffer in flash, cap ~64 events):
- the **40-band spectrogram window** (~1.5 s = 150×40 int8 ≈ 6 KB) — NOT raw audio
  (privacy + bandwidth). Optionally the 64-d bottleneck embedding (~64 B) if the
  verifier head is on-device.
- `prob`, `cutoff`, `vad_flag`, `verifier_score`, `ts`, `fw_version`, `model_hash`.

Upload opportunistically (Wi-Fi, batched) to the collector. ~6 KB/event → trivial.
No audio leaves the device → privacy-preserving, matches HA's on-device ethos.

## 2. Collector + review (off-device)

`field_events/` accumulates uploaded events (spectrogram window + metadata JSON).

Pipeline (`mine_field.py`, scaffolded on top of existing `v5/mine_false.py`):
1. **Reconstruct** model probability + 64-d embedding for each event (run the keras
   non-streaming model on the stored window).
2. **Dedup/cluster** events by embedding cosine-distance (DBSCAN, eps≈0.15). One TV
   jingle that misfires 50× collapses to 1 cluster — prevents one source flooding
   the negative set (the same cap-per-source logic as `mine_false.py`'s per-file cap).
3. **Rank** clusters by (frequency × mean trigger-prob). Surface top-K to a human
   reviewer with a 1.5 s audio snippet *if* the user opted into audio retention;
   otherwise label from context (paired ASR transcript / "assistant woke but user
   said cancel").
4. **Label**: `false_accept` (→ hard-negative set) / `true_accept` (ignore) /
   `false_reject` (→ positive set, weight high — these are gold real-tract positives).

Reviewed labels write to `reviewed/neg/` and `reviewed/pos/` (raw windows + uuid so
the leakage-safe split in `process_device.py` keeps them disjoint from eval).

## 3. Retrain trigger (the "when")

Retrain when ANY of:
- **≥150 reviewed false-accepts** accumulated since last train (enough new hard-neg
  signal to matter past the ~20× augmentation plateau), OR
- **≥40 reviewed false-rejects** (missed real wakes — these move device-FRR, our main
  metric, the fastest), OR
- **canary FAPH regression**: a fixed 6-min held-out device-noise canary, re-scored
  weekly, exceeds 0.5/h at the deploy cutoff, OR
- **30 days** elapsed with ≥1 new cluster (drift catch-all).

On trigger: add `reviewed/neg` to the hard-negative feature sets (penalty_weight 6–8,
matching `features_v4_hardneg`/`features_v5_mined`), add `reviewed/pos` to the device
positive set (sampling_weight 12, like `features_v8_device`), rebuild features,
retrain, and **gate on the same leakage-safe device-eval + strat_eval** — ship only
if it beats the incumbent (same rule we use for v8→v9).

## 4. What's reusable today (already built)

- `v5/mine_false.py` — adversarial false-trigger miner (audio → hard-neg windows,
  per-file cap + cooldown to avoid one-source flooding). This is the off-device
  miner's core; `mine_field.py` would wrap it with the embedding-dedup step.
- `v8/process_device.py` — uuid/source leakage-safe split (keeps mined data out of eval).
- `v8/evaluate_device.py` + `v9/strat_eval.py` — the gates the retrain must pass.
- `v9/extract_embeddings.py` (built for G2) — the 64-d bottleneck embedder used for
  clustering/dedup here too.

## 5. Why embeddings not raw audio (research-backed)

- Privacy: raw user audio never leaves the device (HA selling point).
- Bandwidth: 6 KB spectrogram vs ~48 KB/s PCM.
- The negative-mining literature (Hou 2020) shows *embedding-space* hard-negative
  selection (nearest decision boundary) is what improves FRR/FA trade-off, not raw
  volume of negatives — so the embedding is the right currency for the whole loop.
