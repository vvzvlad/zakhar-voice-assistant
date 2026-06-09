# Production-loop — WORKING prototype (v10 bonus)

Runnable pieces that implement the design in research/PRODUCTION_LOOP.md. Demonstrated on
existing data; drop in real field captures to use for real.

## Components (all run today)
1. `log_misses.py <weights> <cutoff> <pos_dir> <neg_dir> <out.npz>`
   Finds misses and logs 64-d **bottleneck embeddings** (not raw audio): false-rejects
   (missed wakes → gold positives to add, weight 12) and false-accepts (→ hard-neg pipeline).
2. `mine_hardneg.py <tflite> <neg_filelist> <out_dir> <hard_thr> <hard:rand> [cap]`
   Scores a negative pool, emits a NEW negative set mixing HARD (near-boundary) + RANDOM
   windows at a set ratio. **hard:random ≈ 1:2** by default — pure-hard mining collapses the
   model (over-specialises, FRR explodes), the random ballast prevents it. Per-file cap +
   cooldown stop one loud source flooding the set. (Demo: 57 hard + 114 random from 150 music
   files.)
3. Dedup/cluster (design): cluster FA embeddings by cosine (DBSCAN eps≈0.15) so one TV jingle
   → 1 cluster; rank by frequency×prob; human-review top-K → reviewed/neg + reviewed/pos.

## Retrain criterion (the "when")
Trigger a retrain when ANY of:
- ≥150 reviewed false-accepts since last train, OR
- ≥40 reviewed false-rejects (these move device-FRR fastest), OR
- canary device-noise FAPH > 0.5/h at deploy cutoff, OR
- 30 days with ≥1 new cluster.
Then: add reviewed/neg as a hard-neg set (penalty 6-8) **mixed 1:2 with random negatives**,
reviewed/pos to the device-positive set (weight 12), rebuild features (v8 mit_rirs intensity,
NOT SLR28), retrain (v8 arch, NO residuals), gate on device-eval + strat_eval + the SILENCE
check — ship only if it beats the incumbent without FAPH/silence regression.

## Why embeddings not raw audio
Privacy (audio never leaves device), bandwidth (~6 KB vs 48 KB/s), and the negative-mining
literature shows embedding-space (near-boundary) selection is what moves the FRR/FA frontier.
