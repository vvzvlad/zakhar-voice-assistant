#!/usr/bin/env python3
"""Production-loop miss-logger (working prototype).

Given the current model (keras weights for embeddings) + a labeled eval set, find the
MISSES and log their 64-d bottleneck embeddings (NOT raw audio) — the field loop's
off-device proxy for what the device would upload. Misses = positives scored below the
deploy cutoff (false rejects) and negatives scored above (false accepts). Embeddings feed
the dedup/clustering + review stages in PRODUCTION_LOOP.md.

Usage: log_misses.py <weights.h5> <cutoff> <pos_dir> <neg_dir> <out.npz>
"""
import sys, glob, numpy as np
sys.path.insert(0, "/home/claude/zakhar-mww")
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from v9.extract_embeddings import extract_dir

W, CUT = sys.argv[1], float(sys.argv[2])
pos_dir, neg_dir, out = sys.argv[3], sys.argv[4], sys.argv[5]
posp = sorted(glob.glob(pos_dir+"/*.wav")); negp = sorted(glob.glob(neg_dir+"/*.wav"))
pp, pe = extract_dir(W, posp); npr, ne = extract_dir(W, negp)
fr_mask = pp < CUT        # false rejects (missed wakes)
fa_mask = npr >= CUT      # false accepts
np.savez(out,
         fr_emb=pe[fr_mask], fr_prob=pp[fr_mask],
         fa_emb=ne[fa_mask], fa_prob=npr[fa_mask])
print(f"cutoff {CUT}: false_rejects {int(fr_mask.sum())}/{len(pp)}  "
      f"false_accepts {int(fa_mask.sum())}/{len(npr)}  -> {out}")
print("FR embeddings = gold real-tract positives to add (weight 12). "
      "FA embeddings -> cluster/dedup then review -> hard-neg set. See PRODUCTION_LOOP.md.")
