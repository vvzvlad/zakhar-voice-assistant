#!/usr/bin/env python3
"""Build a RaggedMmap shard of NEGATIVE spectrograms from a list of (long) wav
files (e.g., music). Full spectrograms (capped) are stored; the trainer samples
random windows via truncation_strategy='random'.

Usage: build_neg_mmap.py <filelist.txt> <out_mmap_dir> <cap_seconds>
"""
import sys, os
import numpy as np
import soundfile as sf
sys.path.insert(0, "/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.audio.audio_utils import generate_features_for_clip
from mmap_ninja.ragged import RaggedMmap

filelist, out_dir, cap_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
cap_samples = int(cap_s * 16000)
paths = [l.strip() for l in open(filelist) if l.strip()]


def gen():
    for p in paths:
        try:
            d, sr = sf.read(p, dtype="int16")
        except Exception:
            continue
        if sr != 16000:
            continue
        if d.ndim > 1:
            d = d[:, 0]
        d = d[:cap_samples].astype(np.float32) / 32768.0
        if len(d) < 16000:  # skip <1s
            continue
        spec = generate_features_for_clip(d, step_ms=10)
        # need at least one training window
        if spec.shape[0] >= 30:
            yield spec.astype(np.float32)


os.makedirs(os.path.dirname(out_dir), exist_ok=True)
RaggedMmap.from_generator(out_dir=out_dir, sample_generator=gen(), batch_size=50, verbose=True)
print("SHARD DONE", out_dir)
