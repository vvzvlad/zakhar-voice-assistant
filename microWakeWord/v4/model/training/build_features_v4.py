#!/usr/bin/env python3
"""Build augmented 40-band spectrogram features (RaggedMmap) for one positive
clip directory, producing training/validation/testing splits.

Usage: build_features.py <input_dir> <output_dir> <remove_silence:0|1>
"""
import sys, os
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

input_dir, output_dir, remove_silence = sys.argv[1], sys.argv[2], bool(int(sys.argv[3]))
TRAIN_REP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(output_dir, exist_ok=True)

clips = Clips(
    input_directory=input_dir,
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=remove_silence,
    trim_zeros=False,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.3,
        "TanhDistortion": 0.2,
        "PitchShift": 0.35,
        "BandStopFilter": 0.2,
        "AddColorNoise": 0.3,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "GainTransition": 0.4,
        "RIR": 0.6,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k", "/home/claude/zakhar-mww/v3/fma_small_16k"],
    background_min_snr_db=-10,
    background_max_snr_db=12,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

for split in ("training", "validation", "testing"):
    out_dir = os.path.join(output_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    if split == "training":
        split_name, repetition = "train", TRAIN_REP
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
        )
    elif split == "validation":
        split_name, repetition = "validation", 1
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
        )
    else:  # testing -> streaming-style, no artificial repetition
        split_name, repetition = "test", 1
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
        )

    RaggedMmap.from_generator(
        out_dir=os.path.join(out_dir, "wakeword_mmap"),
        sample_generator=spectrograms.spectrogram_generator(
            split=split_name, repeat=repetition
        ),
        batch_size=100,
        verbose=True,
    )
print(f"FEATURES DONE for {input_dir} -> {output_dir}")
