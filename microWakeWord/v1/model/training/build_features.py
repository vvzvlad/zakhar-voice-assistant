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
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "GainTransition": 0.25,
        "RIR": 0.5,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

for split in ("training", "validation", "testing"):
    out_dir = os.path.join(output_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    if split == "training":
        split_name, repetition = "train", 2
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
