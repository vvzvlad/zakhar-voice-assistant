#!/usr/bin/env python3
"""Build 40-band spectrogram features for the operator's REAL room-speech HARD-NEGATIVES.
These clips are ALREADY real device-tract audio (ch1 room, x4), so unlike synthetic positives we use a
LIGHTER augmenter: most clips keep their bare quiet-speech character (the exact false-accept failure mode),
with only mild Gain/background variety. remove_silence MUST be 0 (the ambient IS the negative).
Usage: build_features_hardneg.py <input_dir> <output_dir> <train_rep>"""
import sys, os
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

input_dir, output_dir = sys.argv[1], sys.argv[2]
TRAIN_REP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.makedirs(output_dir, exist_ok=True)

clips = Clips(input_directory=input_dir, file_pattern="*.wav", max_clip_duration_s=None,
              remove_silence=False, trim_zeros=False, random_split_seed=10, split_count=0.1)

# LIGHT aug: keep the bare quiet-speech distribution dominant; add only mild variety.
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.05, "TanhDistortion": 0.0, "PitchShift": 0.0,
        "BandStopFilter": 0.05, "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.3,   # was 0.75 — don't bury his quiet speech under added bg
        "Gain": 1.0, "GainTransition": 0.2,
        "RIR": 0.3,                   # was 0.9 — already real-tract; don't double-convolve most clips
    },
    impulse_paths=["/home/claude/zakhar-mww/device_ir"],
    background_paths=["/home/claude/zakhar-mww/device_noise"],   # only real device ambient (no fma music over speech)
    background_min_snr_db=0, background_max_snr_db=15,
    min_jitter_s=0.195, max_jitter_s=0.205,
)
for split in ("training", "validation", "testing"):
    out_dir = os.path.join(output_dir, split); os.makedirs(out_dir, exist_ok=True)
    if split == "training":
        split_name, repetition, sg = "train", TRAIN_REP, SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10)
    elif split == "validation":
        split_name, repetition, sg = "validation", 1, SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10)
    else:
        split_name, repetition, sg = "test", 1, SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10)
    RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
        sample_generator=sg.spectrogram_generator(split=split_name, repeat=repetition), batch_size=100, verbose=True)
print(f"HARDNEG FEATURES DONE for {input_dir} -> {output_dir}")
