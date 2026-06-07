#!/usr/bin/env python3
"""Generate synthetic Russian 'Захар' positives with a single piper voice.
Usage: generate_synth.py <voice_onnx> <count> <out_dir> <seed>
One voice loaded once; per-clip randomized length/noise + drawn-out spellings.
"""
import sys, os, wave, random

sys.path.insert(0, "/home/claude/zakhar-mww/venv_piper/lib/python3.11/site-packages")
from piper import PiperVoice
from piper.config import SynthesisConfig

voice_path, count, out_dir, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
voice_tag = os.path.basename(voice_path).replace("ru_RU-", "").replace("-medium.onnx", "")
os.makedirs(out_dir, exist_ok=True)
rng = random.Random(seed)

# Phonetic spellings. The wake word is pronounced drawn-out ("захааар"), so weight
# the elongated variants. U+0301 is the combining acute accent (stress mark).
SPELLINGS = (
    ["Захар"] * 3
    + ["Заха́р"] * 3      # explicit stress on the second syllable
    + ["захаар"] * 3           # mild elongation
    + ["захааар"] * 3          # strong elongation (matches recordings)
    + ["Захаар"] * 2
    + ["захааар"] * 1          # very drawn out
)

voice = PiperVoice.load(voice_path)
made = 0
for i in range(count):
    text = rng.choice(SPELLINGS)
    # Longer length_scale = slower/more drawn out. Range covers brisk..very slow,
    # biased toward the slower side to mirror the drawn-out recordings.
    length_scale = round(rng.uniform(0.85, 1.95), 3)
    noise_scale = round(rng.uniform(0.55, 0.85), 3)
    noise_w = round(rng.uniform(0.6, 1.0), 3)
    cfg = SynthesisConfig(
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
        normalize_audio=True,
    )
    out = os.path.join(out_dir, f"synth_{voice_tag}_{i:05d}.wav")
    try:
        with wave.open(out, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=cfg)
        made += 1
    except Exception as e:
        print(f"FAIL {i}: {e}", file=sys.stderr)
print(f"{voice_tag}: generated {made}/{count}")
