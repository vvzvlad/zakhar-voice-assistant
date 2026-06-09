#!/usr/bin/env python3
"""FIXED drawn-out Piper generator: correct spelling only («Захар»/«Заха́р»), drawn-out
via high length_scale (NO broken repeated-vowel spellings that break the phonemizer)."""
import sys, os, wave, random
sys.path.insert(0,"/home/claude/zakhar-mww/venv_piper/lib/python3.11/site-packages")
from piper import PiperVoice
from piper.config import SynthesisConfig
voice_path,count,out_dir,seed=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
tag=os.path.basename(voice_path).replace("ru_RU-","").replace("-medium.onnx","")
os.makedirs(out_dir,exist_ok=True); rng=random.Random(seed)
# Correct spellings ONLY. Stress mark on 2nd syllable is fine (proper phonemization).
TEXTS=["Захар","Заха́р","Захар","Заха́р","Захар."]
voice=PiperVoice.load(voice_path); made=0
for i in range(count):
    # drawn-out = slow correct speech (high length_scale), NOT vowel repetition
    cfg=SynthesisConfig(length_scale=round(rng.uniform(1.3,2.1),3),
                        noise_scale=round(rng.uniform(0.55,0.85),3),
                        noise_w_scale=round(rng.uniform(0.6,1.0),3),normalize_audio=True)
    with wave.open(os.path.join(out_dir,f"v8p_{tag}_{i:05d}.wav"),"wb") as wf:
        voice.synthesize_wav(rng.choice(TEXTS),wf,syn_config=cfg)
    made+=1
print(tag,made)
