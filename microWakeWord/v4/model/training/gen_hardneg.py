#!/usr/bin/env python3
import sys, os, wave, random
sys.path.insert(0,"/home/claude/zakhar-mww/venv_piper/lib/python3.11/site-packages")
from piper import PiperVoice
from piper.config import SynthesisConfig
voice_path,count,out_dir,seed=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
tag=os.path.basename(voice_path).replace("ru_RU-","").replace("-medium.onnx","")
os.makedirs(out_dir,exist_ok=True); rng=random.Random(seed)
# Phonetic neighbours of "Захар" (rhymes / similar onsets) — teach the boundary.
WORDS=["сахар","загар","пожар","комар","базар","товар","кошмар","угар","удар",
       "амбар","гусар","Макар","Захаров","захват","закат","замах","заход",
       "сахаром","загаром","Захарка","забор","задал","зажал"]
voice=PiperVoice.load(voice_path); made=0
for i in range(count):
    text=rng.choice(WORDS)
    cfg=SynthesisConfig(length_scale=round(rng.uniform(0.85,1.7),3),
                        noise_scale=round(rng.uniform(0.5,0.9),3),
                        noise_w_scale=round(rng.uniform(0.55,1.05),3),normalize_audio=True)
    with wave.open(os.path.join(out_dir,f"hn_{tag}_{i:05d}.wav"),"wb") as wf:
        voice.synthesize_wav(text,wf,syn_config=cfg)
    made+=1
print(f"{tag}: {made}")
