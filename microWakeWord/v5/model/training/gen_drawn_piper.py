import sys, os, wave, random
sys.path.insert(0,"/home/claude/zakhar-mww/venv_piper/lib/python3.11/site-packages")
from piper import PiperVoice
from piper.config import SynthesisConfig
voice_path,count,out_dir,seed=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
tag=os.path.basename(voice_path).replace("ru_RU-","").replace("-medium.onnx","")
os.makedirs(out_dir,exist_ok=True); rng=random.Random(seed)
# DRAWN-OUT forms only (elongated а). No short захар.
SP=["захаар","захааар","захаааар","захаааааар","Заха́ар","заха́аар","заха́ааар","захаааар"]
voice=PiperVoice.load(voice_path); made=0
for i in range(count):
    cfg=SynthesisConfig(length_scale=round(rng.uniform(1.3,2.6),3),
                        noise_scale=round(rng.uniform(0.5,0.9),3),
                        noise_w_scale=round(rng.uniform(0.55,1.05),3),normalize_audio=True)
    with wave.open(os.path.join(out_dir,f"dp_{tag}_{i:05d}.wav"),"wb") as wf:
        voice.synthesize_wav(rng.choice(SP),wf,syn_config=cfg)
    made+=1
print(tag,made)
