import sys, os, random, torch, soundfile as sf
torch.set_num_threads(4)
count,out_dir,seed=int(sys.argv[1]),sys.argv[2],int(sys.argv[3])
os.makedirs(out_dir,exist_ok=True); rng=random.Random(seed)
model,_=torch.hub.load('snakers4/silero-models','silero_tts',language='ru',speaker='v4_ru',trust_repo=True)
SPK=['aidar','baya','kseniya','xenia','eugene','random','random','random','random']
SP=["захаар","захааар","захаааар","захаааааар","Заха́ар","заха́аар","заха́ааар"]
sr=48000; made=0
for i in range(count):
    try:
        a=model.apply_tts(text=rng.choice(SP),speaker=rng.choice(SPK),sample_rate=sr,put_accent=True,put_yo=True)
        sf.write(os.path.join(out_dir,f"ds_{seed}_{i:05d}.wav"),a.numpy(),sr); made+=1
    except Exception: pass
print("silero",made)
