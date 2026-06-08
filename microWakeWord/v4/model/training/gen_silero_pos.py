import sys, os, random, torch, soundfile as sf
torch.set_num_threads(4)
count, out_dir, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True); rng=random.Random(seed)
model,_ = torch.hub.load('snakers4/silero-models','silero_tts',language='ru',speaker='v4_ru',trust_repo=True)
SPK=['aidar','baya','kseniya','xenia','eugene','random','random','random']  # bias to random for many voices
TXT=['Захар','Заха́р','захаар','Заха́ар','захааар','Захаар','захааар']
sr=48000; made=0
for i in range(count):
    spk=rng.choice(SPK); txt=rng.choice(TXT)
    try:
        a=model.apply_tts(text=txt,speaker=spk,sample_rate=sr,put_accent=True,put_yo=True)
        sf.write(os.path.join(out_dir,f"sil_{seed}_{i:05d}.wav"),a.numpy(),sr)
        made+=1
    except Exception as e:
        pass
print("made",made,"/",count)
