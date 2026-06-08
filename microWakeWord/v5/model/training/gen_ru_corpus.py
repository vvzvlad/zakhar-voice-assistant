import sys, os, random, torch, soundfile as sf
torch.set_num_threads(4)
count, out_dir, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True); rng=random.Random(seed)
model,_ = torch.hub.load('snakers4/silero-models','silero_tts',language='ru',speaker='v4_ru',trust_repo=True)
SPK=['aidar','baya','kseniya','xenia','eugene','random','random','random']
# с/з-rich words (phonetic neighbours of Захар: с-vs-з, за-/са- onsets, -ар endings). NO «захар».
WORDS=["сахар","сахара","сахарница","загар","базар","кошмар","комар","пожар","гусар",
 "амбар","удар","самовар","словарь","сухарь","сосна","сапог","собака","суббота","забор",
 "заря","зефир","зебра","заноза","закат","захват","замок","засов","зажим","сазан","casar",
 "сазон","захаровка","зухра","сафари","сахаровед","засада","сосед","заказ","сазана","зарево",
 "макар","назар","лазарь","косарь","пахарь","знахарь","товар","договор","забота","засуха"]
TMPL=["{0}","{0} и {1}","Где {0}?","Купи {0}","{0} стоит дорого","Это {0}",
 "{0}, {1} и {2}","Принеси {0} пожалуйста","Я вижу {0}","Тут {0} а там {1}","{0}!"]
sr=48000; made=0
for i in range(count):
    t=rng.choice(TMPL); ws=[rng.choice(WORDS) for _ in range(3)]
    text=t.format(*ws)
    try:
        a=model.apply_tts(text=text,speaker=rng.choice(SPK),sample_rate=sr,put_accent=True,put_yo=True)
        sf.write(os.path.join(out_dir,f"ru_{seed}_{i:05d}.wav"),a.numpy(),sr); made+=1
    except Exception: pass
print("made",made,"/",count)
