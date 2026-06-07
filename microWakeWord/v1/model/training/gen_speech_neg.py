import sys, os, wave, random
sys.path.insert(0,"/home/claude/zakhar-mww/venv_piper/lib/python3.11/site-packages")
from piper import PiperVoice
from piper.config import SynthesisConfig
PHRASES = [
 "Привет как дела сегодня","Включи свет в гостиной","Какая сейчас погода на улице",
 "Поставь таймер на десять минут","Расскажи последние новости","Сколько сейчас времени",
 "Закрой шторы пожалуйста","Добавь молоко в список покупок","Включи музыку погромче",
 "Выключи телевизор в спальне","Захар Петрович пошёл домой","Скоро приедет такси",
 "Завтра будет солнечный день","Мама готовит вкусный обед","Дети играют во дворе",
 "Кошка спит на диване","Поезд прибывает по расписанию","Магазин закрывается в десять",
 "Я люблю гулять в парке","Книга лежит на столе","Сахар закончился вчера вечером",
 "Машина стоит у подъезда","Телефон разрядился совсем","Открой окно тут душно",
 "Назначь встречу на понедельник","Посчитай сумму всех чисел","Найди ближайшую аптеку",
 "Какой курс доллара сегодня","Переведи фразу на английский","Напомни купить хлеб",
]
voices=["denis","dmitri","irina","ruslan"]
rng=random.Random(7)
os.makedirs("speech_neg_raw",exist_ok=True)
loaded={v:PiperVoice.load(f"piper_voices/ru_RU-{v}-medium.onnx") for v in voices}
i=0
for rep in range(3):
  for ph in PHRASES:
    v=rng.choice(voices)
    cfg=SynthesisConfig(length_scale=round(rng.uniform(0.9,1.3),3),
                        noise_scale=round(rng.uniform(0.55,0.85),3),
                        noise_w_scale=round(rng.uniform(0.6,1.0),3),normalize_audio=True)
    with wave.open(f"speech_neg_raw/sneg_{i:04d}.wav","wb") as wf:
      loaded[v].synthesize_wav(ph,wf,syn_config=cfg)
    i+=1
print("generated",i,"speech-neg clips")
