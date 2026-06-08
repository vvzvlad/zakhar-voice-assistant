# Tavily deep-research №2 — как ещё улучшить «захааар» за пределами данных/майнинга

Глубокое исследование (tavily, 3 параллельных «pro»-прогона) по ТЗ
[RESEARCH_BRIEF_2_next_improvements.md](RESEARCH_BRIEF_2_next_improvements.md). Мы уже выжали
масштаб негативов, confusables, 3 раунда майнинга (плато), доп. TTS-спикеров, VAD-гейт и
рычаг длительности. Ниже — что осталось, по убыванию реальной отдачи в деплое.

## TL;DR
- **Главный недооценённый рычаг — train/serve acoustic gap.** Voice PE прогоняет звук через
  XMOS speech-enhancement + TF-Micro NS/AGC ДО модели; мы же учим на чистом TTS/записях.
  Тренировать на post-frontend аудио (захват с устройства или симуляция канала) — это бьёт
  по реальному recall сильнее любых новых негативов.
- **Самый дешёвый прирост по ложным — вторая стадия (verifier).** До **7–8× срез FAPH**
  (ICASSP'23), запускается только после срабатывания → почти бесплатно по среднему компьюту.
  Это и закроет наш остаток (no-VAD вокальная музыка).
- Дальше по ROI: far-field/комната-аугментация → distillation → on-device персонализация
  (few-shot прототипы) → больше реальных данных. Смена архитектуры (BC-ResNet) — только если
  всё выше упрётся.

---

## Tier 1 — наибольшая новая отдача

### 1. Закрыть TRAIN/SERVE ACOUSTIC GAP (вероятно, #1 по реальному эффекту)
Модель учится не на том, что ей реально скармливает устройство (XMOS + TF-Micro NS/AGC).
Три подхода по убыванию точности/росту трудозатрат:
- **A. Захват post-frontend PCM с устройства** (рекомендуемый первый эксперимент). Перевести
  Voice PE в боевую конфигурацию фронтенда и записать то, что реально видит модель
  (`arecord -r 48000 -f S32_LE -c 2`), затем тренировать/до-обучать на этом. ESP-SR: ~15
  произношений на дистанцию; снимать 1/3/5 м, разные комнаты/спикеры.
- **B. Прогон обучающего корпуса через тот же фронтенд** (XMOS SDK / TF-Micro NS/AGC) офлайн —
  максимальная точность, но инженерно дороже (нужен SDK/детерминизм).
- **C. Симуляция канала**: gain-аугментация **+10…−40 дБ** (имитация AGC, рецепт **RepCNN**),
  RIR-реверберация, добавление бытового шума, эмуляция формата 48 кГц→down-sample. Дёшево.
- Источники: RepCNN (arXiv 2406.02652), Amazon data-efficient (2010.06659), XMOS docs, HA
  community (тред про то, что pipeline форсит NS+AGC), ESP-SR audio front-end.

### 2. Вторая стадия — verifier / successive refinement (лучший FA-срез на единицу затрат)
После срабатывания крошечного KWS запускать лёгкий второй классификатор на буфере/эмбеддинге
(только на триггерах → средний компьют почти не растёт).
- Эффект: **до 7–8× меньше ложных** (in/out-of-domain, arXiv 2304.03416); openWakeWord
  custom verifier (логрег/MLP на эмбеддингах, триггерится при score>0.5) режет FA без заметного
  роста false-reject. Это прямой добиватель нашего остатка (вокальная музыка, что VAD пропускает).
- Рецепт: tiny MLP **32→ReLU→10→sigmoid** на bottleneck-эмбеддинге, BCE; негативы = логированные
  ложные + adversarial-синты + подкасты/музыка; запуск на окне 100–300 мс после триггера.
  INT8-совместимо с TFLM; грузить условно, чтобы не держать arena постоянно.
- Цена на ESP32-S3: типовые INT8-модели ~50–500 КБ flash, arena ~30–300 КБ; verifier крошечный.

### 3. Far-field / реальная комната (поднимает РЕАЛЬНЫЙ recall, не offline)
- Мульти-condition аугментация (рецепт Amazon, 20× на позитив): **10% clean / 30% reverb /
  30% noise / 30% reverb+noise**.
- RIR: банк реальных (BUT ReverbDB / AIR) + синтетика (pyroomacoustics / rir-generator /
  IR-GAN), T60 ≈ 0.2–1.0 с, дистанции 0.5–4 м.
- Шум: бытовой (MUSAN/кухня/ТВ), SNR ~N(10 дБ, 3) или 10–20 дБ. Gain +10…−40 дБ.
- Ожидаемо: измеримый прирост deployed-recall; offline-метрика переоценивает, если тест не
  через реальный фронтенд → валидировать на захваченном с устройства аудио.

---

## Tier 2 — сильно, средний труд

### 4. Knowledge distillation (teacher → 32k INT8 student)
Крупная non-streaming teacher-модель «захааар» → дистилл в наш 32k student: temperature-KL +
hard-CE (+ опц. feature/utterance-уровень), затем QAT-finetune. APSIPA'25: **−36% относительный
EER** у дистиллированного student; noisy-student повышает robustness. Совместимо с mWW/QAT.

### 5. On-device персонализация (few-shot прототипы)
- **Реально на ESP32-S3:** enrollment — посчитать эмбеддинг на каждое произношение (та же
  инференс-цепочка), хранить средний прототип, на детекции — косинус-дистанция. Без обучения на
  устройстве (TFLM не умеет train-ops). Хранить в LittleFS + flash encryption.
- Эффект: few-shot KWS (ResNet15) — 10 примеров/класс → ~80% acc @5% FAR; материальный прирост
  под голоса домочадцев. On-device fine-tune НЕ реально на S3 — только сервер/офлайн.

### 6. Больше реальных данных (главный оставшийся data-рычаг — совпало с нашими итогами)
- HA Wake-Word Collective (opt-in краудсорс) поднял recall заметно; on-device logging реальных
  true/false срабатываний с дозаливкой (хранить эмбеддинги/фичи, не сырой звук — приватность).
- Сколько: few-shot гейн уже с ~10 примеров; для незнакомых голосов — больше спикеров/часов
  (точные кривые diminishing returns в литературе не даны — мерить экспериментом).

---

## Tier 3 — опционально, низкий маржинальный ROI

### 7. Смена архитектуры (только если Tier 1–2 упрутся)
**BC-ResNet-2 ≈ 27.3k параметров, 97.3% Speech Commands v1** — лучше accuracy/параметр, чем
MixConv, и в нашем бюджете. Но порт в streaming+quant pipeline mWW + проверка операторов
(TFLM/CMSIS-NN) — высокие усилия, неочевидный выигрыш против data/verifier работы.

### 8. Loss-функции
AM-softmax (margin 0.35–0.4), focal (γ≈2), triplet/metric-learning — инкрементально к
снижению FAR; требует патча train.py. Ниже приоритет — class/penalty-веса уже это закрывают.

---

## Оценка боевого качества (методология)
- Speaker-disjoint train/val/test; НЕ переиспользовать произношения; **ограничить синтетику в
  тесте** (иначе оверфит на TTS-голоса). Метрики: FRR + FA/час + DET-кривые, bootstrap-CI.
- Калибровать пороги на val, валидировать на РЕАЛЬНОМ аудио с устройства (через фронтенд).
  Бенчмарки (Speech Commands/DiPCo/PicoVoice) — для pretrain/ориентира, не предсказывают канал.

## Главный вывод
Мы выжали ось «данные/майнинг». Следующий реальный прирост — **не больше данных того же типа,
а правильного типа**: (1) post-frontend/канал-реалистичные данные, (2) вторая стадия-verifier
под остаток ложных, (3) far-field-аугментация, и (4) больше РЕАЛЬНЫХ протяжных записей под
конкретные голоса. Смена модели/loss — в последнюю очередь.

## Источники (ключевые)
- RepCNN (micro wakeword, gain-aug): https://arxiv.org/html/2406.02652v1
- Successive refinement / verifier (7–8× FA): https://arxiv.org/pdf/2304.03416
- openWakeWord (custom verifier, данные): https://github.com/dscripka/openWakeWord
- Amazon data-efficient (multi-condition split): https://assets.amazon.science/7c/b2/5e3e6a164920bfc167fb5586d3f2/scipub-1260.pdf
- Distillation KWS (−36% EER): https://apsipa.org/proceedings/2025/papers/APSIPA2025_P385.pdf
- Noisy student-teacher KWS: https://isca-archive.org/interspeech_2021/park21_interspeech.pdf
- BC-ResNet: https://ar5iv.labs.arxiv.org/html/2106.04140
- Few-shot KWS (prototypes): https://arxiv.org/html/2506.17686v1 ; on-device personalization: https://arxiv.org/pdf/2408.12481
- HA Wake-Word Collective: https://www.home-assistant.io/blog/2024/10/24/wake-word-collective
- RIR / far-field: BUT ReverbDB (fit.vut.cz), pyroomacoustics, IR-GAN (gamma.umd.edu/pro/speech/ir-gan)
- XMOS XCORE-VOICE / ESP-SR front-end: xmos.com docs ; docs.espressif.com/projects/esp-sr
- ESPHome micro_wake_word: https://esphome.io/components/micro_wake_word
