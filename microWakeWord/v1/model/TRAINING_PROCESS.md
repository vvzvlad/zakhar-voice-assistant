# Процесс обучения wake word «Захар» (microWakeWord) — как повторить

Документ описывает, **как именно** была обучена модель `model/zakhar.tflite` для
on-device распознавания слова «Захар» на ESP32-S3 / HA Voice PE, чтобы прогон можно было
воспроизвести или итерировать. Это пересказ реального пайплайна; формальный отчёт с
метриками — в [model/TRAINING_REPORT.md](model/TRAINING_REPORT.md), а исходники всех шагов —
в [training/](training/).

> TL;DR результата (v1, отгружено): модель `mixednet` 26 049 параметров, INT8-streaming
> TFLite ≈ 62 КБ. Recall **0.81** на 93 отложенных реальных записях @cutoff 0.90; ложные
> срабатывания на русской речи ≈0 (включая близкое «Сахар»); слабое место — непрерывная
> музыка (~16/час @0.90), лечится поднятием `probability_cutoff` до 0.95–0.97 в YAML.

---

## 0. Где это крутилось

- Нода `agent-node` (10.20.30.226): Debian 13, **50 vCPU, ~28 ГиБ RAM, ~200 ГБ диска,
  GPU нет** → обучение CPU-only (TensorFlow CPU). Всё лежит в `/home/claude/zakhar-mww/`.
- Движок: официальный **`OHF-Voice/micro-wake-word`**, поставлен нативно (pip), а не через
  Docker-тренеры сообщества (те GPU-ориентированы и бесполезны без GPU).
- **Python 3.11 через `uv`** (фреймворк рассчитан на 3.10–3.11; системный Python 3.13
  несовместим). Отдельный venv `venv_piper` под Piper-TTS.

> ВАЖНО: в скриптах [training/](training/) пути **абсолютные** (`/home/claude/zakhar-mww/...`).
> Это копия рабочих скриптов «как есть». Для повтора на другой машине поправь пути (или
> воссоздай ту же структуру каталогов).

---

## 1. Данные

### 1.1 Позитивы — реальные голоса (главный актив)
- 621 присланная запись «захааар» (`positive_samples/`, моно/16-bit, 44.1/48 кГц).
- Ресемпл в **16 кГц моно 16-bit**; сплит **528 train / 93 held-out** (seed 42).
  Held-out в обучении НЕ участвует — только для честной оценки recall.

### 1.2 Позитивы — синтетика (Piper TTS)
- Скрипт [training/generate_synth.py](training/generate_synth.py).
- **Важный нюанс:** `piper-sample-generator` из ТЗ умеет только английский → вместо него
  взят **`piper-tts` с 4 русскими голосами** `denis / dmitri / irina / ruslan`.
- По 800 клипов на голос = **3200** синтетических позитивов.
- Слово подаётся кириллицей в нескольких фонетических написаниях с упором на протяжность:
  `Захар`, `Заха́р` (U+0301 — ударение), `захаар`, `захааар`, `Захаар`.
- На каждый клип рандомизируются `length_scale` 0.85–1.95 (медленнее = протяжнее),
  `noise_scale`, `noise_w_scale`.

### 1.3 Негативы и фон
- Скрипт [training/dl_negatives.sh](training/dl_negatives.sh): предгенерированные
  спектрограммы microWakeWord с HF `kahrendt/microwakeword` — наборы `speech`, `no_speech`
  (внутри FMA-музыка + FSD50K + WHAM), `dinner_party`, `dinner_party_eval` (ambient для FAPH).
- Скрипт [training/prep_backgrounds.sh](training/prep_backgrounds.sh): фон для аугментации —
  **FMA-музыка → 16 кГц wav** (`fma_16k`) и **MIT RIR** (270 импульсных характеристик для
  реверберации, `mit_rirs`, тянутся через `datasets`).
- (Опционально) [training/gen_speech_neg.py](training/gen_speech_neg.py) — синтетические
  русские фразы-негативы (включая «Сахар…», «Захар Петрович…») как hard-negatives/пробы.

---

## 2. Генерация признаков (спектрограммы)

- Скрипт [training/build_features.py](training/build_features.py) — строит **40-полосные
  спектрограммы** (`step_ms=10`, т.е. `feature_step_size: 10`) в формате RaggedMmap,
  отдельно для `features_real` и `features_synth`, со сплитами training/validation/testing.
- Аугментация (класс `Augmentation` фреймворка): фон-микс `AddBackgroundNoise` p=0.75 из
  `fma_16k` (SNR −5…+10 дБ), `RIR` p=0.5 из `mit_rirs`, `Gain`, лёгкие EQ/Distortion/
  PitchShift/BandStop/ColorNoise. `augmentation_duration_s=3.2`.
- На train — `slide_frames=10` + 2× повтор → ≈ **51 200** синтетических и ≈ **8 440**
  реальных обучающих срезов; на test — `slide_frames=1` (как в потоковом инференсе).

---

## 3. Обучение

- Конфиг [training/training_parameters.yaml](training/training_parameters.yaml),
  запуск [training/train.sh](training/train.sh) (в `screen`, `OMP_NUM_THREADS=50`).
- Архитектура `mixednet`: `pointwise_filters 64,64,64,64`, `repeat_in_block 1,1,1,1`,
  `mixconv_kernel_sizes [5],[7,11],[9,15],[23]`, `first_conv 32×5`, `stride 3` →
  **26 049 параметров**.
- Расписание: **25 000 шагов** тремя стадиями `[10000,10000,5000]`, LR `[1e-3,5e-4,1e-4]`,
  `batch_size 128`. CPU ~7.5 шага/с → ≈ **50 минут**.
- **Веса классов:** `positive 1`, **`negative 20`** — повышенное подавление ложных
  (короткая 2-сложная фраза).
- **Веса сэмплирования источников:** реальные позитивы **4.0**, синтетика **2.0**,
  `speech 10`, `dinner_party 10`, `no_speech 5`.
- SpecAugment: time/freq-маски размер 5 ×2.
- **Выбор лучших весов (двухступенчатый):** сначала довести
  `ambient_false_positives_per_hour ≤ 2`, затем максимизировать `average_viable_recall`.
- На выходе сразу квантованный INT8 streaming TFLite:
  `trained_models/zakhar/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`.

---

## 4. Оценка и манифест

- [training/evaluate.py](training/evaluate.py) гоняет квантованную потоковую модель
  **так же, как устройство**: `Model(tflite, stride=3)` (stride = входной stride модели, НЕ 1),
  скользящее среднее вероятностей, cooldown 25 срезов после срабатывания. Считает recall на
  held-out реальных и FAPH на длинном ambient-аудио по сетке `cutoff × window`. Результат —
  `model/eval/eval_grid_{speech,music}.json`.
- [training/generate_manifest.py](training/generate_manifest.py) пишет манифест **v2**
  (`feature_step_size: 10`, подобранные `probability_cutoff`/`sliding_window_size`,
  `tensor_arena_size` с запасом, `minimum_esphome_version: 2024.7.0`).
- **Рабочая точка v1:** `probability_cutoff 0.90`, `sliding_window_size 3`.

---

## 5. Итерации

- **v1** — `negative_class_weight 20`. **← отгружено.**
- **v2** — `negative_class_weight 30`, `no_speech` вес 12 (больше музыки/шума), цель — срезать
  музыкальный FAPH. Конфиг [training/training_parameters_v2.yaml](training/training_parameters_v2.yaml),
  запуск [training/train_v2.sh](training/train_v2.sh). **Итог: фронтир recall↔FAPH не улучшил**
  (при равном музыкальном FAPH recall был ниже, чем у v1) → оставлен v1. Вывод: для 2-сложной
  фразы на этих данных и CPU-обучении v1 уже близок к достижимому пределу; остаточная
  чувствительность к музыке — свойство короткой фразы, лечится порогом в YAML.

---

## 6. Грабли (переиспользуемые)

Зафиксировано по ходу — пригодится при повторе/итерациях:

- **PYTHONPATH обязателен:** все команды mWW запускать с
  `PYTHONPATH=/home/claude/zakhar-mww/micro-wake-word` (подпакеты `audio`/`layers` без
  `__init__.py` → namespace-import).
- **Абсолютные пути в YAML:** `train.sh` делает `cd` в каталог фреймворка, поэтому
  `features_dir`/`train_dir` в конфиге ДОЛЖНЫ быть абсолютными.
- **Патчи под TF 2.21 / numpy 2.x:** в `microwakeword/train.py` и `test.py` заменить
  `.numpy()` → `np.asarray()` и `np.trapz` → `np.trapezoid`; доустановить `tensorboard`.
- **`datasets` запинить на 3.6.0** — 4.x/5.x тянут `torchcodec` даже для wav; 3.6 декодит
  wav через `soundfile`.
- **Оценка:** обязательно `Model(tflite, stride=3)` (иначе метрики врут), cooldown 25 срезов.
- TF сыпет CUDA/oneDNN-предупреждения в stderr; потоковая валидация печатает строку через
  `\r` на мини-батч — фильтровать/парсить через `tr '\r' '\n'`.

---

## 7. Повтор «с нуля» (порядок)

```bash
# на ноде, в /home/claude/zakhar-mww, при активном venv (Py 3.11) и PYTHONPATH на фреймворк
bash training/dl_negatives.sh          # негативы с HF
bash training/prep_backgrounds.sh      # fma_16k + mit_rirs для аугментации
python training/generate_synth.py piper_voices/ru_RU-denis-medium.onnx 800 synth_raw 1   # ×4 голоса
#   (real_16k готовится ресемплом positive_samples → 16k моно, сплит 528/93)
python training/build_features.py real_16k  features_real  0
python training/build_features.py synth_raw features_synth 0
bash training/train.sh                 # обучение в screen (~50 мин CPU)
python training/evaluate.py <tflite> real_heldout <ambient.wav> model/eval/eval_grid.json
python training/generate_manifest.py <tflite> zakhar.json zakhar.tflite 0.90 3
```

Чтобы **улучшить** (направление «тяжёлый переобуч»): больше музыкальных/ambient негативов
прямо в обучение, больше шагов (40–80k), больше голосов/вариаций Piper, при необходимости
чуть крупнее сеть (в пределах `tensor_arena`) — и больше итераций с честной held-out оценкой.

---

## 8. Что лежит в репозитории

| Путь | Что |
|------|-----|
| [model/zakhar.tflite](model/zakhar.tflite) | обученная INT8 streaming модель (v1) |
| [model/zakhar.json](model/zakhar.json) | манифест v2 для ESPHome |
| [model/esphome_micro_wake_word.yaml](model/esphome_micro_wake_word.yaml) | готовый сниппет `micro_wake_word` |
| [model/TRAINING_REPORT.md](model/TRAINING_REPORT.md) | формальный отчёт агента (метрики, лимиты) |
| [model/eval/](model/eval/) | сетки оценки порогов (recall/FAPH) |
| [training/](training/) | реальные скрипты и конфиги пайплайна (см. выше) |
| [TZ_zakhar_microwakeword.md](TZ_zakhar_microwakeword.md) | исходное ТЗ |
| [compass_artifact_…md](compass_artifact_wf-7bd4ae7f-f31c-4361-bdd6-e5164733eb58_text_markdown.md) | исследование по microWakeWord |
