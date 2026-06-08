# Запрос на исследование №2: как ещё улучшить on-device «захааар», когда data/mining уже исчерпаны

## Что строим и ограничения
On-device wake word **протяжный «захааар»** (долгая «а»; короткий «захар» НЕ цель) на
microWakeWord (OHF-Voice/micro-wake-word): TensorFlow → квантованный INT8 streaming TFLite
для TFLM на ESP32-S3 (Home Assistant Voice PE) через ESPHome `micro_wake_word`.
Жёсткие рамки: модель ~32k параметров, ~78 КБ INT8, вход 16 кГц моно, 40-полосные
micro_speech спектрограммы (feature_step_size 10), потоковый инференс, tensor_arena ~45 КБ,
обучение CPU-only (GPU нет, но ядер/RAM/диска вдоволь). Фраза фиксирована.

## Текущее лучшее (v5) и ТОЧНЫЕ метрики
Оценка как на устройстве (stride=3, sliding-window mean, cooldown 25; 93 РЕАЛЬНЫЕ протяжные
held-out ↔ чистая disjoint 90-мин MUSAN-музыка):
- Drawn-out recall: **100% клипов ≥0.9** (mean 0.999); recall@cutoff0.9/win3 = 0.946.
- «Сахар» near-miss: 1/90.
- Музыка: **0 ложных/час С VAD** (для всех версий); 2.67/час БЕЗ VAD (остаток — вокальная музыка).

## Что УЖЕ сделано и ИСЧЕРПАНО (НЕ предлагать заново)
- Масштаб и разнообразие негативов (HF kahrendt полностью + MUSAN music/noise + ESC-50 +
  FMA-medium/small) — десятки часов музыки.
- Confusable hard-negatives (сахар/загар/пожар/… , sampling 8 / penalty 5).
- **Adversarial false-trigger mining — 3 РАУНДА (Amazon 2010.06659): ПЛАТО.** Раунд-1 дал v5;
  раунды 2–3 фронтир НЕ улучшили (recall падал / music-FAPH рос).
- Больше синтетических спикеров (Piper 4 + Silero 5+random).
- VAD-прегейт (ESPHome `vad:`) — инструментальная музыка → 0 ложных.
- Drawn-out центрирование позитивов + тайм-стретч + рычаг длительности/окна.
- Класс/penalty-веса в безопасном диапазоне (не [60,75]).

## НЕ пробовано (кандидаты)
Knowledge distillation, second-stage verifier, focal/AM-softmax/metric-learning loss,
смена архитектуры (BC-ResNet/TC-ResNet), QAT-тюнинг, on-device персонализация, и —
главное — закрытие train/serve acoustic gap и сбор реальных данных.

## ГЛАВНЫЙ ВОПРОС
Ось «данные/майнинг» выжата, offline-метрики уже сильные. **Где РЕАЛЬНЫЙ остаток улучшения
для боевого деплоя на Voice PE, и какие техники его дают?** Нужна честная оценка: что
двинет полевое качество, а что — диминишинг после уже сделанного; и стоит ли вообще ещё
крутить модель, или ROI сместился в данные/eval/деплой.

## Вопросы исследования (по убыванию ожидаемой пользы; нужны КОНКРЕТНЫЕ, источниковые ответы)
1. **TRAIN/SERVE ACOUSTIC GAP (вероятно, самый недооценённый).** Voice PE прогоняет звук
   через XMOS speech-enhancement + TF4micro noise-suppression/AGC ПЕРЕД моделью. Наши данные
   (TTS + чистые записи) не похожи на то, что реально видит модель на устройстве. Как закрыть
   разрыв: тренировать на PCM, прошедшем тот же NS/AGC; захват реального post-frontend аудио
   с устройства; симуляция канала Voice PE. Что про это известно (microWakeWord/openWakeWord
   форумы, индустрия)? Насколько это бьёт по реальному recall/FA?
2. **FAR-FIELD / РЕАЛЬНАЯ КОМНАТА.** recall на дистанции, реверберация, шумная комната.
   Методы: far-field augmentation (банки RIR, симуляция комнат — pyroomacoustics), запись с
   нескольких дистанций, multi-condition training. Насколько поднимает РЕАЛЬНЫЙ recall vs
   offline-метрика? Конкретные рецепты/датасеты RIR.
3. **SECOND-STAGE VERIFIER** — закрыть no-VAD вокал-музыку и остаточные ложные. Конкретика:
   openWakeWord custom verifier (логрегрессия на эмбеддингах), successive refinement
   (2304.03416). Что реально даёт по FA, как обучить, цена на ESP32-S3 (flash/RAM/латентность).
4. **KNOWLEDGE DISTILLATION**: крупный non-streaming teacher → 32k INT8 student. Рецепты для
   KWS (2207.03331 и свежее 2024–2026), ожидаемый прирост recall/robustness, совместимость с
   mWW pipeline + QAT.
5. **АРХИТЕКТУРЫ/ТРЕНИНГ для tiny streaming KWS (2025–2026 SOTA):** BC-ResNet, TC-ResNet,
   новые модели; loss-функции (focal, AM-softmax/large-margin, contrastive/metric-learning
   эмбеддинги); quantization-aware training. Что реально превосходит mixednet при тех же
   params/arena и стоит ли портировать в mWW.
6. **ON-DEVICE PERSONALIZATION / ADAPTATION** под голоса домочадцев (few-shot, prototype/
   embedding update, on-device fine-tune). Что применимо к TFLM/ESP32-S3 и сколько даёт.
7. **СБОР РЕАЛЬНЫХ ДАННЫХ — единственный оставшийся data-рычаг (по нашим итогам).** Эффективные
   методы: on-device wake-word logging (захват true/false прямо на устройстве и дозаливка),
   crowdsource, augmentation, мимикрирующая канал устройства. Сколько РЕАЛЬНО нужно (часы/
   спикеров) и какой ожидаемый прирост recall на незнакомых голосах.
8. **ОЦЕНКА БОЕВОГО КАЧЕСТВА.** Как построить device-representative eval, который реально
   предскажет полевое поведение (FRR/FAR/FAPH в реальных условиях), а не offline-метрику в
   отрыве от устройства. Бенчмарки, методология, ловушки (leakage, нерепрезентативный тест).

## Формат ответа
Приоритизированный список конкретных, проверяемых шагов с источниками. Для каждого: что
сделать → ожидаемый эффект (на ЧТО именно: far-field recall / вокал-музыка FA / robustness /
персонализация) → цена/риск/усилия при наших ограничениях (INT8 ~32k, CPU-train, mWW pipeline,
фраза фиксирована, Voice PE с XMOS+TF4micro). Явно отметить: (а) что даёт РЕАЛЬНЫЙ прирост в
деплое после уже сделанного, (б) что — диминишинг, (в) сместился ли ROI с «крутить модель» на
«данные/eval/деплой». Цифры/пороги — directional, помечать степень доказательности.
