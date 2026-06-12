# microWakeWord «захааар» — индекс и текущее состояние

On-device wake word для протяжного **«захааар»** (рус.; короткий «захар» НЕ цель) на
microWakeWord (INT8 streaming TFLite, ESP32-S3 / Home Assistant Voice PE, ESPHome
`micro_wake_word`). Каталог способов и вердиктов — [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md).

## Деплой: v16
- В проде/[../esphome/zakhar-voice-preroll.yaml](../esphome/zakhar-voice-preroll.yaml) стоит
  **[v16/model/](v16/model/)**, **`probability_cutoff 0.95`** (полевая рабочая точка), окно = манифест (5), VAD on.
- **Полевая поправка порога:** на 0.80 v16 ложил в тишине (слышал слово в тихой комнате) → подняли до **0.95**.
  DET-кривая v16 плоская (recall ~21% на всех порогах, #C) → 0.95 убирает ложные в тишине почти без потери recall.
- **v16** = v8-рецепт + реальные device-tract позитивы + **4 реальных негатив-класса**
  (silence/music/speech/vacuum, записаны через устройство). На честном device-eval (held-out
  по источнику): **FRR 21%, FAPH 0 по всем классам с VAD** → строго лучше v8 (тот давал
  12.5/ч в тишине при том же recall). Чинит полевой баг ложных, recall не теряет.
- Путь v8→v11→v8→v16: v11 убрал ложные в тишине, но его recall просел в поле; временно
  откатывали на v8; v16 объединил фикс тишины с реальными позитивами → текущий прод.
  Сводка раунда — [v16/SUMMARY_real_eval.md](v16/SUMMARY_real_eval.md).

## Главная находка: модель — ОНСЕТ-детектор, не duration-aware
Прямой causality-eval ([v16/DURATION_CAUSALITY.md](v16/DURATION_CAUSALITY.md)): модель опирается
на спектральный **ОНСЕТ «заха»**, а НЕ на длину гласного. Следствия:
- **«захааар» от «захар» по длительности НЕ различается** → модель **палит на короткий «захар»**
  (@1.0с: v16 65%, angular-v19 93%) — обычное имя будит ассистента, и FAPH-eval это не видел.
- **recall хрупкий** (онсет voice/room-специфичен) — это и есть реальные ~21% FRR и просадка в поле.
- **Рычаг recall — angular-голова** (FRR 3-6%), а НЕ объём данных: реальные позитивы recall НЕ
  починили (v16 = 21%, как v8; val 98.7% vs held-out 21% = gap обобщения, BCE запоминает источники).

## Баг тишины — НАЙДЕН и ПОЧИНЕН (v11→v16)
Реальные device-негативы сделали баг ВИДИМЫМ: v8 даёт **12.5 ложных/ч в реальной тишине**
(синтетика давала 0 — корень в реальной текстуре комнаты + PCAN-усиление; VAD маскировал в
проде, но любой просвет протекал). Реальные silence/music/speech/vacuum hard-негативы (v11→v16)
убрали FAPH до 0 по всем классам с VAD. Что НЕ сработало: energy/RMS-гейт и SNR-адаптивный порог
ВРЕДЯТ (ложные AGC-громче слов) — только модельный фикс реальными негативами.

## История версий (вердикты)
| версия | что | вердикт |
|--------|-----|---------|
| [v1](v1) | baseline drawn-out синтетик + 621 real | первая рабочая (eval тёк по uuid) |
| v2 | negative_class_weight 30 | регресс |
| [v3](v3) | крупнее модель + муз-негативы | не побили v1 |
| [v4](v4) | масштаб негативов + confusables + Silero | recall 99% (оптимистичный eval) |
| [v5](v5) | drawn-out центрирование + mining + окно | лучшая до v8; [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md) |
| v6/v7 | adversarial mining 2–3 | плато; [v7/CONTAMINATION_REPORT.md](v7/CONTAMINATION_REPORT.md) |
| [v8](v8) | чистый синтетик + device-капчи + uuid-safe split | была прод; вскрылся silence-FA 12.5/ч |
| [v9](v9) | breadth (distill/QAT/loss/арх) | всё регресс/0 |
| [v10](v10) | TRUE angular-margin head | сильнейший рычаг recall, НО растил music-FA → откладывали |
| [v11](v11) | v8 + реальные silence/music негативы | убрал тишину/музыку, но recall просел в поле |
| [v13](v13) | angular + verifier (запечённый) | офлайн хорош, запекание не переносится |
| v14 | SNR-адаптивный порог | регресс (хуже статичного) |
| v15 | feature-KD (angular→студент) | хуже по обеим осям |
| **[v16](v16)** | v8 + реальные позитивы + 4 реальных негатив-класса | **ПРОД: FRR 21%, FAPH 0 (VAD), строго лучше v8** |
| [v19](v19) | angular + annealing (+ verifier_logreg) | **лучший recall FRR 3%**, но FAPH не single-model; verifier 2-ступень data-limited |
| v21/v22 | short-«захар» негативы (PoC) | v21 (станд.) ВЫУЧИЛ duration-awareness (short 65%→17%); v22 (angular) НЕТ |
| v23 | detection-aware loss (noisy-OR + max-pool) | оба варианта регресс → detection-loss на полку |
| #E абляция | Yandex-клоны: вредят? voice-safe сплит + a/b/c | **КЛОНЫ ПОМОГАЮТ** (real+yandex FRR 21.7% ≫ только-real 34.8%); **УТЕЧКИ НЕТ** (незнакомые голоса = знакомые) → v11b пересмотрен для device-eval |

## Recall: два пути, оба упёрлись в ДАННЫЕ
- **A) Стандартная + РЕАЛЬНЫЕ короткие «захар» негативы** — концепт доказан синтетикой (v21:
  short 65%→17%, цена recall 21→26%); реальные короткие дадут лучше + починят «палит на имя».
- **B) angular + verifier как настоящая on-device 2-ступень** — архитектура решена (dual-output
  `[kws,verify]` + per-trigger AND, [v16/VERIFIER_DEPLOY_SPEC.md](v16/VERIFIER_DEPLOY_SPEC.md)),
  streaming-refit лечит domain-mismatch, НО verifier переобучается на ~98 FA → не генерализуется.
  Нужно СИЛЬНО больше реальных FA (dense-speech негативы).

## Данные (в [samples/](samples/), device-tract)
- `positive_mix_recorded/` — **1220** «захааар» через устройство (3 громкости: v0.3≈397 /
  v0.6≈406 / v1.0≈417), **435 источников = 92 реальных человеческих (`section_*`) + 343 Yandex-TTS**,
  ВСЕ проиграны через колонку и записаны микрофоном (device-tract). ASR-вычитка large-v3 + ручная.
  «mix» в имени = именно эта смесь; Yandex тут — НЕ сырой TTS, а device-tract запись (и она ПОМОГАЕТ, см. #E).
- `negative_silence_recorded/` **382** · `negative_music_recorded/` **145** ·
  `negative_speech_recorded/` **477** (ТВ-речь) · `negative_noise_vacuum_recorded/` **117** — реальные негативы.
- `negative_speech_radio_recorded/` — **~10ч+ dense-speech (SOVA radio/public)**, пишется через
  устройство на 2 громкостях (0.6/1.0) → honest FAPH-eval + FA для 2-ступени.
- `positive_samples_real_people/` (107), `positive_samples_yandex/` (343) — СЫРЫЕ источники
  (не device-tract); сырыми в обучение НЕ идут — используются только их device-tract версии в
  `positive_mix_recorded`. (v11b «синтетика вредит» был на НЕ-device eval; device-tract клоны — наоборот, #E.)
- На ноде .226: чистый STT-набор + device-капчи; артефакты раунда в `~/zakhar-mww/`.

## Путь дальше
1. **Записать реальный короткий «захар»** через устройство → hard-негативы (путь A: duration-aware
   стандартная модель, чинит «палит на короткое имя»).
2. **Дописать dense-speech radio негативы** (идёт) → honest FAPH на плотной речи + разблокировка
   verifier (путь B).
3. Переобуч: стандартная + реальные короткие негативы (#A); повтор 2-ступени с бОльшим FA (#B).
4. Открытые device-режимы: barge-in, far-field, шёпот/крик, дети; recall из ДРУГОЙ комнаты
   (текущий eval по одной сессии оптимистичен).

## Ключевые документы
- Каталог гипотез + статусы: [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md)
- Раунд v16 (реальные данные): [v16/SUMMARY_real_eval.md](v16/SUMMARY_real_eval.md) ·
  [v16/DURATION_CAUSALITY.md](v16/DURATION_CAUSALITY.md) · [v16/VERIFIER_DEPLOY_SPEC.md](v16/VERIFIER_DEPLOY_SPEC.md)
- ASR-фильтр device-записей: [samples/_asr_filter_v03.py](samples/_asr_filter_v03.py)
- Сводка ранних версий: [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md)
- Захват device-данных: [v8/capture_playback/](v8/capture_playback/), [v9/REALDATA_RUNBOOK.md](v9/REALDATA_RUNBOOK.md)
- Исследования: v8/RESEARCH_2_*, v9/RESEARCH_3/4/5, [v14/RESEARCH_6_result.md](v14/RESEARCH_6_result.md)
