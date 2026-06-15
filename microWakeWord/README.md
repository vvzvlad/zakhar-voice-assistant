# microWakeWord «захааар» — индекс и текущее состояние

On-device wake word для протяжного **«захааар»** (рус.; короткий «захар» НЕ цель) на
microWakeWord (INT8 streaming TFLite, ESP32-S3 / Home Assistant Voice PE, ESPHome
`micro_wake_word`). Каталог способов и вердиктов — [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md).

> ⚠️ **ЧЕСТНЫЙ РЕ-БЕЙЗЛАЙН 2026-06-15** (полный разбор — [HONEST_EVAL_2026-06-15.md](HONEST_EVAL_2026-06-15.md);
> аудит-предшественник [v27/AUDIT_2026-06-14.md](v27/AUDIT_2026-06-14.md)). Пересобрали leakage-safe eval →
> чистый held-out **всего 16 голосов** (потолок без новых данных). Итоги:
> - реальный device-recall ≈ **60–67%**, и это **полоса по тракту: ~65%** (room-mix) → **~96%** (чистый захват);
> - **деплоибельные модели статистически НЕразличимы** на 16 голосах ни по recall, ни по FAPH;
>   «v27 строго лучший» **НЕ подтверждается** — ничья в шуме, v27 = разумный incumbent;
> - доверять только **radio-FAPH** (остальные классы session-протёкшие/малая выборка);
> - **angular 2-ступень (path B) — мёртв окончательно: 0 из 39 конфигов** не бьют v27 (дыра — silence);
> - различение короткий/протяжный НЕ решено (короткий «захар» палит ~65%, см. [STATE_LEAK_BUG.md](STATE_LEAK_BUG.md)).
> **Единственный блокер всех сравнений — больший bench → идёт fish-раунд** (device-tract diverse-голоса,
> пайплайн [fish_audio/](fish_audio/README.md)). v27 остаётся прод, в репо моделей ничего не залито.

## Деплой: v27
- В проде/[../esphome/zakhar-voice-preroll.yaml](../esphome/zakhar-voice-preroll.yaml) стоит
  **[v27/model/](v27/model/)**, **`probability_cutoff 0.90`** (колено v27), окно = манифест (5), VAD on.
- **v27** = v16-рецепт + **синтетические короткие-«захар» негативы** (лёгкий вес 4/5). Размер 77848 B.
  ⚠️ Прежнее «v27 — небольшое улучшение v16» (recall 21.3→19.6%, music-FAPH 5.8→2.9) на ре-бейзлайне
  **2026-06-15 НЕ подтвердилось**: на чистых 16 голосах v27 и v16 статистически НЕразличимы (Δ recall=0.000,
  FAPH-разница single-fire / в пуассон-шуме). v27 держим как **incumbent**, не как доказанно-лучший —
  [HONEST_EVAL_2026-06-15.md](HONEST_EVAL_2026-06-15.md).
  ⚠️ **Поправка:** прежнее «чинит короткий захар 65→23%» — **артефакт утечки состояния в eval-харнессе**;
  при честном скоринге **обе** модели палят на короткий «захар» ~65%, v27 его НЕ чинит (различение
  короткий/протяжный НЕ решено). Разбор — [STATE_LEAK_BUG.md](STATE_LEAK_BUG.md). Сводка —
  [v27/SUMMARY_real_eval.md](v27/SUMMARY_real_eval.md).
- **Рабочая точка:** 0.90 (FRR 19.3%, FAPH ~1.2/ч с VAD). DET вскрыл, что прежние 0.95 у v16 были
  ПЕРЕзатягиванием (FAPH плато 0.85–0.95). Если в поле полезут ложные в тишине — поднять до **0.95**
  через панель (FRR 22.7%, FAPH 0.86/ч). Колено v16 было 0.85–0.90.
- Путь v8→v11→v8→v16→**v27**: v16 объединил фикс тишины с реальными позитивами; v27 добавил
  short-негативы (чуть лучше recall+музыка; короткий захар НЕ закрыт — был артефакт метрики).
  RepCNN/angular-2-ступень/radio-в-обучение/detection-loss — отброшены.

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
- **B) angular + verifier как on-device 2-ступень — ❌ МЁРТВ (2026-06-15).** 0 из 39 конфигов verifier
  (logreg/GBM/RF/MLP/SVC × silence-boost × сиды) не бьют v27 строго; min silence-FAPH при recall≥.651 =
  0.83 (> v27 0.8). Структурная дыра — **silence**: v19 angular палит на тишине на 1-й ступени (~39/ч),
  эти FA неразделимы в angular-пространстве, 2-я ступень не лечит. Рычаг — резать silence на 1-й ступени
  (переучить angular с silence-негативами) или fish. Разбор — [HONEST_EVAL_2026-06-15.md](HONEST_EVAL_2026-06-15.md).

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

## Путь дальше (стенд-раунд)
1. **Fish-разнообразные позитивы** (сотни голосов, вкл. детские) через устройство → главный
   непробованный рычаг против потолка recall (обобщение на незнакомые голоса). Пайплайн
   (каталог → энтити-дедуп → генерация): [fish_audio/](fish_audio/README.md).
2. **Реальный короткий «захар»** через устройство → hard-негативы (путь A, duration-aware). ⚠️
   синтетические короткие short-fire НЕ снизили (артефакт метрики, [STATE_LEAK_BUG.md](STATE_LEAK_BUG.md)) —
   реальные надо честно ре-тестить; различение короткий/протяжный пока НЕ решено.
3. **Confusables прицельно**: намайнить «закон/закат/зачем/сахар» из СЫРОГО корпуса (15ГБ SOVA, на .128)
   по target-списку .226 → recapture через устройство → hard-негативы (срезать dense-speech FAPH).
4. Открытые device-режимы: barge-in, far-field, шёпот/крик, дети; recall из ДРУГОЙ комнаты
   (текущий eval по одной сессии оптимистичен).

dense-speech radio негативы — УЖЕ записаны (1231/~10.35ч). Прошивочная 2-ступень verifier — отброшена
(на held-out не обобщается, эмбеддинг неразделим — см. реестр).

## Ключевые документы
- **Словарик терминов** (recall/FRR/FAPH/cutoff/…): [GLOSSARY.md](GLOSSARY.md)
- Каталог гипотез + статусы: [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md)
- Раунд v16 (реальные данные): [v16/SUMMARY_real_eval.md](v16/SUMMARY_real_eval.md) ·
  [v16/DURATION_CAUSALITY.md](v16/DURATION_CAUSALITY.md) · [v16/VERIFIER_DEPLOY_SPEC.md](v16/VERIFIER_DEPLOY_SPEC.md)
- ASR-фильтр device-записей: [samples/_asr_filter_v03.py](samples/_asr_filter_v03.py)
- Сводка ранних версий: [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md)
- Захват device-данных: [v8/capture_playback/](v8/capture_playback/), [v9/REALDATA_RUNBOOK.md](v9/REALDATA_RUNBOOK.md)
- Исследования: v8/RESEARCH_2_*, v9/RESEARCH_3/4/5, [v14/RESEARCH_6_result.md](v14/RESEARCH_6_result.md)
