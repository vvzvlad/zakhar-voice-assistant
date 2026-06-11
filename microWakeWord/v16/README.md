# v16 — ПРОД-модель «захааар» (v8-рецепт + реальные device-позитивы + 4 реальных негатив-класса)

Одна drop-in модель, **строго лучше v8**. Детали — [SUMMARY_real_eval.md](SUMMARY_real_eval.md),
карта duration — [DURATION_CAUSALITY.md](DURATION_CAUSALITY.md), спека 2-ступени —
[VERIFIER_DEPLOY_SPEC.md](VERIFIER_DEPLOY_SPEC.md), короткий отчёт — [TRAINING_REPORT_v16.md](TRAINING_REPORT_v16.md).

## Что сделано
v8-рецепт **+ реальные device-tract ПОЗИТИВЫ** (1044 «захааар» через устройство, 3 громкости)
**+ 4 реальных негатив-класса** (silence/music/speech/vacuum, записаны через устройство).
leakage-safe split по источнику (все громкости одного клипа в одном фолде).

## Результат (честный device-eval, held-out по источнику, @0.9/win5)
| метрика | v8 | **v16** |
|---------|---:|--------:|
| silence-FAPH (VAD) | 12.5/ч | **0** |
| music/speech/vacuum-FAPH (VAD) | 0 | **0** |
| FRR (recall) | 21.3% | **21.0%** |

Тот же recall, но FAPH 0 по всем классам → чинит полевой баг ложных в тишине без потери детекта.

## Главное (выводы раунда)
- **Модель — ОНСЕТ-детектор, не duration-aware:** опирается на спектр «заха», не на длину
  гласного → ПАЛИТ на короткий «захар» (v16 65% @1.0с). «Только протяжный» без short-негативов
  иллюзорно. См. DURATION_CAUSALITY.
- **Реальные позитивы recall НЕ починили** (21%, как v8): val 98.7% vs held-out 21% — gap
  обобщения, BCE запоминает источники. Рычаг recall — angular-голова, не объём данных.
- **Потолок recall 21% фундаментален** для стандартной модели этого раунда.

## Деплой
Drop-in: `probability_cutoff 0.80`, окно = манифест (5), VAD on. В
[../../esphome/zakhar-voice-preroll.yaml](../../esphome/zakhar-voice-preroll.yaml). INT8 32k, 76 КБ.

## Что НЕ сработало / отложено в эту ночь (см. SUMMARY)
- **angular** (v17/v19, с annealing): лучший recall FRR 3-6%, НО FAPH не давится на уровне модели
  и палит на короткий «захар» → не single-model.
- **2-ступень angular+verifier** (dual-output [kws,verify] + per-trigger AND): архитектура решена,
  streaming-refit лечит domain-mismatch, НО verifier переобучается на ~98 FA → data-limited.
- **short-«захар» негативы (PoC v21):** ВСЕЛЯЮТ duration-awareness в стандартную модель
  (short 65%→17%, цена recall 21→26%) — путь к реальным коротким негативам.
- **detection-aware loss (noisy-OR):** регресс; max-pool — открыт.

Артефакт: `model/zakhar.tflite` (INT8 streaming). Прежняя прод-модель: [../v8](../v8); angular — [../v19](../v19).
