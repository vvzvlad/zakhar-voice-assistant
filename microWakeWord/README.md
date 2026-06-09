# microWakeWord «захааар» — индекс и текущее состояние

On-device wake word для протяжного **«захааар»** (рус.; короткий «захар» НЕ цель) на
microWakeWord (INT8 streaming TFLite, ESP32-S3 / Home Assistant Voice PE, ESPHome
`micro_wake_word`). Каталог способов и вердиктов — [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md).

## Продакшен-модель: v8
- [v8/model/](v8/model/) — `zakhar.tflite` (INT8 ~78 КБ, 5-блочный mixednet 32k).
- Деплой: **`probability_cutoff 0.95`, `sliding_window_size 5`, VAD on** (см.
  [../esphome/zakhar-voice.yaml](../esphome/zakhar-voice.yaml)).
- Метрика (leakage-safe device-eval): on-device FRR ~5%, device-FAPH 0/ч **с VAD**.
- ⚠️ Метрики переоценены коротким измерением (FAPH мерян на ~40 мин — доказывает лишь
  <~4.5/ч; нужно ≥10–15 ч). Без VAD music-FAPH ненулевой. См. METHODS → «Открытые проблемы».

## Открытая проблема (главная)
**Ложные срабатывания на реальную ТИШИНУ/idle-комнату** (и музыку без VAD). Корень — PCAN во
фронтенде micro_speech усиливает низкоэнергетическую текстуру комнаты; синтетический шум баг НЕ
воспроизводит. Лечится **реальными device-записями** (idle-комната + музыка + barge-in) в
негативы — это единственный оставшийся рычаг (модель-сайд исчерпан). Серверный верификатор
**заблокирован**: прошивка не шлёт на сервер аудио самого «захар» (стрим начинается после
детекции и бипа).

## История версий (вердикты)
| версия | что | вердикт |
|--------|-----|---------|
| [v1](v1) | baseline drawn-out синтетик + 621 real | первая рабочая (eval тёк по uuid) |
| v2 | negative_class_weight 30 | регресс, не отгружена |
| [v3](v3) | крупнее модель + муз-негативы (v3a/v3b) | не побили v1 |
| [v4](v4) | масштаб негативов + confusables + Silero | recall 99% |
| [v5](v5) | drawn-out центрирование + mining + окно | была лучшей до v8; сводка [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md) |
| v6/v7 | adversarial mining раунды 2–3 | плато; [v7/CONTAMINATION_REPORT.md](v7/CONTAMINATION_REPORT.md) |
| **[v8](v8)** | чистый синтетик + **реальные device-капчи** + uuid-safe split | **ПРОД, device-FRR 5%** |
| [v9](v9) | breadth-программа (distill/QAT/loss/арх/far-field) | всё регресс/0 → ничего не отгружено |
| [v10](v10) | TRUE angular-margin head | сильнейший рычаг recall, НО растит music-FA → **отложено** (нужны реальные муз-негативы или + verifier) |

## Данные
- [positive_samples_real_people/](positive_samples_real_people/) — реальные «захааар»,
  **дедуп до 107 уникальных** (был 621 = 107 uuid × section-нарезки; повторы → `_duplicates/`).
- [positive_samples_yandex/](positive_samples_yandex/) — **343** Yandex SpeechKit (46 голос/амплуа,
  vowel-held «захааар»), генератор [gen_yandex_positives.py](v9/gen_yandex_positives.py).
- [negative_samples_recorded/](negative_samples_recorded/) — device-tract шум/музыка для FAPH.
- На ноде .226: чистый STT-набор Piper/Silero (10 246) + device-капчи (272).

## Путь дальше
1. Снять **реальные негативы**: idle-комната + музыка + barge-in через устройство (capture).
2. Переобуч v8-рецепта + реальные негативы (снимет тишина-FA) → опц. **angular-head (v10) + real-music**
   или **+ 2nd-stage verifier**.
3. Честные метрики: FAPH ≥10 ч, отдельно без VAD, FRR с N и доверительным интервалом.
- Runbook записи: [v9/REALDATA_RUNBOOK.md](v9/REALDATA_RUNBOOK.md),
  [v8/capture_playback/DEVICE_CAPTURE_PLAYBOOK.md](v8/capture_playback/DEVICE_CAPTURE_PLAYBOOK.md).

## Ключевые документы
- Каталог способов + вердикты: [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md)
- Сводка версий: [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md)
- Загрязнение синтетики: [v7/CONTAMINATION_REPORT.md](v7/CONTAMINATION_REPORT.md)
- Захват device-данных: [v8/capture_playback/](v8/capture_playback/)
- Исследования: v8/RESEARCH_2_*, v9/RESEARCH_3/4/5_result.md
