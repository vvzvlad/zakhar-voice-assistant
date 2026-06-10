# microWakeWord «захааар» — индекс и текущее состояние

On-device wake word для протяжного **«захааар»** (рус.; короткий «захар» НЕ цель) на
microWakeWord (INT8 streaming TFLite, ESP32-S3 / Home Assistant Voice PE, ESPHome
`micro_wake_word`). Каталог способов и вердиктов — [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md).

## Рекомендованная модель: v11 (прод пока v8 — переключить)
- [v11/model/](v11/model/) — `zakhar.tflite` (INT8). v8-рецепт **+ реальные device-tract негативы**
  (тишина ~166 мин + музыка ~38 мин). Drop-in, бьёт v8.
- Деплой: **`probability_cutoff 0.9`, `sliding_window_size 5`, VAD on**. В
  [../esphome/zakhar-voice.yaml](../esphome/zakhar-voice.yaml) пока стоит v8 — переключить на `v11/model`.
- Результат (реальный device-eval, БЕЗ VAD): silence-FAPH **12.5→0.8/ч**, music **9.3→0/ч**,
  FRR на реальных людях **4.7→0.9%** (dev_heldout FRR 8.5→12.4% — единственная цена).

## Баг тишины — НАЙДЕН и ПОЧИНЕН (v11)
Реальные device-негативы сделали баг ВИДИМЫМ: v8 даёт **12.5 ложных/ч в реальной тишине/idle-комнате**
(синтетика давала 0 — корень в реальной текстуре комнаты, PCAN-усиление; VAD это маскировал в проде,
но любой просвет протекал). **v11** (реальные silence/music hard-негативы) убрал тишину до 0.8/ч и
музыку до 0, и улучшил FRR на живых голосах. Сводка — [v11/SUMMARY_real_eval.md](v11/SUMMARY_real_eval.md).
Что НЕ сработало: energy/RMS-гейт и SNR-адаптивный порог ВРЕДЯТ (ложные AGC-громче слов); серверный
верификатор заблокирован (прошивка не шлёт аудио «захар»); только модельный фикс реальными негативами.

## История версий (вердикты)
| версия | что | вердикт |
|--------|-----|---------|
| [v1](v1) | baseline drawn-out синтетик + 621 real | первая рабочая (eval тёк по uuid) |
| v2 | negative_class_weight 30 | регресс, не отгружена |
| [v3](v3) | крупнее модель + муз-негативы (v3a/v3b) | не побили v1 |
| [v4](v4) | масштаб негативов + confusables + Silero | recall 99% |
| [v5](v5) | drawn-out центрирование + mining + окно | была лучшей до v8; сводка [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md) |
| v6/v7 | adversarial mining раунды 2–3 | плато; [v7/CONTAMINATION_REPORT.md](v7/CONTAMINATION_REPORT.md) |
| [v8](v8) | чистый синтетик + реальные device-капчи + uuid-safe split | была прод; вскрылось: реальный silence-FA **12.5/ч** (маскировал VAD) |
| [v9](v9) | breadth-программа (distill/QAT/loss/арх/far-field) | всё регресс/0 → ничего не отгружено |
| [v10](v10) | TRUE angular-margin head | сильнейший рычаг recall, НО растит music-FA → отложено |
| **[v11](v11)** | v8 + **РЕАЛЬНЫЕ silence/music негативы** | **РЕКОМЕНД.: silence 12.5→0.8/ч, music 9.3→0, real-FRR 4.7→0.9%** |
| [v13](v13) | angular + 2-ступ. verifier | офлайн лучший фронтир, но запечённый деплой НЕ бьёт v11 (silence 6.7/ч) |

## Данные (в [samples/](samples/))
- `samples/negative_silence/` — **382 файла / ~4 ч** реальной тишины/idle через устройство (фикс бага).
- `samples/negative_music/` — **59 / ~55 мин** реальной музыки через устройство.
- `samples/positive_samples_real_people/` — **107** реальных «захааар» (дедуп; был 621 = 107 uuid).
- `samples/positive_samples_yandex/` — **343** Yandex (vowel-held); генератор [v9/gen_yandex_positives.py](v9/gen_yandex_positives.py).
  ⚠️ В v11b добавление этих yandex-позитивов УХУДШИЛО real-FRR (6.5%) — синтетика навредила.
- На ноде .226: чистый STT-набор Piper/Silero (10 246) + device-капчи (272).

## Путь дальше
1. **Переключить ESPHome v8 → v11/model** (0.9/win5/VAD) — фикс полевого бага тишины.
2. Реальные **позитивы через устройство** (`positive_samples_real_people_recorded`) — завтра, агент сам
   наиграет/запишет колонкой. Восстановит dev-FRR и обобщение на незнакомые голоса.
3. Далее (живая запись через устройство): **barge-in, far-field, шёпот/крик, дети** — открытые режимы.
- Runbook записи: [v9/REALDATA_RUNBOOK.md](v9/REALDATA_RUNBOOK.md),
  [v8/capture_playback/DEVICE_CAPTURE_PLAYBOOK.md](v8/capture_playback/DEVICE_CAPTURE_PLAYBOOK.md).

## Ключевые документы
- Каталог способов + вердикты: [v10/HYPOTHESIS_REGISTRY.md](v10/HYPOTHESIS_REGISTRY.md)
- Сводка версий: [v5/SUMMARY_ALL_VERSIONS.md](v5/SUMMARY_ALL_VERSIONS.md)
- Загрязнение синтетики: [v7/CONTAMINATION_REPORT.md](v7/CONTAMINATION_REPORT.md)
- Захват device-данных: [v8/capture_playback/](v8/capture_playback/)
- Исследования: v8/RESEARCH_2_*, v9/RESEARCH_3/4/5_result.md
