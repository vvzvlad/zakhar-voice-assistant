# v8 — НОВАЯ РЕКОМЕНДОВАННАЯ модель «захааар» (чистый синтетик + реальные записи с устройства)

Два изменения против v5: (а) пофикшено загрязнение синтетических позитивов, (б) обучение на
РЕАЛЬНЫХ записях, прошедших аудиотракт устройства (playback re-recording). Полный отчёт —
[model/TRAINING_REPORT.md](model/TRAINING_REPORT.md), аудит загрязнения —
[model/CONTAMINATION_AUDIT.md](model/CONTAMINATION_AUDIT.md).

## Вердикт: v8 — основная модель

v8 **решительно бьёт v5 на device-eval** (метрика, предсказывающая поле) и достигает целей
FRR<5% / FAPH<0.5ч; на сухом unified-eval — фактически вровень (число v5 завышено uuid-утечкой).

### Device-eval (реальный тракт: 153 held-out device-позитива + 6 мин device-шума, leakage-safe по uuid)

| cutoff | v5 FRR → **v8 FRR** | v5 recall → **v8 recall** | device-FAPH (с VAD) |
|-------:|---------------------|---------------------------|---------------------|
| 0.80 | 0.170 → **0.046** | 0.830 → **0.954** | 0.00 |
| 0.90 | 0.183 → **0.059** | 0.817 → **0.941** | 0.00 |
| 0.95 | 0.222 → **0.085** | 0.778 → **0.915** | 0.00 |

→ На РЕАЛЬНОМ устройстве v5 пропускал ~18% активаций, **v8 — ~5%**. Device-капчи закрыли
train/serve gap. С VAD device-FAPH = 0 у обоих.

### Unified eval (сухой: 93 real held-out ↔ disjoint MUSAN, «Сахар»)
v5: recall 100%, Сахар 1/90, music-FAPH 1.33 noVAD / 0 VAD. v8: 99%, 2/90, 4.67 noVAD / 0 VAD.
v8 чуть ниже на сухом, НО (1) v5's 100% **завышен uuid-утечкой** (621 «real» — сегменты с
повторяющимися uuid; старый 93-сплит тёк в обучение); (2) v8 тратит ёмкость на устойчивость к
тракту; (3) с VAD оба = music 0. Поле предсказывает device-eval — там v8 явно лучше.

## Что сделано
1. **Фикс загрязнения:** генерация без ломаных написаний (length_scale + тайм-стретч), STT-фильтр
   всех синтетических позитивов (Vosk) → чистые **10 246 / 15 500** (Piper denis/dmitri выброшены,
   набор держат Silero + irina). Старая отравленная синтетика НЕ использована.
2. **Реальные device-капчи:** 272 записи через устройство (post-frontend, XMOS+NS/AGC) → 119 в
   обучение с **повышенным весом 12** (real-tract золото) + 14 device-шумов как hard-negatives.
3. **Far-field** RIR-аугментация; рецепт в остальном = v5 (drawn-out центр, окно-рычаг).
4. **Leakage-safe сплит по uuid/источнику** — устранена утечка из старого реального сплита.

## Как ставить
`probability_cutoff 0.80` / `sliding_window_size 3`, **VAD обязателен** (device FRR 4.6%,
device-FAPH 0). Манифест v2 с валидным `website`/`author` (пустой website роняет ESPHome на
`cv.url`). `tensor_arena 45000`. См. [model/esphome_micro_wake_word.yaml](model/esphome_micro_wake_word.yaml);
в [../../esphome/zakhar-voice.yaml](../../esphome/zakhar-voice.yaml) модель уже переключена на v8.

## Известные находки
- **uuid-утечка** в легаси-сплите реальных: метрики v1–v7 на «93 real held-out» были оптимистичны.
  Честная оценка — device-eval (leakage-safe).
- Главный рычаг recall дальше — больше реальных device-записей (разные люди/комнаты/дистанции).

Артефакт: `model/zakhar.tflite` (INT8 streaming, 77.8 КБ). Прежние: [../v5](../v5), [../v4](../v4),
[../v1](../v1). Разбор загрязнения старых версий — [../v7/CONTAMINATION_REPORT.md](../v7/CONTAMINATION_REPORT.md).
Сводка всех версий — [../v5/SUMMARY_ALL_VERSIONS.md](../v5/SUMMARY_ALL_VERSIONS.md).
