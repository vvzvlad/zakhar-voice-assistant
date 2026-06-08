# Сводка ВСЕХ версий «захааар» — ЕДИНЫЙ eval (все цифры сопоставимы)

Все строки ПЕРЕОЦЕНЕНЫ заново на ОДНОМ eval (v5/unified_eval.py): 93 РЕАЛЬНЫЕ протяжные held-out (recall), чистая disjoint 90-мин MUSAN-музыка (music-FAPH, Silero-VAD гейт), speech-neg «Сахар». Окно 3, cutoff 0.9. → числа в одной шкале, прямо сопоставимы.

> Замечание: ранние отчёты v2/v3 мерялись на РАННЕМ муз-пробнике (40 мин fma, возможна утечка) и НЕ были сопоставимы; цифры ниже их заменяют (свежий прогон на чистой disjoint-музыке).

> recall@0.9/win3 в этом eval = доля held-out клипов с max(скольз.ср.окна3)≥0.9 (та же величина, что «drawn-out recall %≥0.9»).

| версия | где | что менялось | drawn-out recall %≥0.9 (=rec@0.9/win3) | mean prob | «Сахар» fires/90 | music-FAPH noVAD @0.9/win3 | music-FAPH С VAD |
|--------|-----|--------------|------------------------------------------|-----------|------------------|----------------------------|------------------|
| v1 | output/ | baseline (mixed synth+real, neg wt5) | 95% | 0.981 | 2 | 7.33 | 0.0 |
| v2 | не отгружена | neg_class_weight 30 (over-conservative) | 88% | 0.938 | 0 | 4.67 | 0.0 |
| v3a | не отгр. (output_v3/ = v3b) | dedicated fma_medium music neg wt18, 26k model | 76% | 0.906 | 1 | 0.0 | 0.0 |
| v3b | output_v3/ | bigger 5-block model + rebalance | 94% | 0.973 | 1 | 1.33 | 0.0 |
| v4 | output_v4/ | scaled neg+MUSAN+ESC50+confusables+Silero, music-aware sel. | 99% | 0.996 | 4 | 1.33 | 0.0 |
| **v5** | output_v5/ ← ЛУЧШАЯ | drawn-out centering + round-1 mining + duration lever | 100% | 0.999 | 1 | 1.33 | 0.0 |
| v6 | не отгр. (регресс) | + round-2 mined penalty10, 80k (regression) | 99% | 0.998 | 1 | 6.0 | 0.0 |
| v7 | не отгр. (нет выигрыша) | + round-2 mined penalty4, 45k (no gain) | 99% | 0.997 | 1 | 1.33 | 0.0 |

## Вывод
- **v5 — лучшая**: drawn-out recall 100% (mean 0.999), «Сахар» 1/90, music-FAPH 1.33 без VAD и **0.00 с VAD**. Строго бьёт v4 в деплое (VAD on).
- Прогрессия: v1 (95%) → v2 (88%, neg_class_weight 30 убил recall) → v3a (76%, тяжёлая музыка вес18 пере-консервативна) → v3b (94%, крупнее модель) → v4 (99%, масштаб негативов+confusables+Silero) → **v5 (100%, drawn-out центрирование + майнинг + окно-рычаг)**.
- v6/v7 (раунды 2–3 майнинга) НЕ улучшили v5 (v6 регресс по music-FAPH 6.0; v7 без выигрыша). Остаток (no-VAD вокал-музыка) майнингом не берётся без потери recall и закрыт VAD-гейтом (music-FAPH с VAD = 0 у всех).
- ГЛАВНОЕ для деплоя: любая версия С VAD даёт 0 ложных на музыке; решает recall на протяжном слове, и тут v5 — 100%.