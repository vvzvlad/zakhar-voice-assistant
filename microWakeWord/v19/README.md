# v19 — angular-margin (с annealing) + verifier_logreg. Лучший RECALL, НЕ single-model.

Recall-фронтир раунда. Короткий отчёт — [TRAINING_REPORT_v19.md](TRAINING_REPORT_v19.md);
деплой 2-ступени — [../v16/VERIFIER_DEPLOY_SPEC.md](../v16/VERIFIER_DEPLOY_SPEC.md); сводка —
[../v16/SUMMARY_real_eval.md](../v16/SUMMARY_real_eval.md). НЕ в проде (прод — [../v16](../v16)).

## Что сделано
angular-margin head (AM-Softmax) на реальных device-данных + **annealing** (softmax warmup +
margin 0→0.2 — ОБЯЗАТЕЛЕН на ~32k модели, без него дивергенция и ложный вывод «angular хуже»).
Тяжёлый штраф на негативы (v19). Плюс `verifier_logreg.npz` — 2-я ступень на эмбеддинге.

## Результат (честный device-eval, @0.9/win5)
- **FRR 3.0%** — лучший recall из всех (v16 = 21%). Рычаг recall = angular-голова.
- НО FAPH single-model высокий (silence/speech не давятся: penalty 8/15, cutoff-сатурация,
  запекание verifier — всё провалилось). И **палит на короткий «захар»** (93% @1.0с) — margin
  перебивает short-негативы (duration-awareness в angular НЕ вселяется).

## 2-ступень (verifier) — статус
dual-output `[kws, verify]` (--angular_head 4) + per-trigger AND — правильная форма (не per-frame
произведение). streaming-refit verifier на streaming-int8 эмбеддингах ЛЕЧИТ domain-mismatch
(verify на ложных 0.65-0.99→0.01-0.06). **НО** verifier переобучается на ~98 реальных FA → на
held-out режет лишь ~25% (офлайн 3.8%/0.85 оптимистичны). **Нужно сильно больше реальных FA**
(dense-speech негативы) → тогда путь B оживёт. Контракт/хост-референс — в VERIFIER_DEPLOY_SPEC.

## Вывод
v19 не отгружается как есть. Скачок recall — либо этот angular + verifier с бОльшим FA-набором
(путь B), либо duration-aware СТАНДАРТНАЯ модель + реальные короткие «захар» негативы (путь A, см.
PoC v21). Артефакты: `zakhar.tflite` (angular, INT8), `verifier_logreg.npz`.
