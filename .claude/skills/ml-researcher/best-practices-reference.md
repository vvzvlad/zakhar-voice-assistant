# Reference: строгая методология ML-экспериментов и оценки (sourced)

> Результат независимого ресёрча по best-practices (small-data / on-device audio / KWS). Полные
> формулировки do/don't с источниками. Краткая actionable-версия — в `SKILL.md` (там же ⚑-пометки,
> где конкретное правило ранее ловило баг в этом проекте).

## TL;DR

- **Главное правило честной оценки small-data audio/KWS:** сначала исключите утечки (speaker/session/source-disjoint split + post-hoc near-duplicate аудит + adversarial validation), затем сообщайте каждую метрику с интервалом неопределённости (Wilson для пропорций recall/FRR, Poisson для редких FA/час) и сравнивайте модели только парными тестами (McNemar / paired bootstrap) — «too good» метрика почти всегда означает leakage, а не успех.
- **Для on-device / streaming специфики:** оценивайте stateful-модель в том же streaming-режиме (chunk/lookahead/state-reset), в котором деплоите; репортите FRR @ фиксированный FA/час на большом held-out корпусе негативов с разделением по средам; и обязательно валидируйте на реальной перезаписи через целевое устройство, а не только на синтетике (RIR/noise аугментация закрывает gap лишь частично).
- **Что реально двигает recall-ceiling на unseen speakers:** число РАЗНЫХ дикторов (ошибка падает примерно логарифмически), margin-based loss вместо vanilla softmax (≈25–30 % относительного снижения EER, но чувствителен к настройке margin/scale), и аугментация (noise/RIR/SpecAugment).

## 1. Leakage-safe evaluation
- **DO speaker-disjoint split** — все utterances диктора в одной партиции. Проверка: `set(spk_train)∩set(spk_test)==∅`. Источник: COVYT arXiv:2206.11045 §2.4; SynTTS-Commands arXiv:2511.07821.
- **DO session/source-disjoint** — одна запись/сессия/источник не пересекает границу. Источник: COVYT (file-disjoint схема).
- **DO post-hoc near-duplicate детекция** через аудио-эмбеддинги/хэши train↔test. Источник: MLMastery «Data Leakage in ML».
- **DON'T нормализовать/отбирать фичи до split** — статистики только на train (sklearn Pipeline). Источник: scikit-learn pipelines.

## 2. Offline eval of streaming/stateful models
- **DON'T давать hidden state течь между независимыми примерами** — сброс между utterances. Без сброса у Google «up to 2x accuracy reduction». Проверка: метрики при reset/preserve/shuffle. Источник: Rybakov et al. arXiv:2005.06720; arXiv:1707.07240.
- **DO оценивать в деплой-streaming-режиме** (chunk/lookahead/left-context). Источник: NVIDIA NeMo cache-aware (`compare_vs_offline`).
- **DO cache-aware, не буферный пересчёт.** Источник: arXiv:2312.17279.
- **DO моделировать state-passing и на train, если оно нужно** (RSS/RSP). Источник: arXiv:1910.11455.

## 3. Honest metrics on small samples
- **DO Wilson CI для пропорций** (recall/FRR/accuracy) — Wald некорректен при малом n / p→0/1. `statsmodels…proportion_confint(method='wilson')`. Источник: arXiv:2503.01747; MLMastery.
- **DO Poisson CI для редких событий** (FA/час): SE≈√N/T, exact через χ². Источник: StatsDirect; arXiv:2406.10162.
- **DON'T заявлять разницу при пересекающихся CI** — парный тест: McNemar (`mlxtend`, exact при b+c<25), paired bootstrap; Бонферрони при множественных. Источник: Dietterich 1998; mlxtend; Berg-Kirkpatrick 2012.
- **DO bootstrap ≥1000–10000 ресэмплов, percentile/BCa**, фикс seed. Источник: arXiv:2603.28769.

## 4. False-accept standards (FAPH)
- **DO репортить FRR ВСЕГДА с FA/час** при фикс. рабочей точке («%» как единица FAR недопустима). Источник: Picovoice/Porcupine.
- **DO большой held-out корпус реального непрерывного аудио**; FA/час с Poisson CI. Источник: Higuchi et al. arXiv:2107.07634; U2-KWS arXiv:2312.09760 (60ч негативов).
- **DO environment-held-out негативы** (quiet/noise/music/TV отдельно). Источник: arXiv:2304.09585; micro-wake-word.
- **DO сравнивать при ФИКСИРОВАННОМ FA-rate** (напр. 1 FA/10ч). Источник: Picovoice wake-word-benchmark.
- **DON'T считать мульти-срабатывания в одном сегменте за отдельные TA** — latency-window + lockout. Источник: arXiv:1705.02411.

## 5. Train/serve domain gap (on-device audio)
- **DO RIR-реверберация + инъекция шума** (far-field). Источник: FAST-RIR arXiv:2110.04057; IR-GAN arXiv:2010.13219.
- **DO валидировать на РЕАЛЬНОЙ перезаписи через устройство** (синтетика gap не закрывает). Источник: AV-RIR arXiv:2312.00834; Treble10 arXiv:2510.23141.
- **DO учитывать фронтенд устройства (AEC/AGC/denoise)** — прогонять аугментации через тот же стек. Источник: arXiv:1808.00563; arXiv:1607.05666.
- **DO multi-condition training.** Источник: arXiv:1808.00563.

## 6. Dataset validation (ASR/STT, forced alignment)
- **DO `mfa validate` ПЕРЕД alignment** — ловит OOV/рассогласование. Источник: Montreal Forced Aligner docs.
- **DO фильтровать по ASR/alignment confidence** (low score / большой retry-beam → drop). Источник: MFA beam; arXiv:2204.06322.
- **DO валидировать метки на стадии датасета, не после обучения.** Источник: PMC «Data leakage detection in ML code».
- **DON'T доверять G2P для keyword без ручной проверки.** Источник: MFA workflow guides.

## 7. Adversarial / red-team verification
- **DO считать «too good» метрику сигналом leakage** (резкий test→prod дроп = классика). Источник: MLMastery; TDS.
- **DO adversarial validation train↔test** — классификатор train-vs-test, цель AUC≈0.5; feature-importance вскрывает протекающую фичу. Источник: FastML; Zak's Blog; KDnuggets.
- **DO ablation/sanity (shuffle labels, remove suspicious feature).** Источник: Tonic.ai.
- **DO проверять train претрейн-модели на пересечение с твоим test** (model leakage). Источник: Kaggle Handbook.

## 8. Generalization / recall ceiling (KWS, unseen speakers)
- **DO растить число РАЗНЫХ дикторов** — EER ~логарифмически: 50→5.19%, 500→1.87%, 1500→1.15%, 3000→0.90%. Источник: Sztahó et al. arXiv:2204.06450; VoxCeleb2 arXiv:1806.05622.
- **DO margin-loss (AAM-softmax/ArcFace) вместо vanilla softmax** — ~25–30% сниж. EER. Источник: «Margin Matters» arXiv:1906.07317.
- **DON'T считать AAM-softmax «бесплатным»** — чувствителен к m/s, при типичной настройке уступал triplet. Источник: «In Defence of Metric Learning» arXiv:2003.11982.
- **DO augmentation (MUSAN/RIR/SpecAugment)** — supervised эмбеддинги её эффективно эксплуатируют. Источник: X-Vectors; ECAPA-TDNN arXiv:2005.07143.
- **DO таргетированные синтетические негативы для confusables** (vowel-group метрика). Источник: LLM-Synth4KWS arXiv:2505.22995.

## 9. Duration / prosody-aware KWS
- **DO duration/prosody-фичи** для слов, схожих по onset. Источник: ProKWS arXiv:2603.18024; DMA-KWS arXiv:2605.22120.
- **DO duration-penalty** против FA на аномально длинных фонемах. Источник: US Patent 9672815.
- **DON'T игнорировать train/inference консистентность длительности.** Источник: LLM-Synth4KWS arXiv:2505.22995.
- **DO prosody-modification как аугментацию** (особенно детская речь). Источник: ScienceDirect S1051200418303713.

## 10. Reproducibility
- **DO фиксировать ВСЕ seeds + детерминизм** (random/np/torch/cuda, cudnn.deterministic, PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, DataLoader worker_init_fn). Источник: PyTorch Reproducibility docs.
- **DO трекинг (MLflow/W&B): code(git) + data(версия+хэш) + params + metrics + env.** Источник: W&B; TechSaaS.
- **DO версионировать данные/манифесты (DVC), хэш данных как MLflow-параметр.** Источник: APXML; DagsHub.
- **DO пинить окружение (requirements/conda/Docker) как артефакт.** Источник: APXML; PyTorch docs.
- **DO pre-ship чек-лист (NeurIPS-style):** compute-тип/объём, error bars, доступность кода/данных. Источник: NeurIPS Paper Checklist; JMLR v22/20-303.

## Caveats (важно)
- Часть how-to — инженерные блоги (не peer-reviewed); числовые утверждения подкреплены статьями.
- margin-loss: ~30% над *vanilla softmax*, но при типичной настройке может уступать *triplet* — разные базлайны, не противоречие; подбирайте m/s.
- Изолированный эффект объёма данных на ОДНОГО диктора в источниках чисто не выделен (пробел).
- Лог-зависимость EER от числа дикторов (Sztahó) — основной домен патологическая речь; тренд согласуется с VoxCeleb2, но числа — иллюстрация направления.
- Вендорские числа FAPH/EER (Picovoice и пр.) — контекстные, НЕ переносить как целевые без своего замера.
- Детерминизм PyTorch неполон (atomicAdd, некоторые pooling/ctc_loss недетерминированы; CPU↔GPU не гарантирован).
