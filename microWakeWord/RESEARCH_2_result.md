# Куда реально двигать качество on-device wake-word «захааар» на Voice PE, когда ось data/mining исчерпана

## TL;DR
- **Главный незакрытый рычаг — train/serve acoustic gap.** Официальные модели microWakeWord (okay_nabu/hey_jarvis) обучены на синтетике Piper + augmentation openWakeWord + публичных фоновых корпусах, а **не** на аудио, прошедшем XMOS XU316 + TF4micro. Ваши TTS+чистые записи не похожи на post-frontend поток, который модель реально видит на устройстве. Именно здесь (плюс far-field/реальная комната) находится основной прирост FIELD-качества — не в дальнейшем тюнинге архитектуры.
- **ROI сместился с «тюнинга модели» на «data/eval/deployment».** При ваших оффлайн-метриках (recall 100% клипов ≥0.9, музыка 0 FA/час с VAD) смена архитектуры (BC-ResNet/TC-ResNet), distillation и новые loss-функции — это diminishing returns под жёсткими ограничениями ~32k INT8 / 45 KB arena / CPU-train / fixed phrase. Реальный прирост дадут: (1) сбор реального post-frontend аудио с устройства, (2) far-field RIR-augmentation, (3) device-representative eval, (4) опционально second-stage verifier для остаточных vocal-music FA без VAD.
- **Самое дешёвое и доказанное.** Реальные голоса вместо синтетики снизили false-reject у Home Assistant с 18% до 5%; реалистичная room-simulation давала до 35.8% relative improvement, а diffuse acoustic simulation — +21% recall в far-field KWS. Эти эффекты перебивают любой выигрыш от смены loss-функции или архитектуры под вашими ограничениями.

---

## Key Findings (приоритезированный список)

### Приоритет 1 — Закрыть train/serve acoustic gap (самый недооценённый рычаг) — РЕАЛЬНЫЙ прирост
**Что делать:** обучать модель на PCM, прошедшем тот же NS/AGC-тракт, что и на устройстве. Voice PE прогоняет аудио через XMOS XU316 (echo cancellation, stationary noise removal, auto gain control) и TF4micro **до** того, как модель его увидит. Документация ESPHome/DeepWiki прямо подтверждает: *«The processed audio from the XMOS pipeline feeds into the micro wake word detection system»* (DeepWiki, esphome/home-assistant-voice-pe, раздел Audio Processing Pipeline, со ссылками на home-assistant-voice.yaml).

Официальные модели обучены иначе — по словам Kevin Ahrendt (kevinahrendt.com/micro-wake-word): *«We generate wake word samples using Piper sample generator. We then augment them using openWakeWord's utilities. We use additional data sources for negative samples»*. DiPCo используется **только как held-out тест** на FAPH, а не как обучающие данные. Значит, gap реален и в стоковом пайплайне не закрыт.

**Ожидаемый эффект (на что именно):** real recall и FA в полевых условиях. Доказательство силы эффекта — Amazon, «On Front-end Gain Invariant Modeling for Wake Word Spotting» (Gao et al., arXiv 2010.06676, Interspeech 2020, wake-word «Alexa»): *«when AFE gain changes up to ±12dB, the baseline CNN model lost up to relative 19.0% in false alarm rate or 34.3% in false reject rate, while the model with ΔLFBE demonstrates no performance loss»*. Несоответствие фронтенда обучения/развёртывания бьёт по реальным метрикам в разы сильнее, чем шум в оффлайн-валидации.

**Стоимость/риск/усилия под вашими ограничениями:** низко-средняя. Прямой путь — прогонять обучающее аудио (позитивы и негативы) через эквивалент NS/AGC до извлечения спектрограмм, и/или собирать post-frontend поток с устройства (см. Приоритет 7). Сильная сторона: ничего не меняется в ~32k INT8-архитектуре, только распределение обучающих данных. Альтернатива из 2010.06676 — ΔLFBE-фичи (дельта по времени поверх log-filterbank), но это требует модификации препроцессора и менее совместимо с фиксированным micro_speech-фронтендом mWW (40-band, feature_step_size 10) — оставьте как запасной вариант.

### Приоритет 2 — Far-field / реальная комната (RIR augmentation + multi-condition) — РЕАЛЬНЫЙ прирост
**Что делать:** convolve позитивы с банками RIR и добавлять шум на реалистичных SNR (multi-condition training). Конкретные ресурсы:
- **OpenSLR SLR28 (RIRS_NOISES)** — 325 реальных RIR (RWCP + REVERB-2014 + Aachen AIR) + симулированные RIR + point-source шумы, всё 16 kHz/16-bit; готовый Kaldi-рецепт `augment_data_dir.py` (openslr.org/28).
- **BUT ReverbDB** (arXiv 1811.06795) — 1209 реальных RIR; вывод авторов: *«a limited number of real RIRs, carefully selected to match the target environment, provide results comparable to a large number of artificially generated RIRs, and that both sets can be combined to achieve the best ASR results»*.
- **pyroomacoustics** — для синтетических RIR на CPU (идеально под ваше CPU-only обучение).

**Ожидаемый эффект (на что именно):** far-field recall и noisy-room recall — то, что ваш текущий оффлайн-eval (чистые held-out позитивы) почти не измеряет.
- «A Study on More Realistic Room Simulation for Far-Field Keyword Spotting» (arXiv 2006.02774, APSIPA 2020, задача wake-word «Hey Snips»): *«On a hold-out set of re-recordings under clean and noisy far-field conditions, we demonstrate up to 35.8% relative improvement over the commonly-used (single absorption coefficient) image source method»*; код выпущен в Pyroomacoustics.
- «Improving Reverberant Speech Training Using Diffuse Acoustic Simulation» (arXiv 1907.03988): *«significant performance improvement on real test sets in far-field speech recognition by 1.58% and keyword spotting by 21%, without fine-tuning using real impulse responses»*.
- «Towards Data-efficient Modeling for Wake Word Spotting» (arXiv 2010.06659 — та же работа, что вы использовали для adversarial mining): multi-condition pipeline со stratified augmentation, стартуя с 10 часов domain-mismatched аудио, расширил датасет в 20-100 раз и достиг качества production-grade модели (обученной на 375 ч far-field данных), сэкономив 97% сбора WW-данных.

**Стоимость/риск/усилия:** низкая. Чисто data-augmentation на CPU, ничего не меняется в модели/arena. Риск: переусердствовать с реверберацией и просадить near-field recall — балансируйте долю «сухих» примеров. Это **самый высокий ROI после Приоритета 1** и применимо немедленно.

### Приоритет 3 — Second-stage verifier (для остаточных vocal-music FA без VAD) — РЕАЛЬНЫЙ, но узкий прирост
**Что делать:** добавить лёгкий второй каскад, срабатывающий только после первичного триггера. Два проверенных подхода:
- **openWakeWord custom verifier** — *«simple logistic regression binary classifiers»* поверх shared audio embeddings, тренируется на очень малом объёме данных и фильтрует активации по характеристикам целевого спикера/окружения. Из документации: *«this can be a very effective way at reducing false activations, as the model will be more focused on a the target speaker instead of attempting to activate for any speaker»*.
- **Successive Refinement** — «To Wake-up or Not to Wake-up: Reducing Keyword False Alarm by Successive Refinement» (arXiv 2304.03416, ICASSP 2023): каскад speech→keyword-like→keyword; *«across multiple models with size ranging from 13K parameters to 2.41M parameters, the successive refinement technique reduces FA by up to a factor»* (вплоть до кратного снижения).

**Ожидаемый эффект (на что именно):** именно vocal-music FA, который у вас остаётся без VAD (2.67/час; с VAD уже 0). Если решение работать без VAD не стоит — приоритет падает.

**Стоимость/риск/усилия под ESP32-S3:** logistic-regression verifier практически бесплатен по flash/RAM/latency (несколько КБ, доли мс). НО: openWakeWord-verifier тренируется поверх **embedding-модели openWakeWord**, которой нет в mWW-пайплайне — потребуется либо порт verifier поверх ваших mWW-фичей, либо отдельная крошечная INT8-голова. Риск: дополнительный tensor_arena. Делать, только если хотите убрать VAD; иначе VAD уже даёт 0 FA/час на инструментальной музыке.

### Приоритет 4 — Knowledge distillation (умеренная отдача) — частично diminishing returns
**Что делать:** обучить большой non-streaming teacher и дистиллировать в ваш 32k INT8 student. Рецепты: «Low-resource Low-footprint Wake-word Detection using Knowledge Distillation» (arXiv 2207.03331); обзор «Advances in Small-Footprint Keyword Spotting» (arXiv 2506.11169); для робастности — «VIC-KD» (arXiv 2309.12914), улучшение robust-distillation на 8-12% против ARD/RSLAD на Google Speech Commands.

**Ожидаемый эффект:** небольшой прирост recall/robustness при том же бюджете параметров. Ключевое совместимое свойство (из 2506.11169): *«these methods are applied exclusively during the training phase, leaving the inference process unaffected»* — то есть KD не трогает ваш INT8 32k inference и совместим с QAT.

**Стоимость/риск/усилия:** средняя. Обучение teacher на CPU-only дольше, но выполнимо (плюс времени, не GPU). Риск: при уже-насыщенных оффлайн-метриках выигрыш мал относительно усилий; делать **после** Приоритетов 1-2, когда данные уже device-representative — тогда teacher учит student именно полезному инварианту.

### Приоритет 5 — Архитектуры и loss-функции — DIMINISHING RETURNS под вашими ограничениями
- **BC-ResNet** (arXiv 2106.04140, Qualcomm): *«The smallest BC-ResNet-1 achieves 96.6% accuracy with less than 10k parameters»*, SOTA на Google Speech Commands (98.0%/98.7% v1/v2 у BC-ResNet-8). НО это **closed-set classification на 1-сек клипах**, не streaming wake-word; порт в mWW (mixednet + streaming inference + QAT) — большой инженерный риск ради неясного полевого выигрыша.
- **TC-ResNet** (arXiv 1904.03814): эффективные 1D temporal convolutions, treat frequency как каналы; хорош по footprint, но та же проблема портирования.
- **AM-Softmax / metric learning** (arXiv 2211.00439, «Metric Learning for User-defined Keyword Spotting»): margin-based loss даёт более дискриминативные embeddings и лучше разделяет классы при низких false-alarm — теоретически помогает против «сахар»/confusables. Под fixed-phrase это самый осмысленный из «модельных» рычагов, но прирост маргинальный поверх ваших penalty-весов.
- **QAT**: «Fixed-point quantization aware training for on-device keyword-spotting» (arXiv 2303.02284, Amazon): *«on an in-house KWS dataset … our 8bit FXP-QAT models have a 4-6% improvement in relative false discovery rate at fixed false reject rate compared to full precision FLP models»* и снижение времени исполнения на 68%. Это **единственная «модельная» техника с конкретным сорсед-числом и низким риском** — QAT прямо совместим с TFLite-микро INT8 и mWW. TensorFlow Model Optimization Toolkit предоставляет `quantize_model`.

**Вердикт по Приоритету 5:** из всего блока стоит тратить силы только на **QAT-тюнинг** (низкий риск, +4-6% rel. FDR) и, опционально, AM-softmax против «сахар». Смену backbone (BC/TC-ResNet) — **не делать**: ROI отрицательный под ~32k/45KB/CPU/fixed-phrase.

### Приоритет 6 — On-device personalization (нишевый для fixed-phrase) — ограниченная применимость
**Что делать:** few-shot prototype/embedding-адаптация под голоса домохозяйства. «Few-Shot Open-Set Learning for On-Device Customization of KWS» (arXiv 2306.02161, KU Leuven): deep feature encoder + prototype-based классификатор; *«an accuracy of up to 76% in a 10-shot scenario while the false acceptance rate of unknown data is kept to 5%»*; triplet loss с нормализованными выходами бьёт prototypical networks; авторы отмечают, что scores «lossless с точностью до 8-bit quantization».

**Ожидаемый эффект:** прирост recall на знакомых голосах семьи. НО: ваша задача — **fixed phrase для любого спикера**, а personalization сужает до целевого спикера. Это скорее аналог verifier (Приоритет 3), чем замена основной модели.

**Стоимость/риск/усилия:** on-device fine-tune на ESP32-S3 практически нереалистичен (нет backprop-инфраструктуры в TFLM); реалистичен только prototype-update в embedding-пространстве. Делать только если хотите per-household adaptation — иначе пропустить.

### Приоритет 7 — Сбор реального аудио (единственный оставшийся data-рычаг) — РЕАЛЬНЫЙ прирост
**Что делать:** включить on-device логирование триггеров и переобучать на реальном post-frontend аудио. **TaterTotterson firmware-капчер — turnkey loop**: flash Tater firmware из `TaterTotterson/microWakeWords` → device exposes entities *«Capture Wake Audio toggles upload of wake-word triggers. Capture Close Misses toggles upload of near misses»* → клипы (wake hits / close misses / false wakes) уходят на trainer app (`http://<trainer-ip>:8789`) → *«Every saved sample is normalized to 16 kHz / mono / 16-bit PCM WAV before training»* → переобучение, причём *«personal samples … gives them extra weight over TTS samples»*, а reviewed false-wakes идут отдельным hard-negative feature set.

Технический канал доступа к post-XMOS PCM: micro_wake_word и voice_assistant читают один shared microphone source (ESPHome #8645, релиз 2025.5.0); второй microphone source может стримить оба канала в Home Assistant (требует HA ≥2026.6.0).

**Ожидаемый эффект (на что именно):** real recall на незнакомых голосах + снижение реальных FA. Доказательство — Home Assistant «Help us make voice better in under a minute» (Wake Word Collective, 24 окт 2024): на 5,800+ сэмплах из 30 языков *«our model trained on this data falsely rejects 5% of samples, while previous models rejected 18%»*. Это крупнейший задокументированный прирост во всём стеке.

**Сколько реально нужно:** прямого числа «часов/спикеров для русского drawn-out» в публичных источниках нет, но 2010.06659 показывает, что 10 часов реального WW-аудио + multi-condition augmentation = production-grade. Для домашнего сценария начните с десятков-сотен реальных позитивов от 5-10+ спикеров (вся семья, друзья) разного пола/возраста, записанных **с разных дистанций по комнате** (как инструктирует HA: *«Taking samples from all around the room better reflects the real-world use … Don't worry about background noise; it's useful for training»*).

**Стоимость/риск/усилия:** низкая инженерно (firmware готова), но требует времени на сбор и ручной review клипов. Это **второй по важности рычаг после закрытия gap** и фактически реализует Приоритет 1 правильным способом.

### Приоритет 8 — Production-quality evaluation (предпосылка для всего остального) — КРИТИЧНО
**Что делать:** построить device-representative eval, предсказывающий поле, а не оффлайн-метрику. Методология openWakeWord: *«The false-reject rate should be determined from wakeword/phrases that represent realistic recording environments, including those with background noise and reverberation … The false-accept rate should be determined from audio that represents the types of environments that would be expected for the deployed model, not just on the training/evaluation data»*. Целевые пороги: *«a false-accept rate of <0.5 per hour and a false-reject rate of <5% is often reasonable in practice»*.

**Бенчмарк для FAPH:** DiPCo (Dinner Party Corpus; Van Segbroeck et al., Interspeech 2020, arXiv 1909.13447, Amazon) — *«10 sessions with a duration between 15 and 45 minutes»* (≈5.5 ч), группы из четырёх волонтёров за столом, запись одним close-talk микрофоном плюс пятью far-field 7-микрофонными массивами; лицензия CDLA-Permissive. Официальные mWW v2 модели калиброваны так, что *«on the DipCo set, they have at most 0.16 false accepts per hour and have less than 0.1 false accepts per hour on the PicoVoice benchmark»* (OHF-Voice/micro-wake-word release notes); hey_jarvis v1 давал 0.187 FA/час на DiPCo.

**Pitfalls:** (1) leakage — ваши held-out позитивы и обучающие данные из одного источника (TTS/один спикер) завышают recall; (2) non-representative test set — 93 чистых клипа vs 90-мин MUSAN-music **не** отражают реальную комнату с реверберацией и vocal-music; (3) FAPH на чистой музыке недооценивает реальные false-accepts на разговорной речи/ТВ. Вы уже частично это видите: разрыв между «recall 100% клипов ≥0.9» и реальным `recall@cutoff0.9/win3 = 0.946`.

**Стоимость/риск/усилия:** низкая, но это **первое, что нужно сделать** — без device-representative eval вы не сможете измерить выигрыш от Приоритетов 1-2 и будете оптимизировать оффлайн-прокси.

---

## Details: рекомендуемая последовательность и пороги

**Honest assessment — где реальный прирост, а где diminishing returns:**

| Рычаг | Категория | На что влияет | Сорсед-эффект |
|---|---|---|---|
| Train/serve gap (post-frontend train) | **РЕАЛЬНЫЙ** | real recall + FA | до 19% rel. FA / 34% rel. FRR при mismatch (2010.06676) |
| Far-field RIR augmentation | **РЕАЛЬНЫЙ** | far-field recall | до 35.8% rel. (2006.02774); +21% KWS recall (1907.03988) |
| Сбор реального аудио | **РЕАЛЬНЫЙ** | recall на незнакомых голосах | FRR 18%→5% (HA Wake Word Collective) |
| Device-representative eval | **КРИТИЧНО (предпосылка)** | предсказуемость поля | целевые <5% FRR, <0.5 FA/ч (openWakeWord) |
| Second-stage verifier | РЕАЛЬНЫЙ, но узкий | vocal-music FA без VAD | FA снижается кратно (2304.03416) |
| QAT-тюнинг | малый, низкий риск | FDR/latency | +4-6% rel. FDR, −68% time (2303.02284) |
| Knowledge distillation | умеренный | recall/robustness | inference не затрагивается (2506.11169) |
| AM-softmax / metric loss | маргинальный | confusables («сахар») | лучше при низких FA (2211.00439) |
| BC-ResNet / TC-ResNet порт | **diminishing/отрицательный ROI** | — | SOTA на GSC, но риск порта в mWW высок |
| On-device personalization | нишевый | recall на голосах семьи | 76% @10-shot, FA 5% (2306.02161) — но fixed-phrase ≠ per-speaker |

---

## Recommendations (стадийный план с порогами)

**Этап 0 (сделать первым, иначе всё остальное неизмеримо):**
1. Построить **device-representative eval-set**: записать позитивы и негативы **через сам Voice PE** (post-XMOS+TF4micro), с разных дистанций, в реальной комнате, с фоновой речью/ТВ/vocal-music. Считать FRR и FAPH по методологии openWakeWord (цели: FRR <5%, FAPH <0.5/ч). Добавить DiPCo-подобный far-field корпус для FAPH.
- **Порог перехода:** как только новый eval показывает разрыв с текущим оффлайн-eval (ожидаемо real recall < 0.946 и/или FAPH на vocal-music > 0 без VAD) — это подтверждает, что прирост в Этапах 1-2 будет реальным.

**Этап 1 (наибольший ROI):**
2. **Закрыть train/serve gap**: переобучить, прогоняя все позитивы/негативы через эквивалент NS/AGC-тракта Voice PE до извлечения спектрограмм.
3. **Far-field RIR augmentation**: convolve позитивы с OpenSLR SLR28 + pyroomacoustics-RIR, добавить шум на реалистичных SNR; сохранять долю «сухих» примеров, чтобы не просадить near-field.
- **Порог:** если far-field recall на новом eval вырос, а near-field не упал ниже текущих 0.946 — закрепить.

**Этап 2 (параллельно с Этапом 1):**
4. **Включить TaterTotterson capture loop** на устройстве: собирать wake hits / close misses / false wakes, ревьюить, добавлять реальные позитивы (вес выше TTS) и false-wakes (hard negatives). Цель — десятки-сотни реальных позитивов от 5-10+ спикеров.
- **Порог:** при достижении ~10 ч реального+augmented WW-аудио ожидается выход на production-grade (по аналогии с 2010.06659).

**Этап 3 (низкий риск, добивка):**
5. **QAT-тюнинг** (TF Model Optimization Toolkit) — +4-6% rel. FDR без изменения архитектуры.
6. **AM-softmax** против «сахар»/confusables, если на новом eval near-miss-rate вырастет.

**Этап 4 (опционально, только при отказе от VAD):**
7. **Second-stage verifier** (logistic regression поверх mWW-фичей или successive refinement) для остаточных vocal-music FA.

**Чего НЕ делать:**
8. **Не менять backbone** на BC-ResNet/TC-ResNet — ROI отрицательный под ~32k/45KB/CPU/fixed-phrase.
9. **Не вкладываться в on-device fine-tune** — нереалистично в TFLM; максимум prototype-update, и только для per-household adaptation.

---

## Caveats (сила доказательств)
- **Нет публичного A/B**, изолирующего именно device-captured post-frontend audio vs Piper+clean с FAPH-числами. Связь «train/serve gap → полевой прирост» подтверждена косвенно (2010.06676 о ±12 dB gain; HA-числа 18%→5% — это real-voice-vs-synthetic, не строго post-XMOS). Сила: **средняя-сильная, но косвенная**.
- **TaterTotterson репозиторий явно не утверждает «post-XMOS/post-TF4micro»** — это вывод из архитектуры (micro_wake_word на Voice PE получает XMOS-обработанный поток, что подтверждено DeepWiki). Capture-клипы — это то, что модель видела, но прямой формулировки в README нет.
- **Нет официального именованного «debug audio tap»**, дампящего ровно model-input PCM в файл. Практические пути: TaterTotterson capture-entities или стриминг второго microphone source в HA (≥2026.6.0).
- **Числа far-field/KWS-литературы получены на английском** (Google Speech Commands, «Hey Snips», «Alexa»), не на русской drawn-out фразе «захааар». Относительные приросты (%) переносимы, абсолютные — нет; обязательно валидировать на своём device-representative eval.
- **35.8% и 21%** — это relative improvement в конкретных far-field постановках; ваш стартовый recall уже высок (0.946), поэтому абсолютный прирост будет меньше, а основная выгода — в устойчивости к реверберации/дистанции, которую текущий eval не покрывает.
- **DiPCo** — англоязычный dinner-party корпус; как FAPH-бенчмарк он валиден (нецелевая речь/музыка/шум), но для FRR вашей русской фразы нужен собственный far-field набор.