# Куда вести on-device wake-word модель после сильного v8 baseline: длинный хвост, масштаб данных и production-цикл

## TL;DR
- После v8 главный рычаг улучшений сместился **с модели на данные и операции**: при INT8 ~32k и fixed-phrase архитектурные приёмы (distillation, QAT, focal/AM-softmax) дают убывающую отдачу, тогда как покрытие длинного хвоста (одновременная речь+музыка/ТВ, barge-in/AEC, far-field >5м, дети/пожилые, шёпот/крик) и production loop сбора реальных промахов через playback re-recording — это то, что реально ломает и чинит wake word в живых домах.
- Целевые метрики у вас уже на уровне индустриальных бенчмарков (Porcupine: «Benchmarked at 97.1% accuracy at 1 false alarm per 10 hours at 10 dB SNR» — против Snowboy 68% и PocketSphinx 52% в том же тесте; HA Collective: FRR 5% против 18%), поэтому к практическому «потолку» для двусложной фразы вы близки — дальнейший выигрыш приходит не от средней метрики, а от **сокращения худших режимов (worst-case FRR)** и от стабильности FAPH в новых, незнакомых домах.
- Конкретный приоритет: (1) multi-condition обучение с playback re-recording под музыку/ТВ и far-field RIR; (2) barge-in/AEC данные (устройство играет TTS); (3) production-петля логирования эмбеддингов промахов + периодическое переобучение; (4) опциональный second-stage verifier для FAPH; персонализация и QAT — последними.

## Key Findings

1. **Самые опасные режимы длинного хвоста — это не «средний дом», а сочетания: речь+медиа одновременно, barge-in под собственным TTS, far-field+реверберация+шум, и нестандартные голоса (дети, шёпот).** Far-field literature показывает резкую деградацию FRR с расстоянием/реверберацией, особенно при низком SNR (Convolutional Recurrent NN for Small-Footprint KWS, arXiv 1703.05390).
2. **Масштаб реальных данных важнее архитектуры.** Amazon (Gao et al., «Towards Data-efficient Modeling for Wake Word Spotting», arXiv 2010.06659): стартуя с 10 часов mismatch-данных и расширяя их в 20–100× через multi-condition augmentation + semi-supervised mining, можно достичь уровня production-модели, обученной на 375 часах far-field. Но есть плато: «20x augmentation (A200K) achieves the maximum gain, which suggests that overwhelming augmentation may hurt the performance».
3. **HA Wake Word Collective** документирует ровно ваш сценарий: реальные голоса снизили FRR с 18% до 5% («Our testing shows our model trained on this data falsely rejects 5% of samples, while previous models rejected 18%», на основе 5,800+ сэмплов в 30 языках). Это подтверждает, что данные реальных дикторов — главный рычаг, и что ваш результат v8 совпадает с траекторией Collective.
4. **Second-stage verifier / successive refinement** реально сокращает false alarm: Samsung (Saidutta, Srinivasa et al., arXiv 2304.03416) — «reduces FA by up to a factor of 8 on in-domain held-out FA data, and up to a factor of 7 on out-of-domain (OOD) FA data», для моделей от 13K до 2.41M параметров, plug-and-play.
5. **Personalization** (few-shot/prototype) даёт прирост, но рискует сломать свойство «срабатывает на любого диктора» — это speaker-narrowing, его надо применять как опциональный второй каскад, не как замену базовой модели.
6. **Production loop** с логированием эмбеддингов (не raw audio), периодическим переобучением на промахах и мониторингом field-FRR/FAPH — это то, как работают HA Collective, openWakeWord и федеративные подходы (Snips, arXiv 1810.05512).
7. **QAT** для INT8 даёт небольшой, но реальный выигрыш над PTQ; для streaming KWS — «our 8-bit FXP-QAT models have a 4-6% improvement in relative false discovery rate at fixed false reject rate» против full-precision (Macha, Oza, Escott et al., Amazon, arXiv 2303.02284).

## Details

### 1. Длинный хвост реальных условий: что ломает wake word и чем чинить

**Речь + музыка/ТВ одновременно (VAD пропускает).** VAD-гейт по определению пропускает речеподобный сигнал, поэтому музыка/ТВ с вокалом — главный источник как FA, так и FRR (целевая речь маскируется). Решение из литературы — multi-condition training: Amazon (2010.06659) явно смешивает household-noise и music/movie-библиотеки с целевым сигналом, SNR ~ нормальное распределение со средним 10 dB и σ=3 dB. Под ваши ограничения это реализуется через playback re-recording: проигрывать позитивы поверх фоновой музыки/ТВ через динамик так, чтобы микс прошёл XMOS+NS/AGC. Эффект: снижение worst-case FRR в режиме «медиа играет» и FA на вокальной музыке. Стоимость: время на запись, риск низкий.

**Barge-in / AEC (устройство играет TTS, пользователь перебивает).** Это отдельный тяжёлый режим: динамик ближе к микрофону, чем пользователь, поэтому near-end-to-echo ratio часто < −10 dB (VOCAL, AEC Barge-In). Voice PE имеет AEC в составе XMOS, но wake word под собственный TTS — классический «self-referencing» (iPhonMatchNet, arXiv 2309.06096). Решение под ваши ограничения: записывать позитивы playback-методом, **пока устройство одновременно играет типичный TTS-ответ** — тогда модель учится на остаточном эхе после XMOS AEC. Эффект: recall при barge-in. Стоимость: средняя (нужна синхронизация воспроизведения TTS и проигрывания позитива).

**Far-field >5м / реверберация / шумная кухня.** FRR резко растёт с расстоянием, особенно при низком SNR (1703.05390): far-field augmentation снижает деградацию на дальних дистанциях, но ухудшает на исходном (близком) тестовом наборе из-за train/test mismatch — поэтому нужен баланс близких и дальних примеров. PCEN-фронтенд (Trainable frontend for robust and far-field KWS, Google, ICASSP 2017) спроектирован именно под far-field, но у вас фронтенд фиксирован (micro_speech 40-band + XMOS), так что рычаг — данные: RIR-augmentation + playback re-recording на 3–6м в разных комнатах. Domain-aware training (arXiv 2005.03633) показывает, что можно улучшить far-field, сохранив качество на близком поле.

**Шёпот / крик (vocal effort).** Whisper и Lombard-эффект (крик в шуме) сильно сдвигают спектр и резко роняют распознавание (Kelly & Hansen, «Analysis and Calibration of Lombard Effect and Whisper», TASLP 2021; shouted/whispered mismatch «drastically decreases» performance). TTS-генераторы (Piper) почти не покрывают шёпот, поэтому это слепое пятно. Решение: записать реальные шёпот/громкие варианты фразы через устройство.

**Дети / пожилые.** Детская речь: «The word error rates (WER) for children's ASR were found to be 2 to 5 times worse than adults» (Shivakumar & Georgiou, arXiv 1805.03322), формантные частоты F1–F3 у детей выше взрослых более чем на 60%; внутрисубъектная вариативность падает к 12–15 годам. Это означает, что без детских позитивов модель будет систематически промахиваться по детям. Решение: целевой сбор детских голосов.

**Несколько одновременных дикторов / акустика разных комнат.** Покрывается разнообразием реальных записей в разных комнатах; multi-look/таргетное усиление (End-to-End Multi-Look KWS, arXiv 2005.10386) — это серверные multi-mic решения, для вас неприменимы (один аудио-тракт).

### 2. Масштаб реальных данных (scaling law)

- Amazon (2010.06659): база — «a total of 9487 utterances... total duration of the dataset is 10.18 hrs» против production-baseline на **375 часах** far-field. Multi-condition augmentation + semi-supervised mining дают экономию 97% сбора WW-данных и 86% разметки.
- Плато augmentation: «20x augmentation (A200K) achieves the maximum gain, which suggests that overwhelming augmentation may hurt the performance» — бесконечно множить аугментации одного и того же набора бессмысленно.
- Бенчмарк-лестница при FRR=0.05: только close-talk + multi-condition → в пределах 9× FAR от production; + untranscribed far-field semi-supervised → 5×; оба → 2×; + до 50K transcribed far-field → паритет с production.
- openWakeWord: модели обучены на **~30,000 часов negative** (речь/шум/музыка); «performance seems to increase smoothly with increasing dataset size», минимум несколько тысяч позитивов. Дизайн-цель движка — «Require little to no manual data collection».
- Snips Federated (arXiv 1810.05512): датасет 1,774 пользователя / 69,582 utterances, модель 190,852 параметра. Это даёт референс по числу дикторов: порядок «тысяча+ дикторов» — это production-уровень для англоязычного wake word.
- **Вывод по приоритету диверсити:** для генерализации на **незнакомые голоса** важнее **число дикторов**, для генерализации на **незнакомые комнаты** — число условий/RIR; playback re-recording наиболее ценен там, где он добавляет реальный device-tract сигнал, которого нет в TTS (шёпот, barge-in, far-field). Практический порядок: ~несколько десятков реальных дикторов × несколько комнат даёт основной прирост, дальше — убывающая отдача. Прямого «X дикторов → Y FRR» закона в литературе нет; масштабирование в опубликованных работах идёт по часам/utterances.

### 3. On-device personalization / enrollment

- Rusci & Tuytelaars (arXiv 2306.02161): few-shot open-set, «accuracy of up to 76% in a 10-shot scenario while the false acceptance rate of unknown data is kept to 5%»; triplet-loss энкодер с нормализованными фичами лучше прототипных сетей и имеет меньше параметров.
- Few-Shot KWS in Any Language (arXiv 2104.01454): 5-shot → F1 0.75 на 180 новых ключевых словах; streaming accuracy 87.4% при FAR 4.3%.
- On-device learnable user embeddings (arXiv 2403.07802): снижение error rate с 30.1% до 24.3% (до 19% относительного) на unseen speakers через обновление только user-проекций; FiLM-подход (arXiv 2311.03419) — +1% параметров с минимальным влиянием на latency.
- **Риск:** персонализация сужает модель к диктору (openWakeWord verifier — это явно speaker-narrowing). Под ESP32-S3 без train-ops это реализуемо только как лёгкий второй каскад (logistic regression на эмбеддингах), не как переобучение базовой сети. Применять, только если конкретный дом страдает от FRR на своих голосах.

### 4. Production loop непрерывного улучшения

- HA Collective: централизованный opt-in сбор реальных голосов; именно он дал FRR 18%→5%. TaterTotterson microWakeWord-Trainer + Tater firmware уже умеют «Capture Wake Audio» и «Capture Close Misses» с загрузкой клипов на trainer (Trainer App URL) — готовая референс-архитектура петли.
- openWakeWord: reviewed false-wakes конвертируются в hard-negative feature set для следующей модели; в microWakeWord-Trainer это `reviewed_negative_features/`, вставляемые в training YAML.
- Privacy: логировать **эмбеддинги/спектрограммы, не raw audio**. Федеративный подход (Snips, 1810.05512): «the weight updates (gradients) are sent to the server (encrypted), not the audio»; в их эксперименте 95% recall @ 5 FAH достигается за 100 раундов связи при 8 MB upload на клиента.
- Что мерить: field-FRR (доля промахов на подтверждённых попытках), FAPH по времени, drift распределения входов. Переобучать, когда накопилась пачка новых hard-negatives/промахов или при обнаружении дрейфа.

### 5. Second-stage verifier

- Samsung successive refinement (2304.03416): каскад speech→keyword-like→which-keyword, FA «up to a factor of 8 on in-domain... factor of 7 on out-of-domain», для моделей 13K–2.41M параметров, plug-and-play.
- openWakeWord verifier: «simple logistic regression binary classifiers» на shared audio features, обучается за <5 минут на диктора; срабатывает только когда base-модель >0.5. Документация: минимум 3 позитива + ~10 сек речи диктора (negative) + ~5 сек фоновых клипов окружения.
- Под ESP32-S3: дешёвая VAD-стадия (у вас уже есть) — это и есть первая ступень successive refinement. Полноценный второй verifier стоит flash/RAM/latency; **при вашем уже достигнутом FAPH 0/h с VAD ROI второго каскада низкий** — выгоднее «поднять порог + VAD», пока field-FAPH не вырастет.

### 6. Потолок двусложной фразы

- Picovoice: «Most well-known wake words have at least six phonemes»; короткие фразы (<6 фонем) детектировать труднее и они дают больше false positives; рекомендуется фонетически разнородная фраза с миксом гласных.
- Porcupine: «97.1% accuracy at 1 false alarm per 10 hours at 10 dB SNR» (вендорский self-benchmark, фон LibriSpeech test_clean + DEMAND noise).
- Ваша «захааар» с протяжной «а» — короткая (2 слога), что объективно тяжелее для FA. Но **протяжность фактически удлиняет фонетическую сигнатуру** (длинная стабильная гласная) — это плюс для дискриминации и центрирования. Вы близки к практическому потолку по средней метрике; дальнейший выигрыш — в worst-case режимах, а не в среднем FRR/FAPH.

### 7. UX / latency

- Phone-aligned обучение даёт latency 130–150ms против сильно большей у end-to-end (Low-resource Low-footprint Wake-word Detection, arXiv 2207.03331).
- Tiny-CRNN/CNN: ~172–218ms latency на «Alexa» (median length 700ms) (arXiv 2109.14725) — для реального использования незаметно.
- Протяжная фраза + большое окно → выше latency и больше шанс на FA внутри окна. Trade-off recall↔latency↔FA управляется параметрами `sliding_window_size` (меньше — ниже latency, но больше FA) и `probability_cutoff` (выше — меньше FA, больше FRR) в ESPHome `micro_wake_word`.

## Recommendations

**Этап 1 (наибольший ROI, делать сразу) — закрыть worst-case режимы данными через playback re-recording:**
1. Записать позитивы поверх музыки/ТВ (микс проходит XMOS+NS/AGC), SNR ~средний 10 dB. → снижает FRR в режиме «медиа играет» и FA на вокале. Низкий риск.
2. Записать позитивы, пока устройство играет типичный TTS-ответ (barge-in). → recall при перебивании. Средний эффорт.
3. Far-field: 3–6м в 4–6 разных комнатах + RIR-augmentation, балансируя с близкими примерами. → far-field recall без деградации близкого поля.
4. Собрать реальные шёпот/крик и детские/пожилые голоса. → закрывает крупнейшие слепые пятна TTS (Piper их не покрывает).
- **Порог пересмотра:** если worst-case FRR в каком-либо режиме на расширенном eval остаётся >10%, добавлять данные именно этого режима, а не множить аугментации (помним про плато на 20×).

**Этап 2 — расширить eval и построить production loop:**
5. Расширить device-eval с 153 позитивов / 6 мин шума до сотен позитивов от десятков дикторов и часов реального домашнего шума, со стратификацией по режимам (медиа/barge-in/far-field/vocal effort/дети). **Узкий eval (1–2 комнаты, один тракт) — главный текущий риск:** хорошие метрики на нём не гарантируют поведение в незнакомых домах.
6. Включить on-device логирование эмбеддингов промахов/close-misses (Tater firmware «Capture Close Misses») + privacy-by-design (никакого raw audio наружу). Переобучать при накоплении пачки hard-negatives или дрейфе.

**Этап 3 — модельные приёмы (убывающая отдача, делать после данных):**
7. QAT вместо PTQ при следующем релизе («4-6% improvement in relative false discovery rate at fixed false reject rate»; arXiv 2303.02284). Низкий риск, под ваш INT8/streaming подходит.
8. Distillation от большей teacher-модели на CPU — может помочь при дефиците данных, но эффект ограничен при уже достигнутом v8.

**Этап 4 — опционально, по сигналам поля:**
9. Second-stage verifier — только если field-FAPH вырастет; иначе «порог + VAD» дешевле (у вас FAPH уже 0/h).
10. Персонализация (logistic regression на эмбеддингах, <5 мин/дом) — только для домов с устойчивым FRR на своих голосах, с осторожностью к свойству «любой диктор».

**Сдвиг ROI:** после v8 отдача от модели (distillation/QAT/loss) измеряется единицами процентов; отдача от данных длинного хвоста и от production-петли — это единственный путь к снижению worst-case FRR и к стабильному FAPH в незнакомых домах. **Главная работа теперь — данные и операции, а не архитектура.**

## Caveats
- Относительные выигрыши из англоязычных датасетов переносимы, **абсолютные числа — нет**: «захааар» на русском и через ваш XMOS+TF4micro тракт даст свои абсолютные FRR/FAPH.
- Porcupine 97.1% — вендорский self-benchmark Picovoice, не независимая проверка.
- Amazon FAR-числа анонимизированы (только относительные множители 9×/5×/2× при FRR=0.05); прямого «speaker-count» scaling law в литературе не нашлось — масштабирование там по часам/utterances.
- Samsung successive refinement: на странице Samsung Research America фигурирует более слабый результат (3× до 340K параметров) против arXiv/ICASSP-версии (8×/7× до 2.41M); использую arXiv 2304.03416 как авторитетный.
- Few-shot/personalization числа (76% @ 10-shot, F1 0.75 @ 5-shot и т.п.) получены на Google Speech Commands, не на streaming-микроконтроллере — это верхняя граница ожиданий, не гарантия на ESP32-S3.
- Snips federated 95% recall @ 5 FAH — на их собственном crowdsourced датасете и модели 190k параметров; ваша модель ~32k, поэтому абсолютные пороги будут другими.