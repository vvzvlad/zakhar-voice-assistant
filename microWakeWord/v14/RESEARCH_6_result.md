# Глубокое исследование: on-device KWS для протяжного русского wake word «захааар»

## TL;DR
- Постановка заказчика (recall vs пер-классный FAPH на device-записанном eval, leakage-safe split, реальные device-негативы, playback-recapture) **в целом совпадает с тем, как задачу решают Apple, Amazon, Google и Snips** — это «подтверждено источниками» как правильный каркас; но eval почти наверняка занижает реальный FAPH (слишком короткие/доменно-смещённые негативы) и не покрывает обобщение на незнакомые голоса/far-field.
- Главная необсуждаемая слепая зона — **«опирается ли модель на длительность или на прокси»**: дискриминация протяжного от короткого произнесения это реальная, но почти не формализованная в KWS задача; её надо валидировать time-stretch контрфактами и time-axis saliency, иначе вся постановка «протяжный wake word» держится на непроверенном допущении.
- «Двухголовая» наводка заказчика имеет прямые прецеденты (Apple MTL voice trigger, Panchapagesan 2016), но в литературе выигрыш даёт **большая вспомогательная задача как регуляризатор маленькой**, а не «детектор+верификатор в одном графе»; в бюджете ~32k параметров вспомогательная голова с равным весом скорее навредит — приоритет должен быть на каскаде стадий и на проверке duration-зависимости, а не на красивой архитектуре.

## Key Findings

**Каркас оценки заказчика верен, но measurement-оптимистичен.** FAPH (false accepts per hour) на негативах + FRR/recall на позитивах в формате DET-кривой — это индустриальный стандарт (Amazon, Google federated KWS, Snips, Apple). Подтверждено источниками. Но три систематических искажения: (1) FAPH сильно зависит от того, насколько «плотные речью» негативы — Amazon обучает и оценивает на промышленных масштабах (в «Sub 8-Bit Quantization of Streaming Keyword Spotting Models for Embedded Chipsets», Interspeech 2022: «We conduct large scale experiments, training on 26,000 hours of de-identified production, far-field and near-field audio data (evaluating on 4,000 hours of data)»; в «Latency Control for Keyword Spotting», arXiv 2206.07261: обучение на ~20,000 ч, оценка на ~1,400 ч de-identified data); (2) leakage-safe split по источнику необходим, но недостаточен — нужен и split по acoustic-session; (3) recall на device-eval с известных голосов систематически переоценивает recall на незнакомых.

**В бюджете ≤45 КБ / ~32k INT8 параметров реально существует узкий набор семейств.** microWakeWord основан на MixConv/inception streaming (Rybakov et al. 2020). Конкуренты в этом бюджете: DS-CNN (MLPerf Tiny reference, 38.6k параметров, 52.5 КБ, цель 90% top-1 на Speech Commands), TC-ResNet, BC-ResNet (Kim et al., Qualcomm, Interspeech 2021: «The smallest BC-ResNet-1 achieves 96.6% accuracy with less than 10k parameters… BC-ResNets achieve state-of-the-art 98.0% and 98.7% top-1 accuracy on Google speech command datasets v1 and v2»), TENet, Tiny-CRNN (Amazon, arXiv 2109.14725: «32% reduction in False Accepts at a 50k parameter budget with 75% reduction in parameter size compared to word-level Dense Neural Network models»). Все они — single-head классификаторы.

**Двухстадийность — доминирующий индустриальный паттерн, но почти всегда между чипами/процессорами, а не на одном MCU.** Apple (AOP детектор → AP conformer-checker), Google (DSP → AP cascade), Amazon (on-device → cloud verification). Реальный on-device второй стейдж на одном чипе документирован слабо.

**Playback-recapture и реальные device-негативы — это в точности то, что делают Apple/Amazon (RIR + echo residuals).** Заказчик переоткрыл правильный приём.

**Duration-дискриминация — known task в hesitation/disfluency и stutter-prolongation литературе, но НЕ решена и почти не валидируется в KWS.**

## Details

### A. Постановка задачи и метрики

**Что подтверждено источниками как корректная методология:**

FAPH + FRR в виде DET-кривой с выбором рабочей точки — это стандарт де-факто. Amazon Tiny-CRNN явно меряет «number of False Accepts (FAs) at a fixed Miss Rate (MR) of 15%» и строит DET-кривые. Google production federated KWS: «number of false acceptances per hour of audio (FA/h)… while the false reject rate per keyword instance (FRR) was measured using positive utterances». Snips фиксирует рабочую точку «5 false alarms per hour» как stopping criterion и меряет recall при фиксированном FAPH. Сам micro-wake-word оптимизирует именно «minimizing… false accepts per hour on ambient background noise first», и затем максимизирует accuracy — то есть лексикографическая оптимизация (сначала FAPH-таргет, потом recall), что концептуально совпадает с тем, что делает заказчик.

Леджер «honest device-eval как единственная надёжная шкала» подтверждается напрямую: micro-wake-word предупреждает, что его собственная оценка FAPH во время обучения — «not a perfect estimate of the streaming model's real-world false accepts per hour, but it is sufficient for determining the best weights». То есть авторы фреймворка сами признают, что train-time FAPH — это суррогат, и реальная шкала — device-eval. Заказчик прав, что доверяет только device-eval.

**Где методология заказчика, вероятно, врёт (гипотезы, обоснованные источниками):**

1. **FAPH занижен из-за «лёгких» негативов.** Snips прямо отмечает: dev-set «is a "hard" dataset when it comes to false alarms since it belongs to the same domain as data used for training», и меряет FAPH отдельно на негативном тесте И на внешних наборах (Librispeech и др.). Если заказчик меряет FAPH преимущественно на тишине/музыке/своих device-негативах, он почти наверняка занижает FAPH на **плотной фоновой речи и подкастах/ТВ** — а именно там «захар»-подобные фонемные последовательности встречаются. Amazon оценивает на тысячах часов аннотированной dense speech именно поэтому. Это центральная методологическая дыра.

2. **Split по источнику не ловит acoustic-session leakage.** MLPerf Tiny и Speech Commands требуют, чтобы «a given speaker only appears in one of the sets». Заказчик делает leakage-safe split по источнику — это хорошо. Но playback-recapture создаёт риск, что одна и та же комната/микрофон/AGC-состояние присутствует и в train, и в test, и модель учит акустический отпечаток сессии (классический shortcut: «device-specific signatures in audio processing» задокументировано в обзорах shortcut learning). Нужен явный split по записи/сессии, не только по «источнику» текста.

3. **recall на device-eval — это recall на известных голосах.** Apple валидирует на take-home наборе из реальных пользователей именно потому, что структурированный eval переоценивает recall. У заказчика «реальных данных мало» по незнакомым голосам — значит, опубликованный им recall — оптимистичная верхняя оценка.

4. **Loss-metric mismatch.** Apple «Optimize What Matters» (Shrivastava et al., arXiv 2011.01151): «the mis-match between the DNN training loss (cross-entropy) and the end metric (detection score) is the main source of sub-optimal performance… we show significant reduction in false rejection rate (FRR) at the same false trigger experience (>70% over independent DNN training)». Заказчик тренирует BCE, а меряет sliding-window detection с порогом — это тот самый mismatch. Стоковый microWakeWord использует sliding-window averaging (в реализациях — скользящее среднее вероятностей с cutoff), и порог детектора — отдельный рычаг от обучения.

### B. Класс моделей ~32k / ≤45 КБ

**Подтверждено источниками — что существует и как оценивается:**

- **microWakeWord / MixConv-inception streaming**: основан на «Streaming Keyword Spotting on Mobile Devices» (Rybakov, Kononenko, Subrahmanya, Visontai, Laurenzo, Interspeech 2020), использует MixConv mixed depthwise convolutions, инференс каждые 30 мс по три 10-мс среза. 40 mel-банок, PCAN + noise reduction во фронтенде.
- **DS-CNN**: MLPerf Tiny reference, 38.6k параметров, 52.5 КБ TFLite, цель 90% top-1 на Speech Commands v2 (12 классов). «Hello Edge» (Zhang et al. 2017) показал DS-CNN 95.4% — на ~10% выше DNN при сравнимом числе параметров. Это прямой бенчмарк для бюджета заказчика.
- **BC-ResNet** (Kim et al., Qualcomm, Interspeech 2021, arXiv 2106.04140): broadcasted residual learning, BC-ResNet-1 = 96.6% при <10k параметров, BC-ResNet-8 = 98.0%/98.7% на v1/v2. Самое эффективное по параметрам семейство в литературе. ВАЖНО: использует residual connections, которые заказчик отбросил (регресс/невозможно) — и RepCNN (Apple, arXiv 2406.02652) объясняет почему: residual-структуры «add computational overhead… increase runtime memory size required on-device to store these branched intermediate outputs». RepCNN решает это reparametrization — обучают multi-branch, инференс в single-branch. Это потенциально релевантно: можно получить выигрыш residual-обучения без runtime-стоимости.
- **TC-ResNet, TENet/MTConv** (Li et al. 2020): multi-scale temporal convolution, 1D temporal+depthwise.
- **Tiny-CRNN** (Amazon, arXiv 2109.14725): 50k параметров, streaming, меряется FA at fixed Miss Rate 15%; «32% reduction in False Accepts at a 50k parameter budget».

**Практический предел качества при таком бюджете (гипотеза, обоснованная бенчмарками):** на чистом Speech Commands модели <50k параметров достигают 95–96% top-1. Но это closed-set accuracy на чистом аудио — НЕ FAPH на dense speech в реальной комнате. MLPerf v1.3 ввёл streaming wakeword benchmark с критерием «no more than 8 of either [false positives/false negatives]» — признание, что accuracy ≠ streaming-метрика. Реальный предел для one-keyword streaming в комнате определяется не accuracy, а train/serve gap и качеством негативов, а не размером модели — что заказчик уже эмпирически обнаружил.

### C. Форматы выходов и on-device многостадийность

**Двухстадийные каскады — доминирующий паттерн (подтверждено):**

- **Apple Voice Trigger**: streaming high-recall детектор (DNN-HMM, 5 hidden layers по 32/128/192 units, 4-bit palettization) на Always-On Processor → conformer-checker высокой точности на Application Processor. «low-power first-pass detector… If a detection is made… larger more complex models are used to re-score». Явный trade-off: «balance between power/battery consumption which is determined by the first-pass and overall accuracy which is determined by the larger models in the second-pass».
- **Google Cascade** (Gruenstein et al. 2017): «first stage… very small and power efficient detector that executes on a DSP. Upon trigger, it delegates the final detection decision to a second, much larger and more accurate detector, that executes on the device's main application processor (AP)». Бюджет: «entire system must consume less than 1mA».
- **Amazon**: two-stage DNN-HMM + light second-stage classifier on-device, затем server verification. Monophone-based background modeling для второй стадии.

**Где провести границу между стадиями НА ОДНОМ чипе (критический разбор):**
Вся опубликованная двухстадийность раскладывается между РАЗНЫМИ вычислительными доменами (DSP→AP, или device→cloud) именно потому, что выигрыш каскада — энергетический: дешёвая always-on стадия экономит батарею, дорогая стадия включается редко. На ОДНОМ ESP32-S3 без отдельного always-on домена этот главный мотиватор каскада исчезает: вторая модель всё равно крутится на том же ядре, и если первая стадия часто триггерит, суммарное потребление и задержка растут. Поэтому on-device второй стейдж на одном MCU оправдан НЕ ради энергии, а ради **точности при редком триггере**: если первая стадия даёт мало кандидатов (например после VAD-прегейта, который заказчик уже применяет), вторая (более тяжёлая, с большим контекстным окном) пересматривает кандидата с полным окном — как Apple checker «receive the entire input audio for re-scoring at once and do not need to be run in a streaming setting».

**Компромиссы (гипотезы):** (1) задержка — вторая стадия добавляет окно ре-скоринга после триггера, что для wake word терпимо (сервер всё равно подключается после wake); (2) память — две модели в ≤45 КБ суммарно жёстко; (3) риск рассинхрона — если стадии видят разные фронтенды/окна, их рабочие точки дрейфуют независимо (Apple решает это общим фронтендом).

**Несколько голов в одном графе vs две модели:** Apple MTL voice trigger (Sigtia et al., arXiv 2001.09519) — это ОДНА сеть (4 BLSTM по 256, ~5M параметров) с двумя головами, делящими все веса: фонетическая голова (CTC, 53 символа, ~5000 ч) + дискриминативная голова (2 units, ~90 ч hard-негативов). Ключевой результат: большая фонетическая задача «acts as a regulariser for the trigger phrase discrimination task with a much smaller dataset», и «this strategy yields significant gains on two challenging test sets, reducing the number of errors by half in some conditions, without requiring extra parameters compared to the baseline system». Panchapagesan et al. (Interspeech 2016): shared hidden layers + KW-phone-states head + LVCSR-senones head; LVCSR-init + MTL + class-weighted XENT дал 26% снижение AUC DET (с 0.239x random-init single-task до 0.176x). ВАЖНЫЙ риск: MTL **сама по себе** (random-init multi-task, 0.202x) была на ~1.5% относительных AUC ХУЖЕ, чем простая transfer-learning инициализация (LVCSR-init single-task, 0.199x) — выигрыш появился только в комбинации с init и class-weighting, и вес головы γ чувствителен (оптимум 0.75–0.9 в зависимости от объёма данных). Production-«checker» Siri использует тот же паттерн: «take the encoder branch of the model and add an additional output layer (affine transformation and softmax nonlinearity) with two output units… One unit corresponds to the trigger phrase, while the other corresponds to the negative class», обучаясь «simultaneously minimise the CTC loss and the discriminative loss».

### D. Протяжность как признак — крупнейшая концептуальная слепая зона

**Подтверждено источниками — это known, но не KWS-task:**
Дискриминация протяжного/растянутого произнесения от нормального изучается в двух смежных областях:
- **Hesitation / word-lengthening detection**: Goto, Itou & Hayamizu (Eurospeech 1999) детектируют lengthening «on the basis of small fundamental frequency transition and small spectral envelope deformation» — то есть по **временной стабильности формант на длинном отрезке**, не по громкости. Это ключевая методологическая подсказка: правильный признак протяжности — стабильность спектра во времени, а не энергия. Отдельные HMM-детекторы заявляют способность «distinguish filled pauses from elongated words»; для русской спонтанной речи воспроизведённая точность voiced lengthening ~80%, unvoiced ~66% («Filled Pauses and Lengthenings Detection… for the Spontaneous Russian Speech», Springer 2014).
- **Stutter-prolongation detection**: FluentNet (Kourkounakis et al., arXiv 2009.11394) — SE-ResNet+BiLSTM+global attention над временем для классификации prolongations; StutterNet (arXiv 2105.05599) — TDNN. Эти системы используют MFCC/спектрограммы — то есть подвержены тому же proxy-риску.
- **Phone-duration modeling**: Wei, Hunt & Skilling (Apple, Interspeech 2019, arXiv 1909.03030) — DNN моделирует распределения длительностей фонем, явно с признаком long/short vowel, lexical stress, pre-pausal lengthening, speaking rate. Валидация: «we show that the DNN is able to learn the lengthening effect of these two features». Применение прямо аналогично задаче заказчика: модель флагает аномальные длительности (например «extreme stress… hence lengthened it»). Длительности биннятся в 45 бинов от 30 мс с шагом 10 мс.

**Как ВАЛИДИРОВАТЬ, что модель опирается на длительность, а не на прокси (главный методологический вклад):**
В KWS-литературе НЕТ работы, которая делает полный duration-vs-energy контрфактуальный ablation для elongation. Это genuine gap. Методология собирается из explainability + augmentation toolkit:
1. **Time-stretch контрфакты**: синтезировать duration-controlled тест-сеты — одно и то же произнесение с варьируемой длиной гласного при loudness-нормализации; проверить, что score модели монотонно растёт с длительностью «ааа», а не с энергией.
2. **Time-axis saliency / occlusion**: систематически маскировать временные участки спектрограммы и смотреть, где падение score максимально (occlusion sensitivity); Integrated Gradients/SmoothGrad над STFT; cross-domain Integrated Gradients (Kechris et al., arXiv 2505.13100) разделяет вклад временной и частотной осей. Keyword Transformer (Berg et al., arXiv 2104.00769) ablates time- vs frequency-domain attention и находит time-domain лучше для KWS.
3. **Duration-binned eval**: стратифицировать тест по измеренной (forced-alignment) длине гласного.
4. **Base-rate caution**: психолингвистика (MMN-исследования) показывает, что duration phonemically relevant только для лексически долгих гласных; в естественной речи короткие формы доминируют (в японской infant-directed речи «94% of vowels are short»), что означает сильный proxy-shortcut риск — модель может выучить, что «длинное = редкое = триггер» по base-rate, а не по реальной длительности.

**Критический вывод по D:** заказчик не знает, опирается ли модель на длительность — и это не «нерешённая деталь», а допущение в фундаменте всего проекта. Если модель различает «захааар»/«захар» по спектральным/громкостным прокси, то весь recall в поле хрупок к смене голоса/комнаты. Это должно быть исследовано ПЕРВЫМ (см. H).

### E. Train/serve gap под непрозрачным фронтендом (AEC/NS/AGC + PCAN)

**Подтверждено источниками:**
- **PCEN/PCAN робастность**: Wang, Getreuer, Hughes, Lyon & Saurous (Google, ICASSP 2017, arXiv 1607.05666) ввели PCEN как AGC-based динамическую компрессию вместо лог-компрессии: «On our large rerecorded noisy and far-field eval sets, we show that PCEN significantly improves recognition performance». КРИТИЧНО для заказчика: PCEN/PCAN «suppresses stationary backgrounds and enhances onsets» — то есть усиливает транзиенты. Заказчик прав, что PCAN усиливает текстуру тишины (микроструктуру комнаты + AEC residual онсеты воспринимаются как речевые онсеты). Это структурная причина ложных в тишине, и фронтенд фиксирован.
- **Playback-recapture / RIR + echo residuals**: это в точности индустриальный приём. RepCNN и HEiMDaL (Apple): training data «enhanced to mimic real-world acoustic conditions using room-impulse responses (RIRs) along with echo residuals», gain augmentation 10dB…-40dB. Amazon «Data Augmentation for Robust KWS under Playback Interference» (arXiv 1808.00563) моделирует barge-in/AEC residual.

**Что заказчик, вероятно, не видит (гипотезы):**
1. **Residual echo suppression / self-trigger как известный класс багов**: Amazon-патенты (US 11,380,312; 12,033,631) описывают, что residual echo «may trigger a wakeword engine… to self-trigger», и решают это per-frequency-band статистикой энергии слова + double-talk detection. Заказчик борется с симптомом (ложные в тишине/музыке через hard-негативы), но не использует двухканальность для **прямого детекта double-talk** (см. G).
2. **AEC нелинейности убивают именно протяжный сигнал**: barge-in литература — «WWD does not tolerate non-linear distortions to the wake-up word… eliminates the use of the residual echo suppressor». Протяжный гласный во время playback — наихудший случай для AEC, потому что долгий стационарный сигнал максимально страдает от нестационарностей echo path.
3. **Domain-invariance вместо только data-matching**: заказчик закрывает gap матчингом данных (playback-recapture). Литература по distribution shift (Amazon «Wakeword Detection under Distribution Shifts», arXiv 2207.06423) использует teacher/student SSL, где «gating of the accepted data undergoes a shift in distribution». Это вместо статичного матча — непрерывная адаптация.

### F. Данные

**Подтверждено источниками — сколько и какого рода:**
- Apple HEiMDaL/RepCNN: 500k–1M utterances (wake word + query), augmented RIR+echo. Это production-масштаб.
- Amazon «Towards Data-efficient Modeling» (arXiv 2010.06659): новое слово «Amazon», только 9487 utterances / 10.18 ч close-talk в тихой комнате как старт, против 375 ч production baseline; semi-supervised pipeline увеличивает датасет в 20–100× из untranscribed речи. Вывод: «only untranscribed data is not enough to achieve production performance» — но close-talk seed + SSL сильно сокращает потребность.
- Snips low-resource: subsets 100/500/1000/2000 позитивных utterances; полный набор 5799 utt / 1163 спикера. KD-работа (arXiv 2207.03331) показывает, что 100–2000 позитивов работают с transfer-learning/distillation от 2800-ч учителя.
- microWakeWord: полностью синтетический (Piper TTS) + готовые спектрограммы негативов на HuggingFace.

**Data-efficiency / active learning / hard-negative mining (подтверждено):**
- **Federated/SSL flywheel**: Google production federated KWS (arXiv 2204.06322) кэширует до 20 near-activation примеров/день + activation events, хранит 63 дня; «Near activation events provided examples of (true and false) rejects… activation events provided examples of (true and false) accepts». Метки — от teacher-модели (no-peek). Это прямой шаблон для data-flywheel заказчика: pre-trigger аудио = near-activation cache.
- **Confusable mining через ASR**: Apple MTL voice trigger собирает «much smaller dataset of examples that are challenging for the baseline system» — ровно то, что заказчик делает прогоном ASR (выявил «закон/закат/зачем/сахар»).

**Как избегают катастроф (дрейф/переобучение):**
- Distribution shift при редеплое (Amazon arXiv 2207.06423): новая модель меняет gating → сдвиг распределения accepted data → следующая итерация обучается на смещённых данных. Это **именно та катастрофа**, которой грозит наивный flywheel заказчика: логируя только то, что текущая модель приняла/почти приняла, он обучает следующую модель на её собственных слепых зонах. Решение в литературе — teacher без доступа к новому распределению + label-marginal assumption.
- Relative FRR@FAR (arXiv 2305.05110) как метрика, чтобы новая модель не регрессировала относительно baseline при фиксированном FAR — формализация «честной шкалы» заказчика.

### G. Микрофонные каналы

**Подтверждено источниками:**
- Multi-channel KWS обычно про beamforming/массивы (Ji et al.; Yu et al. End-to-End Multi-Look KWS arXiv 2005.10386): несколько лучей + omni-канал. «omni channel as a non-distorted signal for efficient operation in high signal-to-noise ratio cases» — омни/менее-обработанный канал ценен именно в high-SNR. Это прямо релевантно: канал 1 (тише/менее обработан) — аналог omni-канала.
- Huang et al. (в обзоре arXiv 2507.15558): два микрофонных канала, ANC классифицирует направленность звуков >1 сек до активации как шум для подавления; «3-microphone scheme is the most effective».
- Multichannel Voice Trigger via Transform-Average-Concatenate (Apple, arXiv 2309.16036): объединение каналов внутри сети.

**Что это даёт и подводные камни (гипотезы под ограничения заказчика):**
Главная ценность второго (менее обработанного) канала для заказчика — НЕ beamforming (XMOS уже сделал AEC/NS/AGC на канале 0), а **разностный сигнал для гейтинга**: ложные в тишине/музыке — это AEC-loopback residual + PCAN-усиленная текстура, которые по-разному проявляются на обработанном (канал 0) и менее обработанном (канал 1) каналах. Если ложное срабатывание — артефакт обработки, оно может присутствовать на канале 0 и отсутствовать/отличаться на канале 1; реальный голос присутствует на обоих коррелированно. Это даёт дешёвый on-device double-talk/self-trigger детектор (концептуально как Amazon self-trigger prevention через корреляцию каналов). Подводные камни: (1) каналы рассинхронизированы по AGC-усилению — наивная разность шумна; (2) удвоение фронтенда (две спектрограммы) — память; (3) канал 1 тише → хуже recall тихих произнесений, нельзя использовать его как первичный.

### H. Слепые зоны и риски — приоритизированный разбор (самое ценное)

**Приоритет 1 — Опирается ли модель на длительность? (НЕ спрашивается, должно быть первым).**
Весь проект построен на допущении, что модель различает «захааар»/«захар» по длительности. Это не проверено. Если различение идёт по прокси (громкость, спектральный отпечаток конкретных записей, base-rate), то: (а) recall в поле хрупок к смене голоса/комнаты — что заказчик уже наблюдает («версия просела по recall в поле»); (б) «протяжность» как фича иллюзорна. **Что сделать первым:** time-stretch контрфактуальный eval (один голос, варьируемая длина гласного, loudness-нормализация) + occlusion/IG saliency над временной осью. Это дёшево (не требует новых полевых данных) и либо подтверждает фундамент, либо вскрывает, что задача поставлена неверно. Обоснование приоритета: самый дешёвый эксперимент с самым высоким информационным выигрышем, и он адресует центральное напряжение recall-vs-FAPH.

**Приоритет 2 — FAPH меряется на слишком лёгких негативах.**
Заказчик чинил ложные в тишине/музыке, но фонетические соседи («закон/закат/зачем/сахар») и dense conversational speech — где живёт реальный FAPH — недопредставлены в eval. Snips, Apple, Amazon меряют FAPH на сотнях-тысячах часов dense speech именно поэтому. **Что сделать:** собрать/прогнать FAPH на большом корпусе русской спонтанной речи и подкастов/ТВ; почти наверняка вскроется, что «лучшая» по device-eval версия имеет высокий FAPH на речи. Это та же ловушка, что заказчик уже ловил leakage-safe split'ом — но в измерении FAPH, а не recall.

**Приоритет 3 — Data-flywheel отравит сам себя (distribution shift).**
Логирование только того, что текущая модель приняла/почти приняла, обучает следующую модель на её собственных слепых зонах (Amazon arXiv 2207.06423). Hard-негативы из flywheel смещены к тому, что текущая модель уже почти ловит. **Что сделать:** teacher/student SSL с teacher без доступа к новому gating + сохранение фиксированного «золотого» held-out, не зависящего от продакшн-gating; мониторить Relative FRR@FAR против baseline.

**Приоритет 4 — Acoustic-session leakage в playback-recapture.**
Split по источнику текста не исключает, что та же комната/AGC-состояние в train и test → модель учит отпечаток сессии (shortcut). **Что сделать:** split по записи/сессии/устройству, не только по тексту; держать «чужую» комнату/устройство только в test.

**Приоритет 5 — «Двухголовая» наводка: контекстуализация и критика.**
Прецеденты (Apple MTL voice trigger, Panchapagesan 2016) показывают, что выигрыш двух голов — это **регуляризация маленькой задачи большой вспомогательной** (фонетический CTC на больших данных регуляризует дискриминацию на малых). У заказчика нет большого фонетически размеченного русского корпуса под вспомогательную голову, а в бюджете ~32k параметров вспомогательная голова конкурирует за ёмкость (Panchapagesan: MTL сама по себе была ХУЖЕ на ~1.5% AUC без правильного init/взвешивания). Вывод: «двухголовость ради детектор+верификатор в одном графе» вероятно не туда — выигрыш каскада в литературе энергетический (разные процессоры), которого на одном ESP32-S3 нет. Более перспективно: (а) вспомогательная **duration/CTC-голова как регуляризатор и одновременно инструмент валидации duration-зависимости** (адресует Приоритет 1); (б) RepCNN-style reparametrization (обучать multi-branch, инференс single-branch), чтобы получить выигрыш residual-обучения без runtime-стоимости, которую заказчик справедливо отбросил. Caveat: НЕТ опубликованной <100k-параметрной multi-head wake-word модели с измеренными пер-головными эффектами — экстраполяция на бюджет заказчика это гипотеза.

**Приоритет 6 — Обобщение (незнакомые голоса/far-field/дети/шёпот/крик/barge-in).**
Реальных данных мало; recall на device-eval — оптимистичная верхняя оценка (известные голоса). Far-field — это родная область PCEN, но заказчик не контролирует фронтенд. **Что сделать:** take-home eval на незнакомых пользователях (как Apple) до любых выводов о recall; явно тестировать barge-in (протяжный гласный во время playback — наихудший случай AEC).

**Приоритет 7 — Angular-margin head перепроверяется правильно.**
Заказчик отметил, что angular-margin был сильнейшим рычагом recall, но рос music-FAPH, и сейчас перепроверяется с реальными музыкальными негативами. Это методологически верно. Caveat из литературы: angular-margin (AAM/AM-softmax) известны нестабильностью на lightweight-моделях («divergence with lightweight models», требуют margin annealing/pre-training с softmax). В бюджете ~32k параметров это реальный риск нестабильности обучения.

## Recommendations

**Этап 0 (сделать первым, не требует новых полевых данных):**
1. **Duration-causality eval.** Сгенерировать time-stretch контрфакты (варьируемая длина гласного, loudness-нормализация) + occlusion/IG saliency над временной осью. Порог решения: если score не монотонен по длительности или saliency не концентрируется на гласном — задача «протяжного wake word» поставлена неверно, и приоритет смещается на пересмотр определения цели.
2. **Hard-FAPH eval.** Прогнать FAPH на ≥100 ч русской dense speech/подкастов/ТВ. Порог: если FAPH на речи в разы выше, чем на тишине/музыке — текущая рабочая точка нерелевантна, и надо ребалансировать негативы в сторону confusable речи.

**Этап 1 (после Этапа 0):**
3. Внедрить session/device-level split в дополнение к source-split; вынести «чужую» комнату/устройство в held-out.
4. Take-home eval на незнакомых голосах перед любым релизным решением recall-vs-FAPH.
5. Спроектировать flywheel с teacher без доступа к новому gating + фиксированный золотой held-out; мониторить Relative FRR@FAR.

**Этап 2 (архитектурные эксперименты, только если Этап 0 подтвердил duration-зависимость):**
6. Вместо «детектор+верификатор в одном графе» — протестировать (а) duration/CTC вспомогательную голову как регуляризатор (двойная польза: и валидация, и recall), с тщательным подбором веса γ (литература: 0.75–0.9, чувствительно); (б) RepCNN reparametrization для residual-выигрыша без runtime-стоимости.
7. Второй канал микрофона — использовать как разностный сигнал для on-device double-talk/self-trigger гейтинга (не как первичный вход и не для beamforming), адресуя AEC-residual ложные напрямую.
8. On-device второй стейдж оправдан только если первая стадия (после VAD) даёт редкие кандидаты; тогда вторая стадия с бОльшим окном ре-скоринга по полному контексту (Apple checker pattern). Если триггеры частые — каскад на одном чипе не окупается.

**Что НЕ делать:** не наращивать синтетику (заказчик прав); не вводить angular-margin без annealing/pre-training в этом бюджете; не строить двухголовую модель ради самой декомпозиции без большой вспомогательной задачи-регуляризатора.

## Caveats
- Внутренности микро-wake-word фронтенда (точные параметры PCAN, sliding-window cutoff) известны из публичного репозитория и реализаций; конкретные числовые пороги (напр. 222–250 из 255) взяты из community-форка (TaterTotterson) и могут отличаться от стоковой прошивки заказчика.
- Все опубликованные multi-head wake-word модели имеют ≥млн параметров (Apple ~5M) или неуказанный размер (Panchapagesan); НЕТ опубликованной <100k-параметрной multi-head wake-word модели с измеренными пер-головными эффектами — экстраполяция на бюджет заказчика это гипотеза, не подтверждённый факт.
- НЕТ ни одной KWS-работы, делающей полный duration-vs-energy контрфактуальный ablation для elongation — методология Приоритета 1 собрана из смежных областей (hesitation/stutter detection, explainability), это обоснованная гипотеза, а не воспроизведённый в KWS протокол.
- Двухканальный double-talk гейтинг (G) — гипотеза по аналогии с Amazon self-trigger патентами; прямой публикации для ESP32-S3/XMOS-конфигурации нет.
- FAPH-числа заказчика (12.5→0.8/ч, →0) приняты как данность из брифа, не верифицированы независимо.
- Масштабы данных Amazon (26,000 ч обучение / 4,000 ч eval в «Sub 8-Bit Quantization…»; ~20,000 ч / ~1,400 ч в «Latency Control for Keyword Spotting», arXiv 2206.07261) — это промышленный multi-keyword контекст, не прямой ориентир для single-keyword DIY-проекта; приведены как индикатор того, на каких объёмах меряется «честный» FAPH.