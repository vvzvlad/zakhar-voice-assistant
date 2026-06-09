# Критическая оценка истории разработки on-device wake word «захааар» (v1–v9): что пробовать дальше, что исчерпано, где практический потолок

## TL;DR
- **Модель-сайд исчерпан; почти весь оставшийся прирост — в РЕАЛЬНЫХ device-данных и эксплуатации.** После v8 distillation/QAT/loss/арх дали регресс или ноль; единственное модельное исключение с потенциально положительным ROI — настоящий angular-margin (Sub-center ArcFace / AM-Softmax) head, который в литературе по user-defined KWS даёт существенное снижение EER против plain-softmax/BCE (Jung et al., ICASSP 2023, arXiv:2211.00439), тогда как ваш logit-shift аппрокс закономерно регрессировал.
- **Два открытых режима (barge-in через AEC-остаток и ложные в тишине/idle-комнате) принципиально не закрываются синтетикой** — их распределение воспроизводимо только записью через реальный аудиотракт устройства; это прямое продолжение вашего главного рычага (playback re-recording, FRR 18%→5%).
- **Вы у практического потолка СРЕДНИХ метрик 2-сложной фразы** (recall 0.95, device-FAPH 0 с VAD ≈ индустрия: Porcupine, по данным Picovoice, <3% FRR при 1 ложном срабатывании на 10 часов и 97.1% точности при 10 дБ SNR). Дальнейший выигрыш — только в WORST-CASE (music_snr5, reverb+music10, muffled) и в UX (персонализация, production-петля), не в среднем.

## Key Findings
1. **Angular-margin head — единственный неисчерпанный модельный рычаг.** На user-defined KWS (Jung et al., KAIST / Hyundai Motor / 42dot, ICASSP 2023, arXiv:2211.00439) замена plain-softmax на AM-Softmax (L2-нормированный softmax с косинусным margin, бэкбон res15, претрейн на их 1000-классном датасете LibriSpeech Keywords) и angular-prototypical обучение даёт заметное снижение EER и FRR@fixed-FAR. Ваш «margin-BCE logit-shift» — НЕ то же самое: нужен L2-нормированный embedding + истинный additive-margin head.
2. **PCEN как обучаемый фронтенд — реальный кандидат, но блокируется фиксированным micro_speech.** PCEN (Wang et al., Google, arXiv:1607.05666) даёт gain-инвариантность и подавление стационарного фона, что напрямую бьёт по far-field и AGC-pumping; но вход модели в microWakeWord — фиксированные 40-полосные micro_speech спектрограммы, и PCEN потребует форка фронтенда.
3. **Ложные в тишине — известная проблема always-on KWS** (AGC-разгон в idle, self-noise микрофона); фиксы: silence/idle hard-negatives из реальной пустой комнаты, energy/SNR-гейт, минимальная длительность. Синтетика этого не воспроизводит.
4. **Барж-ин — это задача робастности к остатку эха, а не AEC** (Raju, Panchapagesan, Liu, Mandal, Strom, Amazon, arXiv:1808.00563): дата-аугментация музыкой/ТВ, по их abstract, «we get around 30-45% relative reduction in false reject rates, at a range of false alarm rates, under audio playback from such devices». Но у вас AEC с loopback-референсом убирает полногромкую музыку → синтетический mix = неверное распределение; нужен реальный остаток через устройство.
5. **Scaling law по дикторам реален и переносим.** Park et al. (Google, arXiv:2407.18879): «relatively small amounts of real audio data with speaker diversity (100 speakers, 2k utterances) and large amounts of TTS synthesized data can achieve reasonably high accuracy (within 3x error rate of baseline)» относительно baseline на 3.8M реальных позитивов. У вас «десятки» дикторов — это и есть главный незакрытый data-рычаг.
6. **Second-stage verifier — проверенный приём** (Samsung, «To Wake-up or Not to Wake-up: Reducing Keyword False Alarm by Successive Refinement», arXiv:2304.03416: снижение false alarm в 3–8 раз; cascade arXiv:1712.03603). Ваш 65-парам логрег-verifier уже работает (100% reject намайненных муз-ложных) — готов, но не задеплоен, т.к. с VAD FAPH=0.

## Details

### (А) Что даёт прирост ПОСЛЕ v8

**Приоритет 1 — Реальные long-tail device-данные через playback re-recording (наибольший ROI).**
Это прямое продолжение вашего доказанного главного рычага (FRR 18%→5%). Записать через аудиотракт устройства:
- **idle-комната / тишина (негативы)**: текстура живой комнаты (50 Гц-гармоники сети, self-noise, codec-артефакты, AGC-pumping). → Бьёт по тишина-FAPH, который синтетика не воспроизводит.
- **barge-in через AEC устройства**: реальный остаток эха, не синтетический mix. → Бьёт по barge-in.
- **far-field >5 м, шёпот/крик, дети/пожилые**: расширение акустического и демографического покрытия. → Бьёт по незнакомым голосам и worst-case.
- Цена/риск: сбор данных трудозатратен, но обучение CPU-only это переваривает; риск минимален (доказанный приём). Валидировать на device-eval.
Источник: ваш собственный результат + Amazon arXiv:1808.00563 (playback interference, 30–45% rel. FRR↓), Opatka et al., Interspeech 2021.

**Приоритет 2 — Больше реальных дикторов (главный data-рычаг по литературе).**
Park et al. (Google, arXiv:2407.18879): ~100 дикторов + 2k реальных позитивов + масштабная TTS дают точность в пределах ~3× error rate от baseline на 3.8M реальных. У вас «десятки» → переход к 100+ дикторам должен дать ощутимый прирост на незнакомых голосах. Относительные приросты переносимы с английского; абсолют валидировать на своём device-eval.
Цена/риск: сбор реальных русских дикторов на протяжное «захааар» — дорого, но это самый предсказуемый рычаг генерализации.

**Приоритет 3 — Настоящий angular-margin / metric-learning head (единственный неисчерпанный модельный приём).**
Замена weighted-BCE на L2-norm embedding + additive-margin (AM-Softmax / Sub-center ArcFace) по рецепту Jung et al. (arXiv:2211.00439). Ожидаемый эффект на user-defined KWS — существенное относительное снижение EER и FRR@fixed-FAR против plain-softmax/BCE при той же схеме обучения. Это НЕ ваш logit-shift аппрокс (он закономерно регрессировал, т.к. не нормирует embedding и не вводит истинный угловой margin).
- INT8-квантование: держать L2-нормализацию и cosine-скоринг в float32 (TFLite вставит quant/dequant на границе), квантовать только conv-backbone per-channel (TFLite per-axis поддержан для Conv2D/DepthwiseConv2D). По данным speaker-verification, ResNet-style бэкбоны квантуются почти без потерь (arXiv:2406.05359), attention-heavy головы теряют заметно больше (arXiv:2210.17326) — выбор бэкбона доминирует в quantization-робастности.
- Цена/риск: переобучение + аккуратное квантование головы; риск средний (квантование L2-нормализации — известный footgun, держать в float). Валидировать worst-case на device-eval.

**Приоритет 4 — Задеплоить уже готовый second-stage verifier.**
Ваш 65-парам логрег-verifier отвергает 100% намайненных no-VAD муз-ложных, пропускает 100% истинных wake, ~260 байт, ~0 латентности. Приём проверен (Samsung successive refinement, arXiv:2304.03416: FA↓ в 3–8 раз; cascade Google arXiv:1712.03603). Даже если с VAD FAPH=0 — это страховка для no-VAD режимов и worst-case (когда VAD пропустит речеподобный фон).
Цена/риск: близко к нулю (уже построен). Деплоить как defense-in-depth.

**Приоритет 5 — Production-петля непрерывного улучшения (заскаффолжена — запустить).**
Логирование эмбеддингов промахов (НЕ звука — privacy) → майнинг hard-neg → переобучение. Рабочие практики: микс hard+random негативов 1:1–1:3 (чистый hard дестабилизирует и ведёт к коллапсу), фильтрация false-negatives по порогу, периодический re-mining (еженедельно–ежемесячно), A/B перед раскаткой, инференс FRR из retry-паттернов (юзер повторяет фразу). Источники: hard-negative mining practice; Arun Baby production notes (мониторинг P50/P95/P99 латентности, параллельный shadow-прогон новой модели).
Цена/риск: инженерия логирования + мониторинг дрейфа; средние усилия, высокий долгосрочный ROI.

**Приоритет 6 — On-device few-shot персонализация (опционально, UX-рычаг).**
Few-shot enrollment (4 повтора, ~20 сек) поверх speaker-independent модели. Литература: Sensory заявляет ~55 КБ на 1 юзера, до 150 КБ на 10 (audioXpress, SDK v7.6.0); FiLM-персонализация +1% параметров улучшает underrepresented группы (arXiv:2311.03419); one-shot noise-adaptation последнего FC-слоя даёт +4.9–46% точности при SNR≤18 дБ (Martinez-Rau et al., arXiv:2505.09304).
Цена/риск: НЕ строить как замену «любой диктор» — только как опциональный второй режим/verifier; риск сломать speaker-independence. Средние усилия.

### (Б) Что исчерпано / диминишинг returns (НЕ повторять без существенной модификации)
- **Knowledge distillation в 32k student**: без выигрыша (нужно >400k позитивов) — согласуется с SSL-distillation литературой (требует больших корпусов: amazon.science KWS-KD, arXiv:2307.02720).
- **QAT**: не нужен — INT8 PTQ уже ≤0.7pp.
- **Focal loss / residual connections / арх-свип (BC-ResNet/TC-ResNet)**: регресс или не побили mixednet.
- **SLR28 RIR / агрессивная multi-condition аугментация**: пере-портит (плато ~20×); mit_rirs лучше.
- **Софт-микс музыки в обучение**: неверное распределение для barge-in (AEC убирает полногромкую музыку, KWS видит остаток) — фундаментально неверно, не повторять.
- **Adversarial mining раунды 2–3**: плато.
- **SOTA-архитектуры 2024–2026 (Keyword Mamba, Keyword Transformer)**: НЕ влезают в ~32k INT8 / 45 КБ arena (Keyword Mamba 3.4M, KWT-3 >5M params, arXiv:2508.07363) — нерелевантны вашему бюджету.

### (В) Где практический потолок
- **Средние метрики 2-сложной фразы**: вы на потолке. Recall 0.95, device-FAPH 0 с VAD ≈ референс Porcupine (<3% FRR при 1 FA/10ч, 97.1% точности при 10 дБ SNR — данные Picovoice). 
- **Длина фразы**: industry-практика рекомендует избегать слишком коротких фраз — Picovoice Porcupine FAQ: «you need to avoid using short phrases and make sure your wake word includes diverse sounds and at least six phonemes». Ваше протяжное «захааар» + большое окно — грамотная компенсация через длительность как дискриминатор.
- **Выигрыш теперь в WORST-CASE**, не в среднем: music_snr5, reverb+music10, muffled. Эти режимы закрываются реальными данными (Приоритет 1–2) + verifier (Приоритет 4), не loss/арх.
- **Приемлемые бытовые цели**: FRR <5% и FAR <0.5/час (openWakeWord и Picovoice как референс «below the annoyance threshold») — вы их уже достигли. Дальнейшее давление на средний FRR имеет отрицательный ROI; вкладывайтесь в данные/эксплуатацию/UX.

## Recommendations
**Стадия 1 (немедленно, низкая цена):**
1. Задеплоить готовый second-stage verifier как defense-in-depth (особенно для no-VAD сценариев). Бенчмарк: worst-case FAPH на music_snr5/reverb без регресса recall.
2. Запустить production-петлю логирования эмбеддингов промахов (privacy-safe). Триггер переобучения: накопление ≥N hard-negatives или дрейф device-FRR >7%.

**Стадия 2 (основной ROI, средняя цена):**
3. Собрать реальные long-tail device-данные через playback re-recording: idle-комната/тишина (негатив), barge-in через AEC, far-field, шёпот/крик, дети/пожилые. Бенчмарк: тишина-FAPH=0 на реальной idle-записи; recall на far-field >0.90.
4. Расширить до 100+ реальных дикторов. Бенчмарк: device-FRR на held-out незнакомых голосах <5%.

**Стадия 3 (модельный эксперимент, валидировать строго):**
5. Прототипировать настоящий angular-margin head (AM-Softmax/Sub-center ArcFace по arXiv:2211.00439), L2-norm в float32 при INT8. Порог принятия: worst-case device-FRR улучшается ≥3pp без роста FAPH; иначе откатить (как logit-shift).
6. PCEN-фронтенд — только если готовы форкнуть micro_speech; иначе отложить (high effort, фиксированный фронтенд блокирует).

**Что НЕ делать:** distillation, QAT, focal/residual, новые тяжёлые архитектуры, софт-микс музыки, SLR28 — исчерпано/нерелевантно.

**Что меняет приоритеты:** если device-FRR на незнакомых голосах окажется >10% после Стадии 2 — диктор-рычаг недостаточен, тогда персонализация (Приоритет 6) поднимается в приоритете. Если worst-case music-FAPH вырастет в проде — verifier и hard-neg mining критичны.

## Caveats
- **Относительные приросты с английских датасетов переносимы, абсолютные — НЕТ.** Все цифры (улучшения от angular-margin, 30–45% FRR↓ от playback-aug, scaling по дикторам ~100/2k) — индикативны; обязательна валидация на вашем leakage-safe device-eval на «захааар».
- **Точные таблицы метрик metric-learning и quantization не были извлечены из первоисточников в рамках данного исследования.** Конкретные значения относительного снижения EER от AM-Softmax, числа FRR@FAR, а также quant-числа (ResNet34 4-bit, attention-heavy ~15% rel.) и EdgeSpot (73.7%→82.0% @128k, arXiv:2601.16316) следует проверить по Table 2 первоисточников перед использованием как подтверждённых — направление эффекта надёжно, точные величины требуют верификации.
- **EdgeSpot (arXiv:2601.16316, ICASSP 2026, Analog Devices) не содержит per-component ablation** — нельзя изолировать вклад PCEN vs Sub-center ArcFace vs self-attention; известен только бандл и одно качественное замечание про PCEN. Не переоценивайте PCEN изолированно.
- **EdgeSpot/metric-learning числа — few-shot user-defined KWS**, не ваш fixed-phrase speaker-independent сценарий; перенос относительный, не прямой.
- **Speaker-verification quantization-числа — прокси**, не прямой замер вашей 32k-головы; публичного INT8-замера микроконтроллерной ArcFace KWS-головы найти не удалось.
- **Часть источников по персонализации — вендорский маркетинг** (Sensory, Picovoice, Kardome); цифры enrollment/размеров индикативны, не независимо верифицированы.