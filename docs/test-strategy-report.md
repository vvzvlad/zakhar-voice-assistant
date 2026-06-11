# Отчёт по тест-стратегии — zakhar-voice-assistant — 2026-06-11

**Область анализа**: последний мерж `faeaf5e` (ветка `pipeline-isolation`, рефакторинги R1–R6:
StageError-контракт STT/LLM, VAD как четвёртая стадия-плагин, канонический «+stress» контракт TTS,
транспортно-нейтральный StageEvent вместо VAET, единый `pipeline.speak()`, slow-флаги от источников тулов).
Предлагаются тесты только на **новое поведение мержа**, не покрытое тестами, добавленными в нём же.

## 1. Исполнительное резюме

- Проанализировано модулей: 5 (по диффу мержа)
- Предложено тестов (unit / integration / contract / E2E): **22 / 9 / 3 / 0** = 33
- Отклонено как малоценные / дубли / по skip-list: ~14 целей (тривиальные DTO, wiring, passthrough, уже покрытое)
- Покрытие сейчас (`pytest --cov=src`, все тесты зелёные): **89% общее**; затронутые мержем файлы: pipeline 95%, webrtc-VAD 100%, audio_prep 95%, tts 94%, tool_hub 94%, llm 97%, **esphome_client 66%**
- Мерж сам принёс хорошие тесты — остаточные дыры точечные, но среди них **два реальных бага** (см. §4)

Примечание к бюджету пирамиды: формально unit = 67% (< 70%), но все «integration»-тесты здесь —
in-process respx/фейки без внешних зависимостей, по стоимости равны unit. Unit+contract = 76%. E2E не нужны:
вся новая логика достижима через существующие фейки.

## 2. Рекомендации по модулям

### 2.1 Ядро пайплайна — `src/pipeline.py`, `pipeline_events.py`, `stage_errors.py` (R1, R4, R5)

Писать в `tests/test_pipeline.py` существующим харнессом (`make_pipeline`/`patch_llm`/`FakeAnnouncer`).

- **Unit добавить**:
  1. Не-StageError из LLM (`ValueError`) в `_run` ([pipeline.py:797-819](../src/pipeline.py#L797-L819)) — нет TTS_START/TTS_END, есть `ERROR(server_error)` + RUN_END, `error_stage="pipeline"`, сырой текст ошибки **не** озвучен. Ловит расширение/сужение `except StageError`.
  2. `speak()` при `send_announcement=None` ([pipeline.py:977-989](../src/pipeline.py#L977-L989)) — не падает, синтез и put выполнены. Единственный проактивный путь речи (R5), оба текущих теста биндят announcer.
  3. Полный прогон run при `send_event=None` ([pipeline.py:253-258](../src/pipeline.py#L253-L258)) — закрепляет транспортную нейтральность R4 (headless-прогон без транслятора).
  4. Payload `INTENT_END` == `{"conversation_id", "continue_conversation"}` ([pipeline.py:853-859](../src/pipeline.py#L853-L859)) — нигде не ассертится, а esphome_client шлёт его на устройство дословно.
  5. Филлер чистится `clean_llm_output` ([pipeline.py:786](../src/pipeline.py#L786)): контент `<think>…</think>Щас гляну` → озвучено ровно «Щас гляну». Сейчас замена на голый `.strip()` пройдёт весь сьют.
  6. `_spoken_llm_fallback` ([pipeline.py:570-575](../src/pipeline.py#L570-L575)): `kind="rate_limit"` → reply_rate_limit; неизвестный будущий `kind="timeout"` → reply_error (контракт R1 явно ожидает новые kinds).
- **Integration добавить**:
  7. Снэпшот conversation_id в сигнатуре `_run` ([pipeline.py:516-518](../src/pipeline.py#L516-L518)): STT-двойник посреди прогона вызывает `on_start("cid2")` → INTENT_END всё равно несёт «cid». Закрывает гонку, ради которой сигнатуру и меняли.
- **Contract добавить**:
  8. Канонический контракт LLM→TTS (R3): ответ LLM `"прив+ет, сейчас 5 °C"` доходит до `TtsBackend.synthesize` **дословно** (плюс на месте, без расширения единиц). Ловит возврат пайплайновой постобработки текста. *(Дедуплицировано: тот же тест предлагали оба аналитика — pipeline и TTS; оставлен один, на этом «нижнем» слое.)*
- **НЕ тестировать**: enum StageEvent (полнота маппинга уже contract-тестирована в test_esphome_client), конструктор StageError, read-through-свойства Pipeline, `serve_audio` отдельно (покрыт через speak), str-коэрция `_emit`.

### 2.2 VAD и подготовка аудио — `src/vad.py`, `src/plugins/vad/webrtc.py`, `src/audio_prep.py` (R2)

`audio_prep.py` — новый файл **без своего тест-файла**: 16 его unit-тестов живут внутри `test_pipeline.py:2163-2278`. Рекомендация: завести `tests/test_audio_prep.py`, перенести их туда (перенос, не дублирование) и добавить новые туда же.

- **Unit добавить**:
  1. `highpass`: нечётный хвостовой байт сохранён; входы < 4 байт возвращаются байт-в-байт; нечётное число сэмплов не меняет длину ([audio_prep.py:17](../src/audio_prep.py#L17), непокрытая строка 30). Ловит классику `np.fft.irfft` без `n=`.
  2. `highpass`: тон 30 Гц (ниже 0.5×cutoff) теряет почти всю RMS-энергию — текущие тесты не заметят инверсию маски (DC-тест пройдёт и при «пропускаем низы, режем DC»).
  3. `normalize_peak`: нечётный хвостовой байт сохранён, 1-байтовый вход без исключения (непокрытая строка 55).
  4. `write_wav`: round-trip через `wave` — mono/16-bit/16000, фреймы байт-в-байт ([audio_prep.py:83](../src/audio_prep.py#L83)). *Поправка к отчёту аналитика: функция исполняется тестами через pipeline (coverage подтверждает), но заголовки WAV нигде не ассертятся — дрейф framerate сьют не поймает, тест оставлен.*
  5. `pcm_to_wav_bytes`: стерео с нечётными pcm/pcm2 (хвост отброшен по контракту), mono с пустым pcm → валидный 0-фреймовый контейнер (непокрытая строка 115).
  6. `WebRtcVadSession.feed` — приоритет финализации ([webrtc.py:117-122](../src/plugins/vad/webrtc.py#L117-L122)): оба порога в одном чанке → `"endpoint"`, не `"maxlen"`; `no_speech_timeout==max_utterance` при тишине → `"maxlen"`. Каждый текущий тест пересекает ровно один порог — перестановку if-цепочки никто не поймает.
  7. `_vad_boost` третья ветка gain≤1 ([webrtc.py:52-53](../src/plugins/vad/webrtc.py#L52-L53)): громкий фрейм возвращается байт-идентично.
  8. `feed` с `auto_gain=True` и чанком 641 байт — без исключения, остаток перенесён ([webrtc.py:87-93](../src/plugins/vad/webrtc.py#L87-L93)).
  9. `WebRtcVadConfig`: aggressiveness 4 и −1 → ValidationError (снятие ge/le уронит каждый голосовой прогон на `webrtcvad.Vad(4)`).
- **Integration добавить**:
  10. Бэкенд с **реальным** webrtcvad: нулевые 640-байтовые фреймы → `"no_speech"` ровно на таймауте. Ловит дрейф FRAME_BYTES/SAMPLE_RATE против допустимых C-библиотекой размеров фреймов (текущий тест делает только isinstance).
- **Contract добавить**:
  11. Параметризованный контракт стадии vad по всем зарегистрированным провайдерам: `create→open→feed(b"")` возвращает None или одно из `{"endpoint","maxlen","no_speech"}`, `debug_state()` — dict. Страховка под будущий silero.
- **НЕ тестировать**: `EndpointPolicy` (frozen DTO), `debug_state` по умолчанию, константы, `plugins/vad/__init__.py`, поведение самого webrtcvad.

### 2.3 TTS-контракт текста — `src/tts.py`, `_ru_text.py`, `llm_text.py`, `llm.py` (R3)

- **Unit добавить**:
  1. `PiperTtsBackend._synth` ([tts.py:221-226](../src/tts.py#L221-L226)): записывающий `_StubVoice`; вход `"чт+о нового? 50% и м/с"` → в `synthesize_wav` пришло «што́»/«процентов»/«метров в секунду», без «+». Ловит выпадение/перестановку цепочки адаптации (phonetic до stress ломает «что»→«што»); текущие Piper-тесты используют тексты, которые цепочка не меняет.
  2. `call_llm_api` — раунды исчерпаны, но есть контент ([llm.py:198](../src/llm.py#L198), **строка 199 непокрыта по coverage**): все ответы с tool_call + контентом → возврат `clean_llm_output(last_content)`, а не StageError(tool_rounds).
- **Integration добавить** (respx, стиль существующих тестов test_tts.py):
  3. `TeraTtsHttpBackend.synthesize` ([tts.py:115-122](../src/tts.py#L115-L122)): в URL запроса — «што́»/«процентов», нет литерального «+», U+0301 percent-encoded. Цепочка адаптации добавлена этим мержем и сейчас полностью непроверена (тесты шлют «привет», которое цепочка не меняет).
  4. `YandexTtsBackend.synthesize` ([tts.py:314-319](../src/tts.py#L314-L319)): в POST-теле «прив+ет» сохранён, «%»→«процентов», **нет** phonetic-хаков («что» нетронуто). Ловит «унификацию» цепочек Piper/Yandex.
  5. Yandex: пустой/только-пунктуация вход → POST не выполнен, результат `("audio/mpeg", b"")` ([tts.py:319-324](../src/tts.py#L319-L324)) — иначе Yandex вернёт 400 вместо тишины.
- **НЕ тестировать**: провайдеры piper/teratts (wiring), каталог голосов Yandex, дефолт `reply_error` в LlmConfig, ABC.

### 2.4 Хаб тулов — `src/tool_hub.py`, `src/tool_factory.py` (R6)

Мерж покрыл R6 хорошо; остались 3 unit:

1. `is_slow` после `set_sources` hot-swap ([tool_hub.py:193,228](../src/tool_hub.py#L228)): флип slow вслед за заменой источника; удалённый тул → False, не stale True. Это ровно сценарий панели «toggle slow → reconfig».
2. `is_slow` при коллизии имён: классификация следует **победителю** коллизии, на которого реально маршрутизирует `call()` (существующий тест коллизии проверяет только маршрутизацию).
3. `is_slow` на duck-typed источнике без атрибута `slow` — **блокирован рефакторингом R-A** (§5): сейчас упадёт AttributeError.

**НЕ тестировать**: поле `ToolSource.slow` само по себе, присваивания в `__init__`, поле `McpServerConfig.slow` (DTO), филлер-политику пайплайна (покрыта).

### 2.5 Транспорт и wiring — `esphome_client.py`, `app.py`, `reconfig.py`, `stt.py` и др.

- **Unit добавить**:
  1. `action_for("llm.instances.openrouter.reply_error") == "live"` ([reconfig.py:43](../src/reconfig.py#L43)) — новый лист `_LLM_LIVE_LEAVES` не представлен в табличном тесте.
  2. `migrate_vad_plugin` на битых конфигах ([app.py:29](../src/app.py#L29)): нет ключа `core`; `core["vad"]` null/не-dict; слот `vad` без `instances`. Миграция бежит до всего в `main()` — краш здесь = кирпич на старте.
- **Integration добавить**:
  3. `DeviceClient.play_media` ([esphome_client.py:319-331](../src/esphome_client.py#L319-L331)) — **0% покрытие подтверждено** (строки 325-329 missing): фейковый `pipeline.serve_audio` фиксирует порядок аргументов `(mime, audio)` (старый код был `put(audio, mime)` — реверс при рефакторинге отдаст mime-строку как аудио), announcement вызван с `media_id=url, timeout=30.0`; offline → RuntimeError до serve_audio.
  4. `apply_job` с vad-путём ([reconfig.py:198](../src/reconfig.py#L198)): `rt.vad_backend` заменён, stt/llm/tts нетронуты — сейчас никто не гонит vad сквозь apply_job, hot-reload 4-й стадии может молча отвалиться.
  5. `_rebuild_http` union с offline-vad ([reconfig.py:341](../src/reconfig.py#L341)): коалесцированная джоба `{proxy, vad.*}` пересобирает vad ровно один раз.
- **Contract добавить**:
  6. `VoskSttBackend.transcribe` соответствует контракту ABC «raises StageError('stt')» — **блокирован фиксом R-B (§5), тест сейчас красный против кода**: сырое исключение декодера пролетает мимо `except StageError` в [pipeline.py:722](../src/pipeline.py#L722).
- **НЕ тестировать**: `Runtime.__init__`, wiring `main()`, `STAGE_CATEGORIES`, удаление `VadConfig.aggressiveness`.

## 3. Сквозные аспекты

- **Contract-тесты**: маппинг StageEvent→VAET уже contract-тестирован (полнота по членам enum) — образец для предложенного vad-провайдер-контракта (§2.2 п.11).
- **Property-based**: кандидаты — `highpass`/`normalize_peak`/`pcm_to_wav_bytes` (инварианты «длина выхода == длине входа», «нечётный хвост сохранён/отброшен по контракту») через hypothesis, если появится в dev-deps. Не блокирует план.
- **Test-data factories**: уже есть (`make_pipeline`, `DeviceClient.__new__`-фейки, `_StubVoice`, `_job_runtime`) — новые тесты в них вписываются, новых фабрик не нужно.

## 4. Обнаруженные антипаттерны и баги

1. **Баг (high)**: `VoskSttBackend` нарушает новый StageError-контракт ([stt.py:138-148](../src/stt.py#L138-L148)) — Groq мигрирован, Vosk нет; сырое исключение из `to_thread(_decode)` оставит висящую voice_assistant-сессию устройства без балансирующих STT_END/RUN_END.
2. **Баг (medium)**: асимметрия `is_slow` ([tool_hub.py:231](../src/tool_hub.py#L231)) — голый `source.slow` против защитного `getattr(..., "slow", False)` в `describe()` (строка 250); AttributeError влетает в филлер-гейт [pipeline.py:784](../src/pipeline.py#L784) и роняет прогон, нарушая доктрину «tool call never raises».
3. **Несоответствие (low)**: mono-ветка `pcm_to_wav_bytes` пишет нечётный хвостовой байт в data-chunk (стерео-ветка его отбрасывает) — решить и запинить тестом.
4. **Тест-смеллы**: тесты [test_reconfig.py:76,122](../tests/test_reconfig.py#L76) пинят путь `core.vad.aggressiveness`, удалённый этим же мержем (заменить на живой `core.vad.silence_ms`); `_piper_chain` в [test_ru_text.py:11](../tests/test_ru_text.py#L11) дублирует продакшен-композицию — дрейф продакшена тесты не поймают; точное dict-равенство `debug_state()` в [test_vad_webrtc.py:137](../tests/test_vad_webrtc.py#L137) — хрупкий снапшот; 16 тестов audio_prep живут в чужом файле test_pipeline.py.
5. **Открытые вопросы поведения** (не пинить тестом до решения автора): endpoint, «промахивающийся» внутри большого чанка при возобновлении речи в его хвосте ([webrtc.py:116](../src/plugins/vad/webrtc.py#L116) — докстринг говорит «зеркалит старую inline-логику», возможно намеренно); `clean_llm_output` оставляет незакрытый `<think>` (унаследовано из удалённого text.py, вне скоупа мержа); `expand_units` матчит только кириллическую «°С», не латинскую «°C».

## 5. Необходимые рефакторинги перед написанием тестов

- **R-A**: `is_slow` → `getattr(source, "slow", False)` (1 строка, [tool_hub.py:231](../src/tool_hub.py#L231)) — блокирует тест §2.4 п.3. Это и фикс бага №2.
- **R-B**: обернуть ошибки `_decode` Vosk в `StageError("stt", ...)` по образцу Groq ([stt.py:138-148](../src/stt.py#L138-L148)) — блокирует contract-тест §2.5 п.6. Фикс бага №1.
- Опционально (не блокирует): извлечь дублированную цепочку Piper/TeraTTS в именованную `adapt_for_espeak()` в `_ru_text.py` (3 текстовые копии, включая копию в тестах); перенести audio_prep-тесты в `tests/test_audio_prep.py`.

## 6. План внедрения

- **Фаза 1 — баги + их тесты** (наивысший ROI: реальные падения в проде): R-A + R-B, затем contract-тест Vosk (§2.5 п.6), unit is_slow без атрибута (§2.4 п.3), integration `play_media` (§2.5 п.3 — единственный полностью непокрытый публичный метод мержа), unit LLM non-StageError boundary (§2.1 п.1).
- **Фаза 2 — контракты рефакторинга** (то, ради чего делался мерж): +stress passthrough (§2.1 п.8), цепочки TeraTTS/Yandex/Piper (§2.3 п.1,3,4), приоритет финализации VAD (§2.2 п.6), транспортная нейтральность `send_event=None` (§2.1 п.3), vad через `apply_job`/`_rebuild_http` (§2.5 п.4,5), hot-swap/коллизии is_slow (§2.4 п.1,2).
- **Фаза 3 — края и гигиена**: остальные unit audio_prep/webrtc (§2.2), speak без announcer, INTENT_END payload, филлер-clean, fallback-kind, conversation-id race, миграция битых конфигов, Yandex empty-input, llm rounds-with-content, `action_for(reply_error)`; перенос audio_prep-тестов, чистка смеллов §4 п.4.

## 7. Источники

- Отчёты 5 аналитиков (`module-testability-analyst`): pipeline-core, VAD/audio_prep, TTS-контракт, tool hub, транспорт/wiring.
- Независимая верификация: `pytest --cov=src --cov-report=term-missing` (.venv, все тесты зелёные, 89% total). Подтверждено: непокрыты [llm.py:199](../src/llm.py#L199), [esphome_client.py:325-329](../src/esphome_client.py#L325-L329); опровергнуто заявление «write_wav никогда не исполняется» (строки 83-91 покрыты через pipeline; непокрыты только заголовочные ассерты — тест сохранён с пометкой).
- Фильтрация: дедупликация — 1 (двойной contract-тест +stress); skip-list — ~14 целей отклонено аналитиками и оркестратором; adversarial-проверка — все 33 выживших теста падают при реалистичной молчаливой поломке (для каждого назван класс дефекта).
