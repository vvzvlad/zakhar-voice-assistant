# Отчёт по тест-стратегии — zakhar-voice-assistant — 2026-06-09

> Голосовой ассистент: ESP32-колонка (ESPHome Native API) → STT → LLM (agentic
> tool-calling) → smart-home через MCP → TTS. Backend на Python 3.11 (pytest,
> asyncio), панель управления на React/Vite. Тренировочный пайплайн в
> `microWakeWord/` (offline ML-скрипты, недетерминированные, запускаются вручную)
> **выведен из периметра тест-стратегии рантайма** — см. §4.
>
> **Это обновление** отчёта от 2026-06-08 (коммит `a0e97fb`) на текущий HEAD
> `babad06`. С момента прошлого отчёта внедрена большая часть предложенных тестов
> (см. §«Что изменилось»), переписан `pipeline.py` (+828 строк), добавлен модуль
> горячей перезагрузки конфига `reconfig.py`/`runtime.py` и ~3500 строк тестов.

## 0. Что изменилось с прошлого отчёта (внедрено / разрешено)
- **Покрытие backend 83 % → 86 %** (2896 stmts / 419 miss, проверено `pytest --cov`).
- **device-io вытащен из ямы:** `esphome_client.py` **39 %→70 %** (добавлен
  `test_esphome_client.py`, 343 стр.), `audio_server.py` **72 %→97 %**.
- **Внедрены error-ветки pipeline/panel/config/builtin/tool-hub** — большинство
  unit-кандидатов прошлого отчёта закрыты (см. §2 «уже покрыто» по модулям).
- **Новый модуль hot-reload** `reconfig.py` (95 %) + `runtime.py` (100 %) +
  `tool_factory.py` (100 %) с суитой `test_reconfig.py` (1020 стр.).
- **Рефакторинги, которые уже сделаны (закрыты):**
  - ~~R-DEV-1~~ — `tts_url(base_url, audio_id, mime)` вынесен в единый хелпер
    [audio_server.py:14](src/audio_server.py#L14); дубля mime→ext больше нет
    (`pipeline.py` и `esphome_client.py` импортируют один хелпер).
  - ~~R-DEV-2~~ — маршрутизация `DeviceManager.announce` покрыта (`test_reminders.py`).
  - ~~R-PIPE-1~~ — обрезка по HARD_CAP покрыта (`test_normal_run_hard_cap_truncates_at_60s`).

## 1. Исполнительное резюме
- **Проанализировано модулей:** 9 (по одному `module-testability-analyst` на модуль).
- **Предложено НОВЫХ тестов (unit / integration / E2E / contract):** **~71 / 9 / 0 / 1**
  (итого ~81). Остаточные гэпы после массового внедрения прошлого цикла.
- **Отклонено как малоценные** (skip-list / уже покрыто / тривиально): ~55 кандидатов.
- **Покрытие сейчас:** backend **86 %** (проверено `pytest --cov`, цифры аналитиков
  сверены построчно); frontend **0 %** (тест-раннера по-прежнему нет).
- **Прогноз после внедрения:** backend **~94 %** (`app.py`-bootstrap и `version.py`
  git-обвязка сознательно остаются; главный подъём — `tts.py`/`stt.py` после
  R-TTS-1/R-STT-1 и reconnect-машина `esphome_client`); frontend — покрыта вся
  чистая логика (`schema.js`, `runsModel.js`, `format.js`, `api.js`, `useStageForm.js`).
- **Бюджет пирамиды соблюдён:** unit ~88 % (≥70 %), integration ~11 % (≤20 %), E2E 0 (≤10).

## 2. Рекомендации по модулям

### config-core — `config_store.py` (94 %), `config_service.py` (95 %), `core_config.py` (100 %)
Почти полностью покрыт; остаточные гэпы — поведения, не строки.
- **Unit-тесты добавить:**
  - `apply` deep-merge сохраняет соседей: частичный patch `tts.instances.yandex.voice`
    **не** затирает `stt.instances.groq.api_key` и `core.openweathermap.city` (на диске и в `get()`).
    Ловит: классическую потерю данных при «улучшении» `_deep_merge` до shallow-update.
  - `provider(cat)` ([config_service.py:118-120](src/config_service.py#L118)) возвращает
    именно выбранный provider-объект (assert по типу, не `is not None`). Ловит: тихую
    подмену на `get()`/`create()` — ломается только в момент hot-reconfig в проде.
  - `config_store.save` глотает `OSError` fsync директории ([config_store.py:70-71](src/config_store.py#L70))
    → файл всё равно записан (`load==doc`). Ловит: «save сообщает об ошибке, хотя файл записан».
  - *(условно)* annotation core-схемы через allOf/anyOf fallback ([config_service.py:55-61](src/config_service.py#L55)) —
    **только если** live-схема CoreConfig реально эмитит allOf/anyOf для секций с default;
    иначе это защитный мёртвый код → skip. Проверить эмитируемую JSON-схему перед написанием.
- **НЕ тестировать:** DTO-дефолты `core_config` без валидаторов; `document()/_slot()/core`
  (passthrough); ctor-валидация (покрыто `test_constructor_validates_selected_instances`);
  `json.dump`/`os.replace` (stdlib).

### reconfig-runtime — `reconfig.py` (95 %), `runtime.py` (100 %), `app.py` (0 %)
Hot-reload отлично покрыт (45 тестов): чистые `changed_paths`/`action_for`/`backend_categories`,
push-before-create, latch частичных сбоев, coalescing бурста — всё есть. Единственный гэп — `app.py`.
- **Unit-тесты добавить:**
  - `app.load_or_create_config` ([app.py:29](src/app.py#L29)) — первый boot: пустой
    `config_store.load()` → читается `templates/default_config.json`, `save()` вызван, возвращён template.
    Ловит: сломанный seed.
  - `app.load_or_create_config` — существующий конфиг возвращается **как есть**, `save` не зовётся,
    template не открывается. Ловит: клоббер пользовательского конфига на каждом старте. **Без рефактора.**
  - Boot-предупреждения: legacy `core.mcp` ([app.py:49-53](src/app.py#L49)) и пустой
    `public_base_url` ([app.py:81-86](src/app.py#L81)) логируют warning; иначе молчат. **Требует R-APP-1.**
    Ловит: потерю сигнала «колонки ничего не воспроизведут» / «legacy-ключ игнорируется».
- **НЕ тестировать:** `main()` end-to-end (DI/сокеты/zeroconf/модели); `finally`-teardown
  (порядок `stop()` — тавтология); `Runtime.core`/`llm_cfg` (passthrough); 4 missing-строки
  `reconfig.py` (165-166 QueueEmpty-guard, 286-288/334-336 клоны latch-паттерна, 378-379 log-only) —
  defensive/дублирующие, низкая ценность.

### stage-plugins — `src/plugins/**`, `llm.py` (96 %), `stt.py` (76 %), `tts.py` (69 %)
Главный остаточный гэп проекта — **`tts.py` (69 %)** и **`stt.py` (76 %)**: непокрыты
`PiperTtsBackend._synth` (105-137) и `VoskSttBackend` (85-118), потому что тяжёлая загрузка
модели идёт **в конструкторе** (`PiperVoice.load` / `vosk.Model()`) и не запускается в CI.
- **Unit-тесты добавить (требуют R-TTS-1 / R-STT-1):**
  - `PiperTtsBackend._synth`: сшивка предложений + пауза `sentence_silence` (точный счётчик
    silence-фреймов на границе); все-непроизносимое → валидный тихий WAV, не краш; пустой
    фрагмент пропускается. **Требует R-TTS-1.**
  - `VoskSttBackend.transcribe`: пустой PCM → `""` без обращения к модели; `_decode` извлекает
    и `.strip()`-ает `text` из JSON распознавателя, нет ключа → `""`. **Требует R-STT-1.**
- **Unit-тесты добавить (БЕЗ рефактора — `_decode_v3_audio` уже чистая):**
  - Несколько NDJSON-чанков `audioChunk` конкатенируются по порядку ([tts.py:155-168](src/tts.py#L155));
    объект `error` → `RuntimeError` (а не тихое пустое аудио); JSON-массив объектов принимается.
    Ловит: проглоченную ошибку Yandex v3-стрима, переупорядочивание/потерю чанка.
- **НЕ тестировать:** `Provider.create`/`options` базы (no-op); offline `create()` (passthrough-конструктор);
  `llm.py:82-83` (`reason=None` — дегенеративная ветка, тест проходит даже если её сломать),
  `llm.py:160` (exhausted-with-content — почти тавтология к `test_max_tool_rounds_exhausted`);
  `PiperVoice`/`vosk`/`lameenc` внутренности; уже покрытые `split_sentences`/`yandex_stress_markup`/
  `wav_to_mp3`/`pcm_to_wav`/Groq STT/OpenAICompat.

### pipeline-orchestration — `pipeline.py` (93 %), `context.py` (80 %), `runs_store.py` (96 %), `run_events.py`/`prompt.py`/`text.py`/`run_context.py` (100 %)
93 % — честные; непокрытое почти целиком — **ветки изоляции сбоев** (тихие регрессии: прогон
«успешен», но теряет наблюдаемость/аудио). Рефакторинг не нужен — все ветки достигаются через инъекцию.
- **Unit-тесты добавить (наивысшая ценность — телеметрия ошибок):**
  - `_run` catch-all ([pipeline.py:704-713](src/pipeline.py#L704)): исключение STT/LLM → эмит
    `VOICE_ASSISTANT_ERROR(code="server_error")` + `RUN_END`, запись `result="error"`,
    `error_stage="pipeline"`. **Самый важный непокрытый блок** — гарантия, что колонка
    вернётся в idle при любом сбое.
  - Изоляция: сбой `runs_store.insert` ([722-723](src/pipeline.py#L722)) → прогон завершается,
    нет run_id ⇒ нет broadcast; сбой `put_audio` ([741-742](src/pipeline.py#L741)) → запись есть,
    `has_audio` ложно; сбой `broadcast` ([763-764](src/pipeline.py#L763)) → запись одна, без проброса;
    сбой raw-capture на диск ([550-551](src/pipeline.py#L550)) → прогон жив.
  - `disarm_capture` ([308-322](src/pipeline.py#L308)): Future таймаут-вызвавшего API резолвится
    `RuntimeError("capture cancelled")`, busy-guard снят. Сейчас 0 %.
  - `on_audio` capture-maxlen + поздний чанк ([393-394, 408-410](src/pipeline.py#L408)): finalize по
    байтовому cap (`reason="maxlen"`) и игнор второго чанка после finalize (буфер не растёт).
  - `context.load_context` OSError на open после exists/stale ([context.py:59-61](src/context.py#L59)) → `[]`;
    `append_context` OSError на запись ([92-93](src/context.py#L92)) → None без проброса.
  - `runs_store._percentile`/`_avg` на непустом окне со всеми `None`-таймингами ([175-176/245/258](src/runs_store.py#L245))
    → `p50/p95=None`, не 0 и не краш (панель показывает «нет данных», а не «0 ms»).
- **НЕ тестировать:** `_fail_capture_future("superseded")` (недостижимый safety-net, busy-guard ловит раньше);
  `_write_wav`/`_pcm_to_wav_bytes` (обёртки `wave`, покрыты транзитивно); `@property`-passthrough к `rt`;
  `append_context`-обёртка в `_run` ([660-662](src/pipeline.py#L660)) — дублирует unit context.py;
  `runs_store.get` rounds_json parse-error (175-176 — достижимо лишь при внешней порче БД);
  root `main.py` (тривиальный entrypoint: читает log_level → `app.main()`).

### tool-hub-mcp — `tool_hub.py` (84 %), `mcp_client.py` (85 %), `tool_factory.py` (100 %)
`tool_factory` закрыт полностью. Остаток — формы результата SDK и graceful-degradation.
- **Unit-тесты добавить:**
  - `BuiltinMcpSource._normalize` ([tool_hub.py:118-135](src/tool_hub.py#L118)): три формы —
    tuple `([content],{result})`, dict с/без `result`, не-iterable → `TypeError`-fallback `str(res)`.
    Ловит: модель получает Python-repr вместо текста.
  - `BuiltinMcpSource.call` ([113-116](src/tool_hub.py#L113)): исключение инструмента → строка
    `"error calling …"`, не краш agentic-loop (+ happy-path через `_normalize`).
  - Fault-isolation `ToolHub.ensure_tools` ([215-216](src/tool_hub.py#L215)), `stop` ([260-264](src/tool_hub.py#L260),
    в т.ч. источник без `stop`), `set_sources` остановка старого источника ([204-208](src/tool_hub.py#L204)).
  - `McpToolHub.start` недоступный smart-home MCP ([62-64](src/mcp_client.py#L62)) → `tools=[]`, без проброса;
    `ensure_tools` self-heal reload при пустом кэше ([66-76](src/mcp_client.py#L66)).
- **Contract-тест (единственный в проекте, gated):** `test_mcp_smarthome_wire_contract` —
  записанный JSON-RPC ответ Node-RED (`list_tools`+`call_tool`) через **реальный** декодер
  `mcp.ClientSession` (respx-транспорт, без StubSession). Ловит: дрейф wire-формата → тихо пустой
  список инструментов; текущие StubSession-тесты этого структурно не ловят. **Требует R-MCP-1
  (записанный фикстур, с привязкой версии SDK). Без фикстура НЕ писать** — иначе пере-тест SDK.
- **НЕ тестировать:** абстрактные `NotImplementedError` базы; `HttpMcpSource.*` (passthrough к `McpToolHub`);
  `_to_groq_tool`/`_resolve_transport` (покрыто); `McpToolHub.stop` (no-op `return None`); `mcp`-библиотека.

### builtin-mcp — `calendar.py` (89 %), `builtin_mcp/reminders.py` (88 %), `reminders.py` (93 %), `openweathermap.py` (91 %)
Очень хорошо покрыт. Реальные остаточные гэпы:
- **Unit-тесты добавить:**
  - `_to_yandex_utc` наивный datetime → trailing `Z` через локаль→UTC ([calendar.py:339-342](src/builtin_mcp/calendar.py#L339)).
    Ловит: сдвиг wall-clock на UTC-offset. *Ожидаемое UTC считать из того же naive-входа (детерминизм по TZ CI).*
  - `_event_dict` нечисловой `PRIORITY` → поле опускается ([262-265](src/builtin_mcp/calendar.py#L262));
    `_format_rrule`/`_parse_until` malformed (часть без `=`, пустой `UNTIL`) → внятный `ValueError`
    ([54, 126-127](src/builtin_mcp/calendar.py#L126)).
  - `_format_due` past-due/due-now ([builtin_mcp/reminders.py:23-25](src/builtin_mcp/reminders.py#L23)) →
    без суффикса «(через N мин)», не «через -2 мин».
- **Integration-тесты добавить (in-process, event-driven, БЕЗ hard-sleep):**
  - `ReminderScheduler._loop` при `deliver is None` ([reminders.py:236-240](src/reminders.py#L236)) →
    one-shot снят, warning, цикл жив. *Документирует осознанную потерю — см. §7.*
  - `ReminderScheduler._loop` `deliver` бросает ([243-250](src/reminders.py#L243)) → поймано, **второй**
    reminder всё равно доставлен (цикл не умер). **Высшая ценность** — иначе один сбой глушит все будущие.
- **НЕ тестировать:** `caldav`/`icalendar` адаптеры; повторяющийся `except → "Не удалось…"` по всем
  инструментам (доказан один раз `test_tool_error_returns_string_not_crash`); `_principal`/`_component_of`;
  `openweathermap` parse-error (47-49, json-внутренности); 100 %-покрытые куски.

### device-io — `esphome_client.py` (70 %), `audio_server.py` (97 %)
`audio_server` практически готов. Остаток `esphome_client` — **reconnect-машина** (не announce-маршрутизация,
она уже покрыта в `test_reminders.py`). Поведенческие тесты пишутся **сегодня** через `__new__`-bypass.
- **Unit-тесты добавить:**
  - `DeviceClient._on_connect` happy ([59-78](src/esphome_client.py#L59)): online-флаг, ре-байнд
    `send_event`/`send_audio`, ре-подписка (`_unsub`), `_discover_capture_keys`.
  - `DeviceClient._on_connect` сбой `device_info` ([65-66](src/esphome_client.py#L65)) **проглатывается** →
    подписка всё равно идёт, `online=True`. **Самый ценный тест** — иначе транзиентный сбой оставляет
    колонку навсегда offline/без подписки.
  - `DeviceClient._on_disconnect` ([80-84](src/esphome_client.py#L80)): `_unsub=None`, `online=False`
    (иначе мёртвая колонка принимает `capture`/`announce`).
  - `DeviceClient.announce` ([160-172](src/esphome_client.py#L160)): язык `"ru"`, mime↔url через `tts_url`,
    `media_id`/`timeout=30.0`; offline-guard → `RuntimeError`, синтез не зван.
  - `DeviceManager.statuses` ([201-206](src/esphome_client.py#L201)): `{name,host,online}` в порядке клиентов;
    `announce(None)` без online-колонок ([214-218](src/esphome_client.py#L214)) → drop без проброса (единственная
    непокрытая ветка маршрутизации).
- **НЕ тестировать:** `_handle_start/_handle_audio/_handle_stop` (passthrough к `pipeline.on_*`);
  `start` (делегат к `ReconnectLogic`); `__init__` присвоения; `audio_server._handle_health` (константа);
  prune-тело (46, покрыто транзитивно TTL-тестами); `tts_url` (единый хелпер, mime↔ext покрыт; ветка `flac` —
  максимум один маргинальный add).

### panel-api — `panel_api.py` (87 %), `logging_setup.py` (91 %), `version.py` (0 %)
Главная внешняя write-поверхность (мутация конфига). Сильно покрыта; остаток — мелкие error/routing-ветки.
- **Integration-тесты добавить (реальный aiohttp TestServer):**
  - `_patch_config` ветка `ValueError` (≠ `ValidationError`) ([130-131](src/panel_api.py#L130)) → 422 с
    `detail==[]` (фронт различает две формы 422), документ не изменён. *Если ни один реальный patch не даёт
    «голый» ValueError — провести через стаб `svc.apply` на unit-уровне (см. §7).*
  - `_post_capture` не-dict тело ([205-206](src/panel_api.py#L205)) → 400, capture не зван (иначе
    `AttributeError`→500).
  - `_get_runs` clamp `limit` ([271-274](src/panel_api.py#L271)): `abc`→fallback 100, `0`→1, `99999`→500;
    **ассертить значение, дошедшее до store** (иначе тест проходит при сломанном clamp).
  - `_get_run` невалидный id на «голом» роуте ([285-287](src/panel_api.py#L285)) → 400 `invalid id`
    (сейчас 400-bad-id покрыт только для `/audio`).
  - SPA-статика + `(?!api/)` catch-all ([342-344, 369-377](src/panel_api.py#L369)): `/`→index, deep-link→index,
    `/assets/*`→ассет, **`/api/nope`→404 JSON** (lookahead не должен поглощать API).
- **Unit-тест добавить:** `version._resolve_version` precedence ([version.py:53-55](src/version.py#L53)):
  git→stamp→`0.0.0+unknown` (monkeypatch двух приватных хелперов, **без** запуска git).
- **НЕ тестировать:** `_add_cors` allowlist-ветка (осознанно пустой frozenset, мёртвая); `_post_capture`
  generic-500; `_runs_stream` тело `async for … pass`; `start/stop` (aiohttp lifecycle); `build_app`/таблица
  роутов (покрыты транзитивно); `version._from_git`/`_from_stamp` (subprocess/файл-обвязка, недетерминизм);
  `logging_setup.emit` ValueError-fallback (30-31, маргинальная защита).

### frontend — `frontend/react-export/src/**` (React/Vite, **0 тестов, раннера нет**)
**Prerequisite P-FE-1:** поставить Vitest (переиспользует `vite.config.js`/`import.meta.env`), `test`-скрипт,
node-env по умолчанию + `jsdom` для 2 компонент-групп. **Без него ни один тест не запустится.**
- **Unit-тесты (наивысший ROI, node-env, без DOM):**
  - `schema.js` (виджет-резолвер, security-критичный): `resolve`/`deref` (`$ref`→`$defs`, Optional
    `anyOf:[X,{null}]`→не-null ветка, sibling `default` сохраняется, циклический ref стоп на 10-й,
    неизвестный ref→`{}`); `enumOf` (top-level / в `anyOf` / нет→null); `isSecret`+`humanize`. Ловит:
    поле рендерится **не тем виджетом** (api_key открытым текстом — утечка; enum как текст-инпут).
    *Adversarial:* для `isSecret` обязателен false-positive кейс (`/key/i` ловит `keyboard_layout` — см. §4).
  - `runsModel.js`: `mapRun` (null→null; `audio`/`error` суб-объекты; `t_*`→0); `totalMs`
    (`t_total:0` сохраняется, не подменяется суммой); `fmtSec` (null/NaN→«—»).
  - `format.js`: `fmtUptime` (границы 86400/3600 c, отрицательное→0, null→«—»); `fmtStarted`
    (валид/невалид→raw/falsy→«—»). *Требует фикс TZ — R-Tool-2.*
  - `primitives.segsFor`/`total`: waterfall-сегменты, спец-случаи `empty`/`error`, порог метки 15 %,
    `total({})===0`.
  - `api.js`: `request` (2xx JSON; 204→null; non-2xx `{detail}`/`{error}`/не-JSON→`ApiError.status`;
    reject→`status:0`); `getRuns` query-builder (пустые/null выкинуты, `limit:0` **сохранён**, URL-кодирование).
  - `useStageForm.errorLines`: pydantic `detail[]`→строки, префикс `"body"` срезан, fallback `err.message`.
  - `log.matchesFilters` + `stages.matchPreset` — **требуют R-FE-1 (экспорт / вынос в чистый модуль).**
- **Component/jsdom-тесты (минимум, ассерт идентичности виджета и payload `onChange`, НЕ разметки):**
  - `SchemaForm.SchemaField`: secret→`<input type=password>`; enum ≤3→Seg / >3→Select (граница);
    bool+enum→Toggle; integer+min/max+`widget:slider`→Slider иначе Stepper, целое округляется.
    Ловит: ключевой silent-break — API-ключ открытым текстом / float в int-поле→422.
  - `SchemaForm.schemaNeedsRestart` (`apply:"restart"`→true, уважает `skip`).
- **НЕ тестировать:** презентационные `primitives`/`Sidebar`/`Topbar`/`icons`/страницы-оболочки
  (тавтологичные снапшоты); `useStageForm` хук-тело (обёртка над useState/useMemo); WebSocket
  `openRunsStream`/blob-download `postCapture`/`downloadRunAudio` (браузерные API/таймеры); `nav.js`/
  `stageMeta.js`/`navStore.js` (статические таблицы); `App.jsx`/`main.jsx`/`appData.jsx` (проводка).

## 3. Сквозные аспекты
- **Contract-тесты:** ровно один — wire-формат внешнего Node-RED MCP (§2 tool-hub-mcp), gated на
  записанный фикстур R-MCP-1. Frontend↔backend JSON-контракт закрепляется на стороне `panel_api.py`;
  `mapRun`/`matchesFilters` — consumer-зеркало, отдельный contract не нужен.
- **Property-based (опц., низкий приоритет):** `config_service._deep_merge` (идемпотентность с пустым
  patch; сохранение соседей); `tts.split_sentences` (конкатенация фрагментов = исходный текст без разделителей).
- **Дымовые/нагрузочные:** не применимо (LAN-сервис на N колонок, не публичный web).
- **Test-data factories:** `make_config_doc(**overrides)`; `make_run_row(**overrides)` (backend `runs_store`
  ↔ frontend `mapRun`); записанные MCP-фикстуры (`list_tools`/`call_tool`) для contract-теста.

## 4. Обнаруженные антипаттерны
- **Тяжёлая загрузка моделей в конструкторах** (`PiperVoice.load` в `tts.py`, `vosk.Model()` в `stt.py`) —
  **корневая причина** 69 %/76 %: `_synth`/`transcribe` нельзя инстанцировать в CI. Лечится R-TTS-1/R-STT-1.
- **I/O в конструкторе `esphome_client`:** `APIClient`/`ReconnectLogic` строятся в `__init__`, поэтому
  все тесты используют `__new__`-bypass. Поведенческие тесты это обходят; для теста самой проводки — R-ESP-1 (опц.).
- **Hard-sleep в существующих тестах:** `test_reminders.py:154` (`sleep(0.1)`), `:177` (`sleep(0.35)`) —
  риск флака под нагрузкой CI. Эталон рядом — `:128` (`wait_for(event, timeout=2.0)`). **Новые scheduler-тесты
  (§2 builtin) делать только на `asyncio.Event`/bounded-poll, не копировать sleep.**
- **Over-mocked StubSession** (`test_mcp_client.py`): мокаются и `ClientSession`, и транспорт → реальный
  декодер `mcp` не проверяется (поэтому нужен contract-тест R-MCP-1).
- **Frontend:** gating-логика `matchesFilters`/`matchPreset` спрятана в page-компонентах (не экспортирована) —
  вынести в чистый модуль (R-FE-1); локаль-зависимые форматтеры без инъекции TZ (R-Tool-2); `isSecret`
  регексп `/key/i` over-broad (маскирует `keyboard_layout`) — уточнить намерение.
- **`app.py` 0 %** — длинная `main()` (DI + lifecycle); приемлемо как проводка, кроме 2 извлекаемых
  boot-warning (R-APP-1). `version.py` 0 % — git-subprocess-обвязка, кроме чистого `_resolve_version`.
- **microWakeWord/ — вне периметра:** offline ML-скрипты тренировки, недетерминированные, не часть деплоя.
- **Order-dependent / нестабильных (>3 % фейлов) тестов не обнаружено.** Суиты в целом дисциплинированы
  (реальные WAV-round-trip вместо снапшотов, spy на «не вызван», bounded-poll вместо фиксированных пауз).

## 5. Необходимые рефакторинги перед написанием тестов
| ID | Рефакторинг | Блокирует тесты |
|----|-------------|-----------------|
| R-TTS-1 | Инъекция Piper-voice (опц. ctor-аргумент/фабрика) вместо `PiperVoice.load` в `__init__` ([tts.py](src/tts.py)) | 3 unit: `_synth` сшивка/skip/fallback |
| R-STT-1 | Инъекция Vosk-`Model` вместо `Model()` в `__init__` ([stt.py](src/stt.py)) | 2 unit: `transcribe`/`_decode` |
| R-APP-1 | Извлечь синхронные `warn_legacy_mcp(doc)`/`validate_boot_config(core)` из `app.main` | 2 unit: boot-warnings |
| R-FE-1 | Экспорт/вынос `matchesFilters` ([log.jsx](frontend/react-export/src/pages/log.jsx)) и `matchPreset`/`VAD_PRESETS` ([stages.jsx](frontend/react-export/src/pages/stages.jsx)) в чистый модуль | 2 unit |
| P-FE-1 | Поставить Vitest (+ `@testing-library/react`/`jsdom` для 2 компонент-тестов) | **все** frontend-тесты |
| R-Tool-2 | Зафиксировать TZ раннера (`TZ=UTC`) для time-форматтеров | `fmtStarted`, `mapRun.time` |
| R-MCP-1 | Записать реальный Node-RED JSON-RPC фикстур (initialize/list_tools/call_tool, привязать версию SDK) | 1 contract |
| R-ESP-1 *(опц., низкий)* | Инъекция `APIClient`/`ReconnectLogic` в `DeviceClient` | 1 unit: проводка ctor (поведенческие уже работают через `__new__`) |

## 6. План внедрения (по фазам, ROI-обоснование)
- **Фаза 1 — error/isolation-ветки без рефакторинга (наивысший ROI, дёшево):**
  `pipeline._run` catch-all + изоляция insert/put_audio/broadcast/raw-capture (5-6), `disarm_capture`,
  capture-maxlen late-chunk, `context` OSError; `tool_hub`/`mcp_client` error+graceful (8); `panel_api`
  6 мелких (ValueError-422, non-dict body, limit-clamp, bad-id, SPA-guard, version-precedence); `config-core`
  (3 unit); `builtin-mcp` calendar/`_format_due` (4 unit) + 2 scheduler integration (на `asyncio.Event`);
  `runs_store` percentile-None. Закрывает самые опасные тихие падения.
- **Фаза 2 — reconnect device-io + рефакторинг I/O-конструкторов:**
  `esphome_client` reconnect-машина (`_on_connect`/device_info-сбой/`_on_disconnect`/`announce`/`statuses`, 6 unit
  через `__new__`); R-TTS-1/R-STT-1 → `tts.py`/`stt.py` (поднять 69 %/76 %, +`_decode_v3` без рефактора);
  R-APP-1 → 2 boot-warning + 2 `load_or_create_config` (без рефактора).
- **Фаза 3 — frontend:**
  P-FE-1 (Vitest) + R-Tool-2 (TZ) → чистая логика `schema.js`/`runsModel.js`/`format.js`/`api.js`/
  `useStageForm.js`/`primitives`; R-FE-1 → фильтры/пресеты; затем 2 компонент-теста `SchemaForm`
  (защита от утечки секретов в UI / float-в-int).
- **Фаза 4 — contract + опциональное:**
  R-MCP-1 → MCP wire-contract; опц. property-based (`_deep_merge`/`split_sentences`); вынести hard-sleep
  в `test_reminders.py` на `asyncio.Event`; опц. R-ESP-1 + тест проводки ctor.

## 7. Источники
- **Отчёты 9 аналитиков** `module-testability-analyst`: config-core, reconfig-runtime, stage-plugins,
  pipeline-orchestration, tool-hub-mcp, builtin-mcp, device-io, panel-api, frontend.
- **Вывод coverage-инструмента** — `.venv/bin/python -m pytest --cov=src --cov-report=term-missing`
  (TOTAL **86 %**, 2896 stmts / 419 miss, 421 тест прошёл). Все процитированные аналитиками непокрытые
  строки сверены построчно с реальным прогоном; расхождений нет.
- **Сверка фактов оркестратором:** `tts_url` — единый хелпер ([audio_server.py:14](src/audio_server.py#L14)),
  импортируется `pipeline.py`+`esphome_client.py` (дубля нет → R-DEV-1 закрыт); `main.py` — тривиальный
  root-entrypoint (не `src/main.py`); `app.py`/`version.py` остаются 0 % осознанно.
- **Фильтрация кандидатов:**
  - Шаг 1 (cross-module дедуп): `tts_url` (pipeline ↔ device-io) — один хелпер, тест общий; `announce`-маршрутизация
    учтена как «уже покрыто» в `test_reminders.py` (не дублируется в device-io); `append_context`-сбой —
    владелец context.py (pipeline-обёртка не дублируется).
  - Шаг 2 (skip-list): ~55 кандидатов отклонены (third-party, DI/проводка, тривиальные DTO/геттеры,
    тавтологичные снапшоты, defensive/log-only, уже покрытое на нижнем слое).
  - Шаг 3 (бюджет пирамиды): respx/fakes/monkeypatch → unit; реальный сокет/aiohttp TestServer/jsdom →
    integration. Итог unit ~88 % / integration ~11 % / E2E 0 — в рамках.
  - Шаг 4 (refactor-blocking): 7 тестов привязаны к R-TTS-1/R-STT-1/R-APP-1/R-FE-1/P-FE-1/R-MCP-1; без
    рефактора не обещаны.
  - Шаг 5 (ROI): фазы §6 — сперва бесплатные error-ветки, затем reconnect + рефакторинг наименее покрытого I/O,
    затем frontend, затем contract.
  - Шаг 6 (adversarial): каждый кандидат проверен на «прошёл бы при реалистичной тихой поломке»; слабые
    формулировки (статус-200 без проверки clamp-значения; `is not None` для `provider()`; log-spy-only)
    переписаны на observable-state или отброшены.
- **Открытые вопросы (требуют решения владельца проекта):**
  1. **Contract MCP (R-MCP-1):** нужен реальный записанный ответ Node-RED; без живого фикстура тест лишь
     повторит наши предположения — НЕ писать без него.
  2. **`ReminderScheduler._loop` при `deliver is None`** снимает one-shot **до** проверки → тихая потеря в
     окне инициализации (только warning). Зафиксировать как поведение (тест-спека) или чинить как баг?
  3. **`panel_api._patch_config` ValueError-ветка:** даёт ли `ConfigService.apply()` «голый» `ValueError`
     (≠ `ValidationError`) на реальном patch? Если нет — тест через стаб `svc.apply` на unit-уровне.
  4. **`config_service` allOf/anyOf:** эмитит ли live-схема CoreConfig allOf/anyOf для секций с default?
     От этого зависит, реальна ли ветка [config_service.py:55-61](src/config_service.py#L55) или это мёртвый код.
  5. **frontend `isSecret` `/key/i`** маскирует `keyboard_layout`/`hotkey` — намеренно или баг?
  6. **`esphome_client.announce`** хардкодит язык `"ru"` — мультиязычные reminders в требованиях?
