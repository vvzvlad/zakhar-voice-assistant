# Отчёт по тест-стратегии — zakhar-voice-assistant — 2026-06-08

> Голосовой ассистент: ESP32-колонка (ESPHome Native API) → STT → LLM (agentic
> tool-calling) → smart-home через MCP → TTS. Backend на Python 3.11 (pytest,
> asyncio), панель управления на React/Vite. Тренировочный пайплайн в
> `microWakeWord/` (offline ML-скрипты, недетерминированные, запускаются вручную)
> **выведен из периметра тест-стратегии рантайма** — см. §4.

## 1. Исполнительное резюме
- **Проанализировано модулей:** 8 (по одному `module-testability-analyst` на модуль).
- **Предложено тестов (unit / integration / E2E / contract):** **58 / 9 / 0 / 1** (итого 68).
- **Отклонено как малоценные** (skip-list / уже покрыто / тривиально): ~60 кандидатов.
- **Покрытие сейчас:** backend **83 %** (проверено `pytest --cov`, цифры аналитиков
  совпали построчно); frontend **0 %** (тест-раннера нет).
- **Прогноз после внедрения:** backend **~90 %** (`app.py`-проводка и `version.py`
  сознательно остаются непокрытыми), frontend — покрыта вся чистая логика
  (`schema.js`, `runsModel.js`, `format.js`, `api.js`, `useStageForm.js`).
- **Классификация слоёв:** respx/fakes/monkeypatch в одном процессе = unit; реальный
  сокет (aiohttp TCPSite), многофайловый flow на диске, jsdom-рендер = integration.
- **Бюджет пирамиды соблюдён:** unit 85 % (≥70 %), integration 13 % (≤20 %), E2E 0 (≤10).

## 2. Рекомендации по модулям

### config-core — `config_store.py`, `config_service.py`, `core_config.py` (94–100 %)
Модуль уже почти полностью покрыт; гэпы — это непроверенные *поведения*, не строки.
- **Кандидаты в чистые unit:** `config_service._deep_merge` ([config_service.py:37](src/config_service.py#L37)) — рекурсивный merge, list-vs-dict семантика нигде не утверждается напрямую.
- **Unit-тесты добавить:**
  - `_deep_merge`/`apply`: patch со списком `core.mcp_servers`/`core.devices` **заменяет** список целиком (не сливает поэлементно). Ловит: тихую порчу данных, если merge «улучшат» до рекурсии в списки — текущий тест на скаляр это не поймает.
  - `ConfigDoc` defaulting/миграция: загрузка дока без секции `core`/`core.runs`/`version` → применяются дефолты. Ловит: падение на старом config.json после эволюции схемы.
  - `apply` round-trip через диск: apply → `config_store.load` → новый `ConfigService` → значения совпадают (float `speed=1.2`, unicode «Москва»). Ловит: ассиметрию `model_dump(json)` ↔ `ConfigDoc(**doc)`.
  - `config_store.load` на пустом файле `""` → `JSONDecodeError` (не тихий `{}`). Ловит: «битый конфиг тихо сбрасывается в дефолт».
- **Integration-тесты добавить:** `app.load_or_create_config` — первый boot из `templates/default_config.json`; существующий config **не** перетирается. Ловит: сломанный seed / клоббер пользовательского конфига. *(владелец теста — этот модуль; см. дедуп в §7.)*
- **НЕ тестировать:** `config_store.save` стр. 67-68 (`except OSError` на fsync директории — недетерминируемо, нет наблюдаемого поведения); DTO-модели `core_config` без валидаторов; атомарную запись посимвольно (контракт уже покрыт `test_failed_save_*`).

### stage-plugins — `src/plugins/**`, `llm.py`, `stt.py`, `tts.py`
Главный гэп — **`tts.py` (63 %)** и **`stt.py` (76 %)**.
- **Unit-тесты добавить:**
  - `llm.call_llm_api` — 4 ветки ошибок: невалидный JSON в `tool_calls.arguments` → `{}` (не падение); пустой `choices` → произносимая строка-ошибка, не `IndexError`; non-2xx с неразбираемым телом (`reason=None`); раунды исчерпаны, но `last_content` есть → вернуть его, не «слишком много вызовов».
  - `tts.PiperTtsBackend._synth` (блок ~105-137): сшивка предложений + паузы (`sentence_silence`); fallback «нечего произносить» → валидный тихий клип, не краш. **Требует R-TTS-1.** Ловит: рассинхрон фреймов (искажённый звук), краш на пунктуации.
  - `tts.PiperTtsBackend._synth`: сбой синтеза одного фрагмента **пропускается** (`except: continue`), хороший фрагмент остаётся. **Требует R-TTS-1.**
  - `tts.TeraTtsHttpBackend.synthesize` (0 % — это TTS-провайдер по умолчанию): 200 → (mime, body); нет `Content-Type` → дефолт `audio/mpeg`; non-2xx → проброс; `quote(text, safe='')` кодирует спецсимволы. respx-mock → классифицируется как unit.
  - `stt.VoskSttBackend.transcribe`: пустой PCM → `""` без обращения к модели; непустой → `.text.strip()`. **Требует R-STT-1.**
- **НЕ тестировать:** REGISTRY/`@register` (полностью покрыто `test_registry`); все `create()` (passthrough-конструкторы); `ConfigModel`-схемы (DTO, покрыто `test_providers`); внутренности vosk/piper/lameenc (third-party); `wav_to_mp3`/`pcm_to_wav`/`split_sentences`/`yandex_stress_markup` (уже покрыты).

### pipeline-orchestration — `pipeline.py`, `app.py`, `context.py`, `run_events.py`, `runs_store.py`, `prompt.py`, `text.py`, `main.py`
`pipeline.py` 92 %, `context.py` 80 %, `app.py` **0 %**. Непокрытое — это ветки ошибок.
- **Unit-тесты добавить (in-process, fakes уже есть в `test_pipeline.py`):**
  - `Pipeline.on_audio`: поздний чанк после finalize → ранний выход ([pipeline.py:143](src/pipeline.py#L143)); переполнение `HARD_CAP_BYTES` → обрезка + `reason="maxlen"`, минуя VAD (**требует R-PIPE-1**).
  - `Pipeline._run` изоляция сбоев (5 тестов, 368-397/319-321): исключение из STT/LLM → эмит `VOICE_ASSISTANT_ERROR`+`RUN_END`, `result="error"`; сбой `append_context` / `runs_store.insert` / `broadcast` не ломает прогон; корректная классификация `result="tool"` vs `"ok"`. **Высокая ценность — телеметрия ошибок.**
  - `context`: TOCTOU `getmtime` после `exists()` → stale, не краш (26-27, monkeypatch — низкий приоритет); пропуск пустых/не-dict JSONL-строк (48, 54); `OSError` при чтении → `[]` (59-61); `OSError` при записи `append_context` проглатывается (92-93).
  - `runs_store._percentile` напрямую: pct 0/50/100, один элемент, пустой → None. Ловит: off-by-one в `round((pct/100)*(N-1))`.
- **НЕ тестировать:** `version.py` (тривиальная константа); `main.py` / `app.main` (DI-проводка — см. R-APP-1, без рефактора осмысленного теста нет); `run_context.current_device` (ContextVar); `RunEventsHub.register/unregister/count` (тривиально, `broadcast` покрыт 100 %); `_emit`/`prompt.*` (passthrough / покрыто 100 %).

### tool-hub-mcp — `tool_hub.py` (84 %), `mcp_client.py` (85 %)
- **Unit-тесты добавить:**
  - `BuiltinMcpSource._normalize` (131, 134-135): формы SDK — tuple `([content],{result})`, dict с/без `result`, content-sequence, не-iterable → `str(res)`. Ловит: неверную ветку → модель получает Python-repr вместо текста.
  - `BuiltinMcpSource.call` (113-116): исключение инструмента → строка `"error calling …"`, не краш agentic-loop.
  - `ToolHub.ensure_tools` (193-194) и `ToolHub.stop` (238, 241-242): сбой одного источника не валит остальные.
  - `McpToolHub.start` (62-64) и `ensure_tools` (72-76): недоступный smart-home MCP → `tools=[]`, без проброса (ассистент не падает при старте).
- **Contract-тест добавить (единственный в проекте):** `test_mcp_smarthome_response_schema_contract` — закрепить записанный JSON-RPC ответ Node-RED-сервера (`list_tools` + `call_tool`) и прогнать через **реальный** декодер `mcp.ClientSession` (respx на `MCP_SMARTHOME_URL`), проверив `name/description/inputSchema` и извлечение `.text`. Ловит: дрейф wire-формата → тихо пустой список инструментов. *Текущие StubSession-тесты этого НЕ ловят.* **Зависимость: нужен реальный записанный фикстур (см. §7 открытые вопросы).**
- **НЕ тестировать:** абстрактные `NotImplementedError` базы; `HttpMcpSource.*` (тонкие делегаты к `McpToolHub`); `_to_groq_tool`/`_resolve_transport` (покрыто); `mcp`-библиотека.

### builtin-mcp — `builtin_mcp/calendar.py` (89 %), `builtin_mcp/reminders.py` (88 %), `reminders.py` (93 %), `openweathermap.py` (91 %)
Модуль очень хорошо покрыт (суиты 944 + 489 строк). Только реальные остаточные гэпы:
- **Unit-тесты добавить:**
  - `calendar.CalendarClient._to_yandex_utc` (наивный datetime → trailing `Z`, локаль→UTC). Ловит: неверное wall-clock время в Яндекс-календаре для naive ISO от модели.
  - `calendar._event_dict` (262-265): нечисловой `PRIORITY` → поле опускается, не краш чтения.
  - `calendar._format_rrule` (124-127) и `_parse_until` (53-54): malformed RRULE-часть без `=` / пустой `UNTIL` → внятный `ValueError`.
  - `reminders._format_due` (23-25): due в прошлом → без суффикса «(через N мин)», не «через -3 мин».
- **Integration-тесты добавить (in-process):**
  - `ReminderScheduler._loop`: due при `deliver is None` (184-188) → reminder снят (one-shot) + warning, цикл жив. *Документирует латентную потерю в окне инициализации — см. §7.*
  - `ReminderScheduler._loop`: `deliver` бросает (189-192) → поймано, цикл переживает и стреляет дальше. Использовать `asyncio.Event` (не hard-sleep).
- **НЕ тестировать:** `caldav`/`icalendar` (third-party — тестируем делегирование, не escaping); остальные tool-wrapper `except → "Не удалось…"` (поведение доказано на одном инструменте `test_tool_error_returns_string_not_crash`); `build_openweathermap_server` (passthrough, 100 %); тривиальные `_iso`/`_component_of`/`close`.

### device-io — `esphome_client.py` (**39 %, без своего теста**), `audio_server.py` (72 %)
`esphome_client.py` — наименее покрытый файл с реальной логикой.
- **Unit-тесты добавить:**
  - `DeviceManager.announce` (**требует R-DEV-2**): named online → маршрутизация; `None` → первый online; named offline / не найден → drop без вызова; нет online → drop + warning, не исключение. Ловит: reminder уехал не на ту колонку / упал вместо drop.
  - `DeviceManager.statuses`: name/host/online на клиента; пустой список → `[]`. Ловит: неверный online-статус в панели.
  - Общий хелпер `tts_url(base_url, audio_id, mime)` (**требует R-DEV-1**): `audio/wav→.wav`, `audio/mpeg→.mp3`, `audio/flac→.flac`, неизвестный → `.mp3`; trailing slash срезается один раз. **Дедуп: один тест на извлечённый хелпер покрывает оба call-site** (`esphome_client` + `pipeline`).
  - `AudioServer.put` prune (29-31): протухший entry удаляется на следующем `put`; свежий выживает; текущий вставляемый не вычищается. Ловит: рост памяти / преждевременное вытеснение.
  - `AudioServer._handle_tts` граница TTL (44, строгий `expiry > now`): entry «есть, но протух» → 404; имя без расширения / с несколькими точками. Ловит: off-by-one на границе TTL, отдачу протухшего аудио.
  - `DeviceClient._on_connect/_on_disconnect` (**требует R-DEV-4**): online-флаг; ре-байнд `send_event/send_audio`; ре-подписка через `_unsub`; `device_info()` бросает → warning, но connect идёт. Ловит: «события колонки тихо теряются после reconnect».
- **Integration-тест добавить:** `AudioServer` поверх **реального** aiohttp TCPSite: `GET /tts/<id>.mp3` 200 + body + Content-Type; `/health`; `GET /tts/<unknown>` 404; `stop()` идемпотентен. Ловит: сломанную регистрацию роутов (текущие FakeRequest-тесты её не трогают).
- **НЕ тестировать:** `__init__` обоих классов (DI/проводка `APIClient`/`ReconnectLogic`); `start/stop` (делегаты к third-party lifecycle); `_handle_start/_handle_audio/_handle_stop` (passthrough к pipeline, покрыто ниже); `aioesphomeapi`/`aiohttp` внутренности.

### panel-api — `panel_api.py` (86 %), `logging_setup.py` (91 %), `version.py` (0 %)
Главная внешняя write-поверхность (мутация конфига) — важна валидация.
- **Integration-тесты добавить (реальный aiohttp TestServer):**
  - `_patch_config` ветка `ValueError` (108-109): patch с неизвестным провайдером (`{"tts":{"selected":"нет-такого"}}`) → 422, `pending_restart is False`, документ не изменён. Ловит: «невалидный конфиг принят и сохранён → ломает рантайм». Отдельно от уже покрытой `ValidationError`-ветки.
  - `_get_run` невалидный id (188-189): `GET /api/runs/abc` → 400 `{"error":"invalid id"}`. Ловит: `int()` на пользовательском вводе → 500.
  - `_get_runs` плохой `limit` (174-175): `?limit=abc` → fallback 100; большой → clamp 500. Ловит: 500 на параметре пагинации / выгрузку всей таблицы.
  - `_spa_index` + статика (228, 254-258): `index.html` в tmp + `assets/`; `GET /` и deep-link → index.html; `/api/...` **не** перехвачен catch-all (`(?!api/)`). Ловит: API затенён SPA-fallback / deep-link 404.
- **Unit-тест добавить:** `logging_setup.InterceptHandler.emit` (30-31): уровень без loguru-имени → fallback на `record.levelno`. Ловит: пропажу лог-строк third-party либ с нестандартным уровнем.
- **НЕ тестировать:** `version.py`; `start/stop` (aiohttp lifecycle-проводка); `_cors_middleware`/`_add_cors` (покрыто); `_runs_stream` тело `async for … pass` (no-op, ассерт тавтологичен); таблицу роутов в `build_app` (декларативная проводка).

### frontend — `frontend/react-export/src/**` (React/Vite, **0 тестов**)
**Prerequisite P-FE-1:** поставить Vitest (переиспользует Vite-конфиг, `import.meta.env`), добавить `test`-скрипт. Чистая логика идёт в node-env без DOM; для 3 компонент/hook-тестов — `@testing-library/react` + `jsdom`.
- **Unit-тесты добавить (наивысший ROI, без DOM):**
  - `schema.js`: `resolve` (`$ref`→`$defs`; Optional `anyOf:[…,{type:null}]` → не-null ветка; sibling `default` сохраняется; неизвестный ref → `{}`); `enumOf` (top-level / вложенный в `anyOf` / нет → null); `deref` (циклический ref останавливается на 10-й итерации); `isSecret`+`humanize`. Ловит: поле рендерится **не тем виджетом** (api_key как открытый текст — утечка в UI; enum как текст-инпут).
  - `runsModel.js`: `mapRun` (null-тайминги→0; `audio`/`error` суб-объекты; `null`-строка→null); `fmtSec` (null/NaN→«—»).
  - `format.js`: `fmtUptime` (<1ч / часы+мин / дни+pad / отрицательное→0 / null) ; `fmtStarted` (валид/невалид→raw/falsy→«—»).
  - `useStageForm.errorLines`: pydantic `detail[]`→строки, `loc`-префикс `"body"` срезан, fallback на `err.message`.
  - `api.js`: `request` (2xx JSON; 204→null; non-2xx `{detail}`/`{error}`/не-JSON→`ApiError` со `status`; `fetch` reject→`status:0`); `getRuns` query-builder (пустые/null параметры выкинуты, значения URL-кодированы). Ловит: проглоченный non-2xx (валидация не доходит до формы).
  - `log.matchesFilters` (**требует R-FE-1**: экспорт) и `stages.matchPreset` (**требует R-FE-1**): фильтры логов и подсветка VAD-пресета.
  - `primitives.segsFor`+`total`: waterfall-сегменты, спец-случаи `empty`/`error`, порог метки 15 %.
- **Component/integration-тесты (jsdom, минимум):**
  - `SchemaForm.SchemaField` — таблица выбора виджета: bool→Toggle; enum ≤3→Seg / >3→Select; number+min/max+`widget:slider`→Slider иначе Stepper; integer округляется; secret→masked KeyInput; `options:dynamic`→fetched Select. Ловит: ключевой silent-break — API-ключ открытым текстом / temperature 0–2 на step-1. Ассертить идентичность виджета и payload `onChange`, **не** разметку.
  - `SchemaForm.schemaNeedsRestart` (`apply:rebuild`→true).
  - `useStageForm` (renderHook): `onChange`→`dirty`; ре-сид при смене `initialValues`; `save`→`patch(buildPatch)`, ошибка 422 в `err`.
- **НЕ тестировать:** `nav.js`/`stageMeta.js`/`navStore.js` (статические таблицы/геттеры); `main.jsx`/`App.jsx`/`appData.jsx` (проводка/контекст над уже покрытым `api.*`); `icons.jsx`; `Sidebar`/`Topbar` (презентационные — тавтологичные снапшоты); презентационные примитивы и страницы-оболочки (логика вынесена в unit выше); снапшот-тесты.

## 3. Сквозные аспекты
- **Contract-тесты:** ровно один — wire-формат внешнего Node-RED MCP (§2 tool-hub-mcp). Frontend↔backend JSON-контракт закрепляется на стороне `panel_api.py`; `mapRun`/`matchesFilters` — его consumer-зеркало, отдельный contract не нужен.
- **Property-based:** кандидаты после извлечения чистых функций — `config_service._deep_merge` (идемпотентность merge с пустым patch; ассоциативность вложенных dict) и `tts.split_sentences` (конкатенация фрагментов = исходный текст без разделителей). Низкий приоритет, опционально.
- **Дымовые/нагрузочные:** не применимо (LAN-сервис на N колонок, не публичный web).
- **Test-data factories:** `make_config_doc(**overrides)` (вместо полного `_doc()` в каждом тесте config-service); `make_run_row(**overrides)` (backend `runs_store` ↔ frontend `mapRun`); записанные MCP-фикстуры (`list_tools`/`call_tool`) для contract-теста.

## 4. Обнаруженные антипаттерны
- **`esphome_client.py` — god-ish класс с I/O в конструкторе:** `APIClient` создаётся в `__init__`, чистая логика (выбор цели, mime→url, connect/disconnect) сварена с живым сокетом → 39 % и ноль своих тестов. Лечится R-DEV-1/3/4.
- **Дублирование mime→extension:** идентичная карта в [pipeline.py:340-343](src/pipeline.py#L340-L343) и [esphome_client.py:103](src/esphome_client.py#L103) — дрейф-хазард. Лечится R-DEV-1 (общий хелпер).
- **`app.main` — длинная процедура, мешающая чистые решения и I/O:** ветки выбора MCP-источников (`core.mcp_servers`/openweathermap/calendar/reminders) спрятаны в bootstrap → 0 % и нетестируемы без R-APP-1.
- **Скрытая загрузка тяжёлых моделей в конструкторах:** `PiperVoice.load` / `Model()` в `__init__` Piper/Vosk → нельзя инстанцировать в CI без модели (R-TTS-1/R-STT-1).
- **Hard-sleep в тестах (существующих):** `test_reminders.py:145,168` и `test_audio_server.py:41-47` зависят от wall-clock (`asyncio.sleep(0.35/0.01)`) → потенциальный флак на нагруженном CI. Новые scheduler-тесты делать на `asyncio.Event`, не копировать sleep.
- **Over-mocked StubSession (`test_mcp_client.py`):** мокает и `ClientSession`, и оба транспорта → реальный декодер `mcp` не проверяется (поэтому и нужен contract-тест).
- **microWakeWord/ — вне периметра:** offline ML-скрипты тренировки (`build_features*.py`, `evaluate*.py`, `gen_*`), недетерминированные, запускаются вручную, не часть деплоя. В тест-стратегию рантайма **не включены** осознанно.
- **Order-dependent / нестабильных (>3 % фейлов) тестов не обнаружено.**

## 5. Необходимые рефакторинги перед написанием тестов
| ID | Рефакторинг | Блокирует тесты |
|----|-------------|-----------------|
| R-TTS-1 | Инъекция Piper-voice (опц. ctor-аргумент/фабрика) вместо `PiperVoice.load` в `__init__` ([tts.py](src/tts.py)) | 2 unit: Piper `_synth` сшивка + skip-фрагмента |
| R-STT-1 | Инъекция Vosk-`Model` вместо `Model()` в `__init__` ([stt.py](src/stt.py)) | 1 unit: `VoskSttBackend.transcribe` |
| R-PIPE-1 | `HARD_CAP_BYTES` → атрибут инстанса (как vad_* пороги) ([pipeline.py:40](src/pipeline.py#L40)) | 1 unit: обрезка по HARD_CAP |
| R-DEV-1 | Извлечь общий `tts_url(base_url, audio_id, mime)` (убирает дубль pipeline↔esphome) | 1 unit (покрывает оба call-site) |
| R-DEV-2 | Извлечь чистый `_select_target(device_name)` из `DeviceManager.announce` | 1 unit: маршрутизация announce |
| R-DEV-4 | Инъекция `APIClient` в `DeviceClient` (фабрика/сеттер) | 1 integration: connect/disconnect state-machine |
| R-FE-1 | Экспортировать `matchesFilters` ([log.jsx](frontend/react-export/src/pages/log.jsx)) и `matchPreset`/`VAD_PRESETS` ([stages.jsx](frontend/react-export/src/pages/stages.jsx)) | 2 unit |
| P-FE-1 | Поставить Vitest (+ `@testing-library/react`/`jsdom` для 3 компонент-тестов) | **все** frontend-тесты |
| R-APP-1 | Извлечь чистый `build_tool_sources(core, clients)` из `app.main` | (тест не обещан) делает `app.py` осмысленно тестируемым |
| R-DEV-3 *(опц.)* | Инъекция clock в `AudioServer` | уточняет границу TTL (не блокирует) |

## 6. План внедрения (по фазам, ROI-обоснование)
- **Фаза 1 — гэпы ошибок без рефакторинга (наивысший ROI, дёшево):**
  `pipeline._run` изоляция сбоев + телеметрия ошибок (5), `tool_hub`/`mcp_client` error-ветки (6), `panel_api` валидация write-поверхности (4 integration + logging unit), `context` error-ветки, `config-core` (4 unit + bootstrap), `builtin-mcp` остаточные (5+2), `audio_server` TTL/prune + TCPSite. Ноль рефакторингов, закрывает самые опасные тихие падения.
- **Фаза 2 — рефакторинг I/O-конструкторов + их тесты:**
  R-TTS-1/R-STT-1 → `tts.py`/`stt.py` (поднять 63 %/76 %); R-DEV-1/2/4 → `esphome_client.py` (39 %→~80 %, устранить дубль); R-PIPE-1 → HARD_CAP. ROI: закрывает наименее покрытый реальной логикой код.
- **Фаза 3 — frontend:**
  P-FE-1 (Vitest) → чистая логика `schema.js`/`runsModel.js`/`format.js`/`api.js`/`useStageForm.js`; R-FE-1 → фильтры; затем 3 компонент/hook-теста (`SchemaForm` выбор виджета — защита от утечки секретов в UI).
- **Фаза 4 — contract + опциональное:**
  MCP wire-contract (после получения записанного фикстура Node-RED); R-APP-1 + тест `build_tool_sources`; опц. property-based на `_deep_merge`/`split_sentences`; вынести hard-sleep на `asyncio.Event`.

## 7. Источники
- **Отчёты 8 аналитиков** `module-testability-analyst` (по модулю на каждый): config-core, stage-plugins, pipeline-orchestration, tool-hub-mcp, builtin-mcp, device-io, panel-api, frontend.
- **Вывод coverage-инструмента** — `.venv/bin/python -m pytest --cov=src --cov-report=term-missing` (TOTAL 83 %, 2182 stmts / 377 miss). Все процитированные аналитиками непокрытые строки сверены построчно с реальным прогоном; расхождений нет (у `tts.py` сдвиг номеров ~2 строки, те же блоки `TeraTts.synthesize`+Piper `_synth`).
- **Фильтрация кандидатов:**
  - Шаг 1 (cross-module дедуп): mime→url-тест (pipeline ↔ device-io) → один тест на общий хелпер; `load_or_create_config` (config-core ↔ pipeline) → владелец config-core; `max_tool_rounds` exhaustion (pipeline → llm) → владелец stage-plugins. **−3.**
  - Шаг 2 (skip-list): ~60 кандидатов отклонены (third-party, DI/проводка, тривиальные DTO/геттеры, тавтологичные снапшоты, уже покрытое на нижнем слое).
  - Шаг 3 (бюджет пирамиды): respx/fakes/monkeypatch → unit; реальный сокет/многофайловый flow/jsdom → integration. Итог unit 85 % / integration 13 % / E2E 0 — в рамках.
  - Шаг 4 (refactor-blocking): 7 тестов привязаны к рефакторингам (§5); без рефактора не обещаны.
  - Шаг 5 (ROI): фазы §6 — сперва бесплатные error-ветки, затем рефакторинг наименее покрытого I/O, затем frontend, затем contract.
  - Шаг 6 (adversarial): каждый аналитик подтвердил, что тест падает при реалистичной тихой поломке; кандидаты, проходящие при сломанной реализации, отброшены.
- **Открытые вопросы (требуют решения владельца проекта):**
  1. **Contract-тест MCP** нужен реальный записанный ответ Node-RED (`node-red-contrib-mcp-server`): `list_tools` + `call_tool` JSON-RPC. Без живого фикстура тест лишь повторит наши же предположения.
  2. **`ReminderScheduler._loop`**: due-reminder при `deliver is None` снимается (one-shot) **до** проверки → тихо теряется в окне инициализации (только warning). Зафиксировать как поведение или чинить как баг?
  3. **`DeviceManager.announce(device_name=None)`** fallback на «первый online» помечен «не должно случаться через pipeline» — живой путь (нужен тест) или мёртвый код (skip)?
  4. **`esphome_client.announce`** хардкодит язык `"ru"` — мультиязычные reminders в требованиях? Если да — это непротестированное скрытое ограничение.
