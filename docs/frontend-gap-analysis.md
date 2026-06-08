# Ревизия дизайнерского веб-экспорта vs наша система конфига

Разбор `frontend/react-export` (Vite + React, дизайнерский макет панели) и сопоставление с тем,
что реально есть в бэкенде после рефактора на `data/config.json` (см.
[settings-storage-design.md](settings-storage-design.md)). Что норм, что дизайнер выдумал, что забыл,
и какие метрики придётся дописывать.

---

## TL;DR — вердикт

Экспорт **сильный и близкий к нашему домену**: все стадии, структура «провайдер на стадию +
самоописываемые настройки», лог с waterfall таймингов, MCP с кастомным промптом — всё совпадает с
тем, что мы спроектировали. Это рабочая основа, не выкидывать.

Но есть четыре класса расхождений, которые надо закрыть прежде, чем «прикручивать к API»:

1. **Именование.** Дизайнер смоделировал данные на **старых `.env`-именах** (`INTENT_API_KEY`,
   `YANDEX_TTS_*`, `VAD_*`), которых у нас больше нет — у нас вложенный `config.json` + `catalog()`
   с JSON Schema. Формы надо строить из нашего `catalog()`, а не из хардкоженных env-имён.
Мигрировать на наши имена. главное ты понял что ты имелл ввиду.


2. **Выдуманные настройки** — несколько полей сделаны редактируемыми, а в коде они **захардкожены**
   (LLM-фолбэки, язык/температура/таймаут STT, таймаут LLM).
если их можно настроить, то давай сделаем их настраиваемыми.

3. **Забытая настройка** — **погоды нет в UI вообще** (`core.weather.api_key/city`), хотя она
   подмешивается в системный промпт.
погода потом сделается mcp сервером, это норм

4. **Метрик нет в бэкенде** — весь раздел Operations (KPI, p50/p95, лог прогонов, тайминги стадий,
   токены по раундам, статусы устройств, аптайм) дизайнер нарисовал, но у нас **ничего из этого не
   персистится** — пайплайн только пишет в loguru. Это самый большой кусок работы.
метрики надо делать.

Плюс две вещи, которые в UI есть, а в бэкенде их нет как фичи: **RUAccent** (целая стадия) и
**мульти-MCP** (у нас один сервер, без `name/enabled/prompt`).
это тоже надо сделать, но после того как обавим фронт.
---

## 1. Что это и как устроено

- Vite + React, без CSS-фреймворка. Точка входа `App.jsx` → sidebar + topbar + роутер по секциям.
- **`src/data.js` (`ZAPP`)** — мок всей модели данных; README прямо говорит «swap for your API».
  Это и есть карта того, что дизайнер ожидает от бэкенда.
- Страницы: `dashboard` (KPI + service-map + последние прогоны), `stages` (VAD/STT/LLM/RUAccent/TTS),
  `log` (журнал + drawer с waterfall/раундами/аудио), `integrations` (MCP/Prompt/Context/Devices),
  `network` (proxy/audio + System).
- Важно: атрибут `param="INTENT_MODEL"` в формах — это **только подпись для разработчика**, в UI не
  рендерится (README, п. «Notes»). Значит жёсткой завязки на env-имена в разметке нет — но в `data.js`
  модель построена вокруг них.

---

## 2. Что норм (совпадает с нашей системой)

| Блок UI | Наш бэкенд | Статус |
|---|---|---|
| Service-map стадий, клик → конфиг стадии | фиксированный конвейер STT→LLM→(accent)→TTS | ✅ концепция совпала |
| VAD: 5 полей (aggressiveness, silence/min_speech/max_utterance/no_speech) | `core.vad.*` — те же 5 | ✅ 1:1 |
| STT: выбор `groq`/`vosk` + поля | провайдеры `stt`, `GroqSttConfig`/`VoskSttConfig` | ✅ (см. выдуманные поля groq) |
| LLM: выбор провайдера, api_key, model, temperature, max_tokens, max_tool_rounds | `LlmConfig` | ✅ совпали |
| TTS: `teratts`/`piper`/`yandex`, taймаут | провайдеры `tts` + `core.tts_timeout` | ✅ |
| TTS Yandex: api_key, voice, emotion, speed, folder_id, url | `YandexTtsConfig` — **полное 1:1** | ✅ идеально |
| Piper: voice_path, sentence_silence | `PiperConfig` | ✅ |
| Context: max_turns, ttl_seconds | `core.context.*` | ✅ |
| Devices: список name/host/psk + esphome port + public_base_url | `core.devices[]`, `core.esphome.port`, `core.audio.public_base_url` | ✅ (placement public_base_url у нас в `audio`) |
| Network: external_proxy, audio host/port/ttl | `core.network`, `core.audio` | ✅ |
| System prompt: путь, текст, плейсхолдер `<<<<<TDW>>>>>`, reset-to-default | `core.prompt.system_prompt_path` + `templates/default_prompt.md` | ✅ совпало, даже плейсхолдер тот же |
| Log: waterfall стадий, drawer с раундами/tool-calls/аудио | — (см. §7) | ✅ дизайн правильный, бэкенда нет |
| «Restart required» / per-field restart-флаги | `apply: live\|rebuild` в метаданных полей | ⚠️ концепция есть, трекинга нет (§7) |

Вывод: **UX-модель дизайнера и наша архитектура совпадают**. Это сэкономит много времени.

---

## 3. Несоответствие именования (архитектурно важно)

Дизайнер думает плоскими env-именами; у нас — вложенный документ + schema-driven формы:

| Дизайнер (`data.js`) | Наш `config.json` |
|---|---|
| `INTENT_API_KEY`, `INTENT_MODEL` | `llm.instances.openrouter.{api_key, model}` |
| `STT_API_KEY`, `STT_MODEL` | `stt.instances.groq.{api_key, model}` |
| `YANDEX_TTS_VOICE`, `YANDEX_TTS_SPEED`, … | `tts.instances.yandex.{voice, speed, …}` |
| `VAD_SILENCE_MS`, … | `core.vad.silence_ms`, … |
| `CONTEXT_MAX_TURNS`, … | `core.context.max_turns`, … |
| `EXTERNAL_PROXY`, `AUDIO_PORT`, … | `core.network.external_proxy`, `core.audio.port` |

**Рекомендация:** панель не должна знать ни env-имён, ни структуры провайдеров. Она запрашивает наш
`GET /catalog` (категории → провайдеры → JSON Schema из `ConfigModel.model_json_schema()` + значения) и
**рендерит формы из схемы**. Тогда новый провайдер появляется в UI сам, без правок фронта — ровно ради
этого делалась плагинная система. Сейчас `data.js` нужно заменить на загрузку `catalog()`.
да, тут перепиши просто фронт


---

## 4. Настройки, которые дизайнер ВЫДУМАЛ (редактируемы в UI, но захардкожены / нет в бэкенде)

| Поле в UI | Реальность в коде | Что делать |
|---|---|---|
| **LLM «System replies»**: rate_limit / empty_after_tools / empty — редактируемые | Это **хардкод-константы** в [llm.py](../src/llm.py): `EMPTY_REPLY_AFTER_TOOLS`, `EMPTY_REPLY_FALLBACK`, текст 429. (И текст 429 в макете другой, чем у нас.) | Либо вынести в `LlmConfig`/`core` как редактируемые строки (просто), либо в UI пометить read-only. Рекомендую вынести в конфиг — дёшево и полезно. |
вынести к конфиг


| **STT groq**: Language (`ru`), Temperature (`0`), Request timeout (`60s`) | Захардкожено в `GroqSttBackend.transcribe` (`language="ru"`, `temperature="0"`, `timeout=60`). В UI честно помечены `(hardcoded)`. | Решить: вынести в `GroqSttConfig` или оставить read-only. Язык русский — вынести стоит. |
вынести к конфиг


| **LLM Request timeout (`300s`)** | Захардкожено в `OpenAICompatLlmBackend` (`timeout=300`). Помечено `(hardcoded)`. | По желанию вынести в `LlmConfig`. |
вынести к конфиг

| **VAD preset (Fast/Balanced/Patient)** | Бэкенд-поля нет — это **UI-макрос** над `silence_ms`+`min_speech_ms`. | Оставить чисто фронтовым пресетом; маппинг пресет→значения держать в панели. Норм. |
Ок

| **RUAccent** (вся стадия: enabled/model/useDict/homographs/dict/preview) | **Не реализовано вообще** — ни стадии, ни провайдера, ни конфига. См. §6. | Отдельная фича. Пока стадия в UI должна быть «coming soon» / скрыта. |
да, пока скрыть

| Device **fw** (прошивка), **rssi** (сигнал) | Это не настройки, а **живые метрики устройства** (см. §7). `fw` достаётся из `device_info.esphome_version`, `rssi` — отдельный сенсор, может быть недоступен. | Перенести из «настроек» в метрики. |
ок


| Подписи «stored encrypted» у ключей (STT/LLM/TTS) | Мы **не шифруем** ключи (по твоему решению — обычные строки, доверенная зона). | Поправить копирайт: «plain field, trusted-LAN», убрать «encrypted». Маскирование показа (`SHOW`) — опционально, не критично. |
переименовать, да. в панели максировать.
---

## 5. Настройки, которые дизайнер ЗАБЫЛ (есть у нас — нет в UI)

| Наш конфиг | Где должно быть в UI | Статус |
|---|---|---|
| **`core.weather.api_key` + `core.weather.city`** | Нигде. Погода подмешивается в системный промпт (OpenWeatherMap), но **экрана настройки нет**. | ❌ Главная забытая настройка. Добавить блок «Weather» (ключ + город), напр. на страницу Integrations или Network. |
погода уйдет в mcp

| `core.context.dir` (`CONTEXT_DIR`) | Есть в `data.js`, но на странице Context **не выведено** (только turns + ttl). | Минор — «advanced», можно скрыть, но в `data.js` он есть. |
не надо, CONTEXT_DIR захардкодить вообще, он всегда data

| MCP `name` / `enabled` на сервер | UI это рисует (мульти-сервер), а в нашем `core.mcp` — **один сервер, без name/enabled** (§6). | Это про фичу мульти-MCP, не про забытое поле. |
у нас будет мультисервер

| `core.tts_timeout` (отдельно от провайдера) | В UI под Yandex как `TTS_TIMEOUT` — ок, но это core-уровень, общий для всех бэкендов. | Минор, учесть при маппинге. |


---

## 6. Две вещи в UI, которых нет как фич (не просто «настройка»)

### RUAccent (стадия 04)
Дизайнер сделал полноценный экран: тумблер, выбор модели (`tiny/turbo/big`), словарь, омографы,
предпросмотр, «+N ms» в impact. **В бэкенде этого нет**: нет стадии `accentuator`, нет провайдера, нет
поля в конфиге, пайплайн её не вызывает (ударения сейчас ставит LLM знаком `+`, переводит `text.py`).
- Чтобы оживить экран — отдельная задача: стадия `accentuator` + провайдер `ruaccent`
  ([Den4ikAI/ruaccent](https://github.com/Den4ikAI/ruaccent)) + место в `build_pipeline` + поле в
  лог-таймингах. Реальные имена моделей RUAccent: `tiny/turbo3.1/big_poetry` (в макете — `tiny/turbo/big`).
- До реализации: пометить стадию в UI как недоступную.

### Мульти-MCP с промптом на сервер
UI рисует **список** серверов, у каждого `name`, `enabled`, `token`, **`prompt`** (добавляется в
системный промпт), и список инструментов (read-only). У нас `core.mcp` — **один** `{url, token}`,
`McpToolHub` коннектится к одному адресу, инструменты читает в рантайме, но `name/enabled/prompt`/список
наружу не отдаёт.
- Чтобы оживить — задача: `core.mcp` → список `{id,name,url,token,enabled,prompt}`; `McpToolHub` →
  агрегатор по N серверам; сборка системного промпта добавляет `prompt` каждого активного сервера;
  API отдаёт список инструментов на сервер. (Эту мульти-MCP-фичу ты уже хотел раньше.)

---

## 7. Метрики и observability — что придётся ДОПИСЫВАТЬ (самый большой кусок)

Весь раздел Operations (Dashboard KPI + Log + статусы) дизайнер нарисовал, **но в бэкенде сейчас НЕТ
ничего из этого**: пайплайн меряет тайминги и пишет их в loguru, но **не персистит**. По дизайну для
лога предусмотрен SQLite `data/runs.db` — он **не реализован**. Что конкретно нужно:

### 7.1. Хранилище прогонов (`data/runs.db`, SQLite) — не реализовано
На каждый прогон UI ждёт запись: `time, device, stt (текст), llm (текст), result, reason, tokens,
model, audio{ms,bytes,fmt}, t{vad,stt,llm,ruaccent,tts}, rounds[]`.

| Поле UI | Есть ли сейчас | Что дописать |
|---|---|---|
| `time, device` | да (в логах) | писать строкой в БД |
| `stt`, `llm` тексты | да | сохранять |
| `result` (ok/tool/empty/error) | частично (по событиям) | вывести явный статус прогона |
| `reason` (endpoint/maxlen/no_speech/device_stop) | **да** (в пайплайне) | сохранять; **`tts_timeout` из макета — это не reason, а `result=error` + `error.stage=TTS`** |
| `tokens`, `model` | да (`total_tokens`, `data.model` логируются) | сохранять |
| `t.stt / t.llm / t.tts` | да (меряются `time.perf_counter`) | сохранять |
| **`t.vad`** (длительность захвата) | **нет как длительность** — VAD идёт непрерывно | измерять время от старта до финализации; для `no_speech` это таймаут (в макете `8000`) |
| **`t.ruaccent`** | нет (стадии нет) | появится с RUAccent |
| **`audio.ms`** (длительность аудио) | **нет** (есть только bytes+mime) | вычислять из mp3/pcm |
| `audio.bytes/fmt` | да (`len(audio)`, mime) | сохранять |
| **`rounds[]`** (раунд, note, `tokens` за раунд, tool-calls с `args`+`result`) | **только в loguru, не структурировано** | захватывать в агентном цикле `llm.py` и сохранять |
| **`error{stage,text}`** | частично («TTS failed: …» в логах) | структурировать: какая стадия упала + текст |

### 7.2. Агрегаты / KPI (Dashboard + футер лога) — не реализовано
`Requests · 24h`, `p50`/`p95 latency`, `Error rate`, спарклайны-тренды, **средняя латентность на стадию**
в service-map, футер «X of N runs · p50 · p95 · 1 error · 1 empty».
- Всё это — **SQL-агрегации поверх `runs.db`**. Нужны запросы (perc/avg/count за окно) + endpoint
  `/metrics`. Пока хранилища нет — KPI неоткуда брать.
сделать

### 7.3. Метрики устройств (страница Devices) — частично доступны
- `status` online/offline — **есть** (через `ReconnectLogic`), вывести.
- `fw` (прошивка) — **есть**: `device_info.esphome_version` (сейчас логируется, не отдаётся).
- `rssi` (сигнал Wi-Fi) — **нет**: нужен сенсор на устройстве, может быть недоступен → опционально.
rssi — добавить в конфиг еспхоума

### 7.4. Активные диалоги (страница Context) — выдуманная метрика, выводима
`activeDialogs: [{device, turns, idle}]` — не хранится, но **выводится на лету** из файлов контекста:
`turns` = строк/2, `idle` = `now - mtime`. Нужен endpoint.
это убрать из UI, не надо


### 7.5. Системные метрики (страница System) — частично
`version, uptime, started, running` — процессные метрики, сейчас **не отдаются** (нужно завести
версию приложения + засечь время старта). `running` тривиально.
сделать

### 7.6. «Pending restart» + список отложенных изменений — не реализовано
UI показывает баннер «N changes need restart» и список (`{stage, what}`). Наш `ConfigService.apply()`
есть, но **не трекает**, какие изменённые поля относятся к классу `rebuild`. Нужно: проставить
`apply: live|rebuild` всем полям (сейчас только у части) и в `apply()` считать diff по классу
применения → флаг + список.
по возможности избавиться от необходимости перезагрузки вообще, проблейма уйдет сама собой

### 7.7. Test-действия (кнопки в UI) — нет бэкенда
«Run self-test», «Test connection» (MCP/устройство/proxy/public_base_url), «Recognize sample» (STT),
«Speak» (TTS preview), «Re-run preview» (RUAccent) — всем нужны **тестовые endpoint'ы/действия** на
бэкенде (часть ложится на `provider.options()` и отдельные test-ручки).
сделать, да

---

## 8. HTTP API, который ждёт панель (сейчас его нет)

Чтобы заменить `data.js` на реальные данные, бэкенду нужен HTTP-слой (его пока нет вообще):

- `GET /catalog` — категории + провайдеры + JSON Schema + значения (из `ConfigService.catalog()`).
- `GET /config`, `PATCH /config` — чтение/патч значений (через `ConfigService.apply()`).
- `GET /options?category&plugin&field` — динамические списки (голоса/модели) из `provider.options()`.
- `GET /runs` (список + фильтры device/result/reason/поиск) и `GET /runs/{id}` (детали: rounds, аудио).
- `GET /metrics` — KPI/p50/p95/error-rate/спарклайны.
- `GET /devices` — статусы/fw/(rssi).
- `GET /context/active` — активные диалоги.
- `GET /system`, `POST /restart` — статус/аптайм/версия/перезапуск + pending-changes.
- `GET/PUT /prompt` — текст системного промпта (+ preview с подстановкой).
- `POST /test/*` — проверки соединений/сэмплов.
- Аудио прогонов для плеера — отдавать через уже существующий audio-server или `/runs/{id}/audio`.

---

### Итог
Дизайн отличный и архитектурно совпадает с нашей плагинной системой — переиспользуем. Перед интеграцией:
переключить формы с env-имён на наш `catalog()`; вынести/пометить выдуманные поля; добавить забытую
**погоду**; и главное — **построить observability-бэкенд (runs.db + метрики)**, которого сейчас нет,
плюс отдельными фичами **мульти-MCP** и **RUAccent**.
