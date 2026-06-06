# Plan: `zakhar-voice-assistant` — свой сервер вместо Home Assistant для колонки Voice PE

## Context (зачем это)

Сейчас колонка **HA Voice PE** (ESP32-S3) подключена к Home Assistant, а «умная» часть вынесена в [ha-voice-logic](/Users/vvzvlad/Data/Projects/ha-voice-logic): STT и LLM-интент идут в облако (Groq), команды умного дома уходят POST'ом на `SMARTHOME_URL` (Node-RED), TTS делает сам HA. HA выступает только координатором голосового пайплайна.

Цель — **полностью убрать HA из голосовой схемы**: новый сервер `zakhar-voice-assistant` сам встаёт на место HA, напрямую общается с колонкой, гоняет STT/LLM/TTS и возвращает озвучку. Проект делаем **по канону** `home-network/instructions/new-project-guide.md` (скелет `new-project/`).

### Ключевая архитектурная поправка к исследованию
Исследование советует форкнуть **OHF-Voice/linux-voice-assistant** — это **неверная сторона протокола**:
- В ESPHome Native API **устройство (колонка) — TCP-сервер на :6053**, а Home Assistant — **клиент** (`aioesphomeapi`), который к ней подключается.
- `linux-voice-assistant` — программный *сателлит* (сам прикидывается устройством, чтобы к нему подключался HA), т.е. заменяет железную колонку, а не HA.
- Чтобы заменить **HA**, наш сервер = **клиент `aioesphomeapi`**, коннектится к реальной колонке и драйвит voice_assistant. Подтверждено по исходникам HA `assist_satellite.py`.

### Зафиксированные решения (Q&A)
Docker на x86; STT/LLM — облако (Groq Whisper `whisper-large-v3-turbo` + Groq `openai/gpt-oss-120b`); TTS — локальный [home_assistant_tts](/Users/vvzvlad/Data/Projects/home_assistant_tts) (TeraTTS GLaDOS2, RU, `GET /synthesize/<text>` → MP3) **через абстракцию** под замену; команды не трогаем (`<command>device:value</command>` → POST `{"command":{...}}` на `SMARTHOME_URL`/Node-RED); мозг переносим **целиком**; шифрование Noise — по ключу колонки; колонка будет отвязана от HA. **Закладываем несколько колонок** (сейчас одна, дальше будут ещё): сервер обслуживает N устройств одновременно и независимо.

---

## Целевой поток

Каждая колонка обслуживается своим клиентом и своим независимым циклом; «мозг», TTS и audio-сервер общие.

```
Колонка Voice PE #N (ESP32, TCP-сервер :6053, on-device wake word)
        ▲ на КАЖДУЮ колонку — свой КЛИЕНТ aioesphomeapi (Noise PSK), ReconnectLogic
        │ subscribe_voice_assistant(handle_start, handle_stop, handle_audio)
        │ handle_start → return 0  (0 = аудио in-band по TCP, без UDP)
   zakhar-voice-assistant (asyncio, Docker, x86) — DeviceManager: по 1 клиенту+pipeline на колонку
   1. handle_start → send RUN_START, STT_START; сброс буфера
   2. handle_audio → копим PCM 16k/16/mono
   3. handle_stop  →
        STT:  PCM→WAV→Groq Whisper (lang=ru) → STT_END {"text": ...}   (пусто → RUN_END)
        LLM:  INTENT_START → Groq chat(промпт=дата+погода) → парсинг <command>
                 → POST SMARTHOME_URL → чистка текста → INTENT_END
        TTS:  TTS_START {"text": reply} → TeraTTS → MP3 в TTL-кэш
                 → TTS_END {"url": "http://<PUBLIC_BASE_URL>/tts/<id>.mp3"}
        RUN_END
        ▼ колонка сама GET'ом качает URL и проигрывает MP3
```
Контракт событий (из HA `assist_satellite.py`): `STT_END {"text"}`, `TTS_START {"text"}`, `TTS_END {"url"}`, `RUN_END` без данных. Блокирующих вызовов в loop нет — весь HTTP на `httpx.AsyncClient`.

---

## Структура проекта (за основу — скелет `new-project/`)

Старт: `cp -r home-network/instructions/new-project/* zakhar-voice-assistant/` (сохранив research-`.md` и `.code-workspace`), затем заполняем `src/` и адаптируем конфиги. Файлы скелета (Makefile, Dockerfile, pytest.ini, CI, .gitignore/.dockerignore, conftest-паттерн, settings-паттерн) берём **как есть**.

```
zakhar-voice-assistant/
  src/
    __init__.py
    settings.py        # pydantic-settings по канону скелета (см. ENV ниже)
    esphome_client.py  # NEW: APIClient + ReconnectLogic + subscribe_voice_assistant
    pipeline.py        # NEW: оркестрация цикла + последовательность событий
    stt.py             # NEW: PCM→WAV→Groq Whisper (httpx async)
    llm.py             # PORT groq_client (httpx async, семантика 1-в-1)
    prompt.py          # PORT (промпт + дата/время + погода)
    weather.py         # PORT (OpenWeatherMap, httpx async)
    commands.py        # PORT (<command> парсинг + POST SMARTHOME_URL, httpx async)
    context.py         # PORT 1-в-1 (data/context.txt, авто-сброс 60с)
    text.py            # PORT 1-в-1 (чистка ответа + фонетика) — чистая логика
    tts.py             # NEW: TtsBackend(ABC) + TeraTtsHttpBackend
    audio_server.py    # NEW: aiohttp-раздача MP3 по /tts/<id>.mp3 + /health
    app.py             # NEW: сборка (asyncio): поднять audio_server + esphome_client
  templates/default_prompt.md   # PORT персоны (плейсхолдер <<<<<TDW>>>>>)
  data/.gitkeep        # рантайм-стейт (context.txt, system_prompt.md) — volume
  tests/conftest.py    # выставить обязательные ENV ДО импорта src.settings
  tests/test_*.py
  main.py              # тонкий: loguru + asyncio.run(app.main())
  Dockerfile, docker-compose.yml, requirements.txt, requirements-dev.txt
  Makefile, pytest.ini, .env.example, README.md, AGENTS.md, .gitignore, .dockerignore
  .github/workflows/ghcr-check-publish.yml
```

---

## Конфигурация (`src/settings.py`, по правилам гайда)

- **Креды — без дефолта** (нет в ENV → падаем): `GROQ_API_KEY`, `WEATHER_API_KEY`.
- **Список колонок — без дефолта** (адреса+ключи своих устройств): `ESPHOME_DEVICES` — плоская строка по канону гайда («списки храним строкой, парсим в `@property`»). Формат: устройства через `;`, поля через `|`: `name|host|noise_psk;name2|host2|psk2` (разделители `;|` не встречаются в base64-PSK). Парсится в `@property` → список `DeviceConfig(name, host, psk)`. Так добавление колонки = правка одной ENV-строки.
- **Адреса своих сервисов — без дефолта**: `SMARTHOME_URL` (Node-RED), `TTS_BASE_URL` (home_assistant_tts), `PUBLIC_BASE_URL` (адрес, по которому колонки качают аудио, напр. `http://10.x.x.x:8200`).
- **Публичные внешние API — константы в коде** (как в ha-voice-logic): URL Groq chat/STT, OpenWeatherMap.
- **Не-секретная конфигурация — дефолты ок**: `ESPHOME_PORT=6053`, `GROQ_MODEL=openai/gpt-oss-120b`, `GROQ_STT_MODEL=whisper-large-v3-turbo`, `GROQ_PROXY=""`, `WEATHER_CITY=Moscow`, `TTS_BACKEND=teratts`, `TTS_TIMEOUT=30`, `AUDIO_HOST=0.0.0.0`, `AUDIO_PORT=8200`, `LOG_LEVEL=INFO`, пути `data/...`.

`.env.example` — полный список с плейсхолдерами (`ESPHOME_DEVICES=living|10.0.0.5|psk_here`); `conftest.py` выставляет обязательные ENV до импорта `src.settings` (как в скелете для `TELEGRAM_BOT_TOKEN`).

---

## Детали по модулям

- **`esphome_client.py`**: класс `DeviceClient` на одну колонку — `APIClient(host, ESPHOME_PORT, noise_psk=psk)` + `ReconnectLogic` (авто-реконнект независимо для каждой) + свой `Pipeline`; `subscribe_voice_assistant(handle_start→pipeline.on_start (return 0), handle_audio→pipeline.on_audio, handle_stop→pipeline.on_stop)`; отправка `send_voice_assistant_event(type, data)`. **`DeviceManager`** поднимает по одному `DeviceClient` на каждый элемент `settings.devices`. Запасной TTS-стрим (`TTS_STREAM_START`→`send_voice_assistant_audio`→`TTS_STREAM_END`) заложить, по умолчанию выкл.
- **`pipeline.py`**: **экземпляр на колонку** (свой буфер аудио + свой lock от перекрытия циклов; колонки работают параллельно и не мешают друг другу); последовательность событий как выше; зависимости (STT/LLM/TTS-клиенты, audio-сервер) инжектятся — они общие и stateless. Ошибки STT/LLM/TTS всегда закрывают цикл `RUN_END`; страховочный таймаут длины фразы.
- **`stt.py`**: PCM 16k/16/mono → WAV (`wave`+`io.BytesIO`) → multipart POST Groq Whisper (`model=GROQ_STT_MODEL, language=ru, response_format=json, temperature=0`, опц. прокси); ошибка → пустой текст.
- **Мозг (PORT, поведение 1-в-1)**: `llm.py` (Groq chat, спец-фраза на 429, `<command>` до чистки), `prompt.py`/`weather.py`/`commands.py`/`context.py`/`text.py`, `templates/default_prompt.md`. HTTP-вызовы переписываем на `httpx.AsyncClient` с теми же параметрами; чистая логика (`text`, парсинг `commands`) — дословно. **Контекст — per-device** (`data/context_<name>.txt`), чтобы реплики разных колонок не смешивались; пишется как лог `USER/GLADOS` (в промпт не подставляется — как сейчас).
- **`tts.py`**: `TtsBackend.synthesize(text, lang) -> (mime, bytes)`; `TeraTtsHttpBackend` → `GET {TTS_BASE_URL}/synthesize/<urlenc>`; выбор по `TTS_BACKEND`.
- **`audio_server.py`**: `aiohttp` web (один на сервер, общий для всех колонок); MP3 в TTL-кэше (dict + чистка по времени), ключ — уникальный id запроса, `GET /tts/<id>.mp3` (`audio/mpeg`) и `GET /health`.
- **`app.py`**: поднять общий `audio_server`, общие httpx-клиент и TTS-бэкенд, затем `DeviceManager.start()` (по клиенту на колонку); один общий asyncio event loop.

---

## Инфраструктура (по гайду + одно обоснованное отклонение)

- **requirements.txt** (пин `==`): `pydantic-settings`, `loguru`, `httpx`, `aioesphomeapi`, `aiohttp` (точные версии — текущие стабильные на момент сборки). **requirements-dev.txt**: `-r requirements.txt` + `pytest`, `pytest-asyncio`, `pytest-cov`, `respx`.
- **Dockerfile**: скелет как есть (`python:3.11-slim`, deps до кода, `--no-cache-dir`, `mkdir -p data`, exec-CMD, без EXPOSE). Системные пакеты не нужны (ffmpeg живёт в TTS-сервисе).
- **CI** `ghcr-check-publish.yml`: скелет; в джобе `test` в `env:` прокинуть обязательные плейсхолдеры (`GROQ_API_KEY`, `WEATHER_API_KEY`, `ESPHOME_DEVICES`, `SMARTHOME_URL`, `TTS_BASE_URL`, `PUBLIC_BASE_URL`).
- **docker-compose.yml — ОТКЛОНЕНИЕ от веб-варианта (есть причина):** это **не** публичный веб за Traefik, а LAN-сервис «устройство↔сервер». Поэтому:
  - убираем Traefik-лейблы; оставляем volume на `/app/data`, `restart: always`, ротацию логов, watchtower-лейбл, `TZ`;
  - **публикуем порт аудио в LAN**: `ports: ["${AUDIO_PORT}:${AUDIO_PORT}"]`, чтобы колонка достучалась до `PUBLIC_BASE_URL`;
  - исходящий коннект к колонке `:6053` и к Node-RED/TTS работает с bridge-сети;
  - healthcheck — на `/health` audio-сервера.
- **AGENTS.md / README.md** — по шаблону гайда (структура, `make install/test/run`, ENV, соглашения; комментарии в коде — на английском).

---

## Тесты (pytest, без сети — моки)
`test_settings` (падение при отсутствии креды; **парсинг `ESPHOME_DEVICES`** в список из 1 и N устройств, кривые строки), `test_commands` (парсинг/диспетч), `test_text` (чистка+фонетика), `test_weather` (парсинг ответа, `respx`), `test_stt` (корректность WAV-обёртки PCM), `test_pipeline` (последовательность событий voice_assistant с замоканными STT/LLM/TTS; **два независимых pipeline-экземпляра не делят буфер/контекст**). Внешний HTTP мокаем `respx`; реальную колонку/сеть не дёргаем.

---

## Риски и закрытие
1. **TTS URL vs стрим** — берём URL (поддержан прошивкой); стрим — запасной (решаем на Stage 1).
2. **Достижимость `PUBLIC_BASE_URL`** колонкой по LAN — порт опубликован, адрес верный (Stage 1).
3. **Noise-ключ** — hard-блокер, берём из YAML колонки.
4. **Имена `VoiceAssistantEventType`/ключи data** — сверить с enum `aioesphomeapi` при сборке (контракт `text`/`url` подтверждён).
5. **STT батчевый** — копим до `handle_stop`; приемлемо.
6. **Конец речи** — on-device VAD (`USE_VAD`) → `handle_stop`; страховка таймаутом.
7. **Мультиколоночность** — состояние строго per-device (буфер/lock/контекст на экземпляр), общий код stateless; реконнект и сбой одной колонки не влияют на другие; одна TTL-запись аудио на запрос (id уникален).

## Вне scope (MVP)
Таймеры, announcements/continue-conversation, английский TTS, подстановка контекста в промпт. (Несколько колонок — **в scope**: поддержка нескольких устройств закладывается сразу.)

---

## Верификация (поэтапно)
- **Stage 1 (де-риск транспорта):** запустить, подключиться к колонке по Noise-ключу, wake word → в логах `handle_start`→`handle_audio` (байты)→`handle_stop`; колонка реагирует на `RUN_START/STT_START` (LED); проверить доступность `PUBLIC_BASE_URL` с колонки.
- **Stage 2 (пайплайн):** полный цикл «фраза → STT → LLM → команда видна в Node-RED → TTS проигран».
- **Stage 3 (устойчивость):** реконнект при ребуте колонки; пустой STT; Groq 429 → спец-фраза; таймаут TTS.
- **Тесты/CI:** `make test` локально зелёные; CI `test`→`build`→push в ghcr.

---

## Режим реализации
Делегирование (по умолчанию). Подзадачи: (1) скопировать скелет + settings/ENV/инфра под гайд (вкл. парсер `ESPHOME_DEVICES`); (2) транспорт `esphome_client` (`DeviceClient` + `DeviceManager` на N колонок); (3) `pipeline` (per-device) + `stt`; (4) порт мозга (`llm/prompt/weather/commands/context`(per-device)`/text`+промпт) на httpx; (5) `tts`+`audio_server`(общий)+`app`+compose-отклонение; (6) тесты (вкл. парсинг устройств и независимость экземпляров). После кода — обязательный прогон ревью-субагента и верификация (Phase 4–5). Реальная среда (ключ колонки, Node-RED, TTS) — у пользователя; код пишем так, чтобы Stage 1–3 он прогнал у себя.
