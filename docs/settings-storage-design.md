# Архитектура конфигурации (in-repo, провайдеры со своими схемами)

Новая система настроек, проектируется с нуля. **Старый `.env`-монолит и плоский `Settings`
выбрасываются целиком** — никакой совместимости и миграций. Источник истины — провайдеры стадий,
каждый сам декларирует свои настройки; конфиг-ядро про конкретные провайдеры (Yandex SpeechKit и т.п.)
не знает ничего.

Зафиксированные ограничения (определяют простоту дизайна):

- **Все провайдеры — в репозитории, в коде.** Внешних/сторонних плагинов не будет никогда → нет
  entry points, нет динамического discovery, нет публичного API контракта.
- **Стадии конвейера фиксированы**: `STT → LLM → (accentuator, опц.) → TTS`. Новые типы стадий не
  добавляются → конвейер **захардкожен**, никаких generic-слотов и типизированных пайпов.
- **Типы данных между стадиями статичны** (PCM → text → text → audio).
- **Внутри стадии провайдеров может быть много** (TTS: teratts/piper/yandex и т.д.) → выбор провайдера
  на стадию + самоописываемые настройки провайдера — это и есть ядро дизайна, оно остаётся.
- **Секреты — обычные поля конфига** (`SecretStr`, маскируются в API). Отдельного хранилища секретов нет.

Связано с панелью: [web-panel-brief.md](web-panel-brief.md).

---

## 0. Что было не так (и почему рушим)

`src/settings.py` — один `BaseSettings` с плоским списком ~40 полей: `yandex_tts_voice`,
`piper_voice_path`, `vad_silence_ms`… Чтобы добавить провайдера, надо править общий конфиг-модуль,
тесты, `.env.example`, фабрики. Конфиг-ядро «знает» параметры каждого провайдера, панель пришлось бы
хардкодить под каждый. Тупик.

**Цель:** провайдер стадии = самодостаточный модуль, который приносит свою схему настроек, валидацию
и реализацию. Добавить новый TTS = положить один файл. Ноль правок в ядре и в панели.

---

## 1. Принципы

1. **Self-describing providers.** Каждый провайдер декларирует свою схему настроек сам (pydantic-модель).
2. **Plugin-agnostic core.** Ядро оперирует абстракциями «категория стадии / провайдер / схема /
   значения» и не содержит ни одного имени поля конкретного провайдера.
3. **Schema-driven UI.** Панель строит формы автоматически из JSON Schema провайдера; новый провайдер
   появляется в UI сам.
4. **Конвейер захардкожен.** Порядок стадий и типы между ними фиксированы в коде (а не собираются из
   слотов). Динамична только начинка стадии — какой провайдер выбран.
5. **Единообразие рендера, не классов.** И провайдеры, и «ядровые» настройки (контекст/аудио/VAD/сеть)
   отдают панели JSON Schema одним способом — но это НЕ значит, что они одного класса (core — не
   провайдеры, см. §8).

---

## 2. Слои

```text
┌──────────────────────────────────────────────────────────────────┐
│ Web-панель / API        рисует формы из JSON Schema                │
├──────────────────────────────────────────────────────────────────┤
│ ConfigService           catalog() · get() · apply(patch) · on_change()   ← plugin-agnostic
│   ├── REGISTRY (статический dict)  провайдеры по категориям stt/llm/tts/accentuator
│   ├── config_store      load()/save() → data/config.json (атомарно, .bak, 0600)
│   └── CoreConfig        plain pydantic-секции: context, audio, vad, network
├──────────────────────────────────────────────────────────────────┤
│ build_pipeline()  тонкая функция: берёт выбранные провайдеры →     │
│                   собирает фиксированный STT → LLM → [accent] → TTS │
├──────────────────────────────────────────────────────────────────┤
│ Провайдеры (in-repo): TTS{teratts,piper,yandex} STT{groq,vosk}     │
│   LLM{openrouter,groq} Accentuator{ruaccent}; MCP{http}/Device{esphome} (multi) │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Конвейер захардкожен

Никаких слотов и обобщённой композиции. Сборка — явная тонкая функция; `Pipeline` из
[`src/pipeline.py`](../src/pipeline.py) и так принимает стадии явными аргументами:

```python
def build_pipeline(cfg: ConfigService, deps: Deps) -> Pipeline:
    stt  = cfg.create("stt")                       # выбранный провайдер STT
    llm  = cfg.create("llm")
    accent = cfg.create("accentuator") if cfg.enabled("accentuator") else None
    tts  = cfg.create("tts")
    return Pipeline(stt_backend=stt, llm=llm, accentuator=accent, tts_backend=tts, ...)
```

Связь стадий по типам (PCM→text→text→audio) статична и неявна — типизированный пайп не нужен.

---

## 4. Контракт провайдера и пример (ядро про него ничего не знает)

Провайдер = обычный базовый класс (не `Protocol` — внешних реализаций нет): метаданные + своя
pydantic-схема + фабрика рантайм-объекта стадии.

```python
class Provider:                          # base, in-repo only
    category: str                        # "stt" | "llm" | "tts" | "accentuator" | "mcp" | "device"
    id: str                              # unique within category, e.g. "yandex"
    label: str                           # human label for UI
    ConfigModel: type[BaseModel]         # provider's own settings schema

    def create(self, cfg: BaseModel, deps: "Deps"): ...      # build the stage runtime object
    def options(self, field: str, cfg, deps) -> list | None: # optional dynamic lists (voices/models)
        return None
```

```python
class YandexTtsConfig(BaseModel):
    api_key: SecretStr = Field(..., json_schema_extra={"secret": True, "apply": "rebuild"})
    voice:   str       = Field("zahar", json_schema_extra={"widget": "select", "options": "dynamic"})
    emotion: Literal["neutral", "good", "evil"] = "neutral"
    speed:   float     = Field(1.0, ge=0.1, le=3.0, json_schema_extra={"widget": "slider"})
    folder_id: str     = ""
    url:     str       = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

@register
class YandexTtsProvider(Provider):
    category = "tts"; id = "yandex"; label = "Yandex SpeechKit"
    ConfigModel = YandexTtsConfig

    def create(self, cfg: YandexTtsConfig, deps) -> TtsBackend:
        return YandexTtsBackend(deps.http_cloud, api_key=cfg.api_key.get_secret_value(),
                                voice=cfg.voice, emotion=cfg.emotion, speed=cfg.speed,
                                folder_id=cfg.folder_id, url=cfg.url, timeout=deps.tts_timeout)

    def options(self, field, cfg, deps):
        return YANDEX_VOICES if field == "voice" else None
```

Весь Yandex-specific (voice/emotion/speed) живёт **только здесь**. Добавить TTS = ещё один такой файл.

---

## 5. Метаданные полей → JSON Schema → авто-формы

Метаданные кладём в `Field(json_schema_extra=...)`; `model_json_schema()` их выносит, панель читает:

- `secret: true` — поле `SecretStr`; API маскирует (`{"is_set": true}`), в логи не пишем, запись —
  отдельной операцией «заменить».
- `apply: "live" | "rebuild"` — применять сразу или пересобирать стадию.
- `widget`, `min`/`max` (через `ge`/`le`), `options: "dynamic"` — подсказки рендера.

Никакого хардкода «у Яндекса есть голос» — панель рисует форму из схемы провайдера.

---

## 6. Реестр (статический, in-repo)

```python
REGISTRY: dict[str, dict[str, Provider]] = defaultdict(dict)

def register(cls):                       # decorator: self-registration
    p = cls()
    REGISTRY[p.category][p.id] = p
    return cls
```

Discovery — простой явный импорт пакета `src/plugins/` при старте (каждый модуль регистрируется
декоратором). Никаких entry points и динамической загрузки — все провайдеры известны на этапе сборки.

---

## 7. Хранение значений

Хранилище конфига — **JSON-файл `data/config.json`** (и точка; SQLite в этой системе используется
только для лога прогонов, см. §11). Доступ — две функции с атомарной записью, без лишней абстракции:

```python
# src/config_store.py
def load() -> dict: ...
def save(doc: dict) -> None: ...   # atomic: temp + fsync + os.replace; keep .bak; file mode 0600
```

Документ generic; стадии-провайдеры имеют форму `{selected, instances}`, мульти-категории (mcp/device) —
список инстансов. Значения каждого среза валидируются **схемой соответствующего провайдера** (берётся
из `REGISTRY`). Секреты лежат тут же как обычные значения (`SecretStr`), маскирует их API-слой по
флагу `secret:true`.

```jsonc
{
  "version": 1,
  "stt": { "selected": "groq", "instances": { "groq": { "model": "whisper-large-v3-turbo", "api_key": "gsk_…" }, "vosk": {} } },
  "llm": { "selected": "openrouter",
           "instances": { "openrouter": { "model": "anthropic/claude-haiku-4.5", "api_key": "sk-or-…",
                                          "temperature": 0.8, "max_tokens": 4096, "max_tool_rounds": 5 } } },
  "accentuator": { "enabled": false, "selected": "ruaccent", "instances": { "ruaccent": { "model": "tiny" } } },
  "tts": { "selected": "yandex",
           "instances": { "yandex": { "voice": "zahar", "speed": 1.0, "api_key": "AQVN…" },
                          "piper":  { "voice_path": "models/ru_RU-ruslan-medium.onnx", "sentence_silence": 0.4 } } },
  "mcp":     [ { "instance_id": "home",   "plugin": "mcp-http", "values": { "url": "http://10.0.0.5:8001/mcp", "token": "", "prompt": "…" } } ],
  "devices": [ { "instance_id": "living", "plugin": "esphome",  "values": { "host": "10.0.0.5", "psk": "…" } } ],
  "core": { "context": { "max_turns": 5, "ttl_seconds": 300 },
            "audio":   { "host": "0.0.0.0", "port": 8200, "ttl": 300, "public_base_url": "http://10.0.0.10:8200" },
            "vad":     { "aggressiveness": 2, "silence_ms": 800, "min_speech_ms": 200,
                         "max_utterance_ms": 15000, "no_speech_timeout_ms": 8000 },
            "network": { "external_proxy": "" } }
}
```

Формы хранения — ровно две (single `{selected,instances}` для stt/llm/tts/accentuator; список для
mcp/device), они известны заранее из фиксированного набора стадий — обобщённого параметра
«cardinality» у категории заводить не нужно.

> Замечание: секрет в JSON-документе хранится как значение; права файла `0600` и «не логировать
> SecretStr» — требования к `config_store`/логгеру, а не повод для отдельной подсистемы.

---

## 8. Ядровые настройки (`core.*`) — это plain-секции, не провайдеры

У `context/audio/vad/network` нет ни выбора провайдера, ни рантайм-фабрики стадии конвейера, ни
множественности — три из четырёх частей контракта провайдера для них пустые. Поэтому оформляем их
**не** как провайдеры, а как обычные pydantic-модели (`CoreConfig` с под-секциями). В панель они
попадают тем же способом — через `model_json_schema()`, — но мимо `REGISTRY` и без `create()`.

Единообразие в UI достигается тем, что **всё отдаёт JSON Schema**, а не тем, что всё — один класс.

---

## 9. ConfigService (ядро, plugin-agnostic)

```python
class ConfigService:
    def catalog(self) -> dict: ...      # категории + провайдеры + JSON Schema (+ маскированные значения)
    def get(self, category, instance=None) -> BaseModel: ...   # валидный типизированный конфиг
    def create(self, category): ...     # сконструировать рантайм-объект выбранного провайдера
    def apply(self, patch) -> Result: ...# validate against provider/core schema -> save -> emit change
    def options(self, category, plugin, field): ...            # проксирует provider.options()
    def on_change(self, cb): ...        # live-применение / пересборка стадии
```

`catalog()` — единственное, что нужно фронту, чтобы нарисовать вообще все настройки.

---

## 10. Применение изменений: live vs rebuild

Класс применения берётся из метаданных поля (`apply`):

- **live** — промпт, контекст, модель, погода, кастомные промпты MCP — читаются на следующий запрос.
- **rebuild** — провайдер стадии / HTTP-клиент / список колонок — `on_change` пересобирает только
  затронутую стадию (или, для MVP, помечает «нужен перезапуск» + кнопка).

---

## 11. Лог запросов — отдельно, SQLite (`data/runs.db`)

Конфиг и лог не смешиваем. **SQLite в этой системе — только для лога прогонов** (конфиг — JSON-файл).
Лог — append-heavy time-series с фильтрами и ретенцией: таблицы `runs` + `tool_calls`,
`journal_mode=WAL`, чистка по TTL/числу строк. Тайминги стадий приложение уже измеряет в `pipeline.py`
— писать строкой в `runs`.

---

## 12. Куда делся `.env`

Его нет вообще. Источник истины = `config_store` (JSON), правится через панель/API. Первый старт:
дефолтный документ без настроенных провайдеров — панель ведёт визардом.

**Аутентификации у панели нет:** она всегда работает в доверенной зоне (за reverse-proxy/туннелем или
на доверенном сегменте). Поэтому бутстрап-токен/логин не нужны, и из окружения не остаётся ничего —
адрес/порт самой панели — обычная `core`-настройка с дефолтами (как `audio`), правится в панели.

---

## 13. Как добавить нового провайдера (ради чего всё)

1. Создать `src/plugins/tts/elevenlabs.py`: `ElevenLabsConfig(BaseModel)` + `@register`-класс провайдера.
2. Реализовать `create()` → объект с `TtsBackend.synthesize(...)`.
3. (Опц.) `options()` для динамических списков (голоса).

Всё. Ноль правок в `ConfigService`, `config_store`, панели, других провайдерах. Провайдер появляется
в выпадашке TTS, форма рисуется из его схемы автоматически.

---

## 14. План рефакторинга кода (не в этом раунде)

- Завести `src/plugins/` и базовый `Provider` + интерфейсы стадий (часть ABC уже есть: `TtsBackend`,
  `SttBackend`).
- Перенести текущие реализации (teratts/piper/yandex, groq/vosk, openrouter/groq) в провайдеры со
  своими `ConfigModel`. Удалить плоский `Settings` и `.env`-загрузку.
- Написать `REGISTRY`+`@register`, `config_store` (атомарные load/save), `CoreConfig`, `ConfigService`,
  `build_pipeline`.
- API панели: `GET /catalog`, `GET/PATCH /config`, `GET /options`, `POST /restart`, `GET /runs`.
- Тесты: контракт провайдера (схема валидна, `create` даёт рабочий объект), round-trip `config_store`,
  маскирование секретов, generic-валидация патча.

---

## 15. Решения (всё зафиксировано, открытых вопросов нет)

- Внешние плагины / entry points — **нет**, только in-repo `plugins/`.
- Отдельный SecretStore — **нет**, секреты = `SecretStr`-поля в `config.json` (`0600`), маскирование в API.
- Generic-слоты конвейера — **нет**, конвейер захардкожен.
- `core.*` как провайдеры — **нет**, это plain pydantic-секции, отдающие JSON Schema.
- Аутентификация панели — **нет**, панель всегда в доверенной зоне; ENV не используется вообще.
- Хранилище: конфиг — **JSON** (`data/config.json`), лог прогонов — **SQLite** (`data/runs.db`,
  сложная структура под веб-фильтры/ретенцию).

---

### Итог
Провайдеры стадий — самодостаточные in-repo модули, каждый со своей pydantic-схемой; конфиг-ядро
(`REGISTRY` + `ConfigService` + `config_store`) plugin-agnostic; панель строит формы из JSON Schema
автоматически; конвейер захардкожен (динамична только начинка стадии); значения — в `data/config.json`
с атомарной записью, секреты — обычные `SecretStr`-поля; лог — отдельно в SQLite. `.env`-монолит
уничтожен. Добавление провайдера — один файл, ноль правок ядра и UI.
