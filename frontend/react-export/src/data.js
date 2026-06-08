// Zakhar Control Panel — app data model
const ZAPP = {
  meta: { version: "v0.9.2", uptime: "6d 04h 12m", started: "2026-06-02 09:48", running: true, pendingRestart: true, logLevel: "INFO" },

  // ── Pipeline stages (dashboard service map) ─────────────────────────────
  stages: [
    { key: "vad", name: "VAD", role: "Voice capture", provider: "WebRTC", detail: "silence 800 ms", status: "ok", restart: true },
    { key: "stt", name: "STT", role: "Speech → text", provider: "groq", detail: "whisper-large-v3-turbo", status: "ok", restart: true },
    { key: "llm", name: "LLM", role: "Reasoning + tools", provider: "openrouter", detail: "claude-haiku-4.5", status: "ok", mcp: { servers: 2, tools: 14 } },
    { key: "ruaccent", name: "RUAccent", role: "Stress marks", provider: "tiny", detail: "disabled", status: "off", optional: true },
    { key: "tts", name: "TTS", role: "Text → speech", provider: "yandex", detail: "voice · zahar", status: "ok", restart: true }
  ],
  stageOrder: ["vad", "stt", "llm", "ruaccent", "tts"],
  stageColor: { vad: "#64748b", stt: "#0891b2", llm: "#4f46e5", ruaccent: "#9333ea", tts: "#0d9488" },

  system: { speakersOnline: 3, speakersTotal: 4, proxy: true },

  resultMeta: {
    ok: { label: "OK", tone: "good" },
    tool: { label: "OK · tool", tone: "good" },
    empty: { label: "Empty", tone: "muted" },
    error: { label: "Error", tone: "bad" }
  },

  kpis: [
    { k: "Requests · 24h", v: "1.84k", u: "", d: ["up", "+12%"], spark: [8, 11, 9, 14, 12, 16, 13, 18], color: "#0d9488" },
    { k: "p50 latency", v: "2.41", u: "s", d: ["down", "−0.2s"], spark: [3.1, 2.9, 3.0, 2.6, 2.7, 2.4, 2.5, 2.41], color: "#2563eb" },
    { k: "p95 latency", v: "3.62", u: "s", d: ["flat", "±0"], spark: [3.5, 3.7, 3.6, 3.8, 3.5, 3.6, 3.6, 3.62], color: "#7c3aed" },
    { k: "Error rate", v: "1.4", u: "%", d: ["down", "−0.6%"], spark: [3, 2.4, 2.1, 1.8, 2.0, 1.6, 1.5, 1.4], color: "#dc2626" }
  ],

  // ── Stage settings models ───────────────────────────────────────────────
  vad: {
    preset: "balanced",
    fields: [
      { key: "VAD_AGGRESSIVENESS", label: "Filter aggressiveness", type: "int", min: 0, max: 3, value: 2, hint: "Higher = cuts non-speech harder (0–3)." },
      { key: "VAD_SILENCE_MS", label: "Trailing silence", unit: "ms", type: "int", value: 800, hint: "How much silence after speech ends a phrase." },
      { key: "VAD_MIN_SPEECH_MS", label: "Min speech", unit: "ms", type: "int", value: 200, hint: "Speech needed to arm the end-of-utterance detector." },
      { key: "VAD_MAX_UTTERANCE_MS", label: "Max utterance", unit: "ms", type: "int", value: 15000, hint: "Hard cap — finalize even without trailing silence." },
      { key: "VAD_NO_SPEECH_TIMEOUT_MS", label: "No-speech timeout", unit: "ms", type: "int", value: 8000, hint: "If no speech at all — end the turn." }
    ]
  },
  stt: {
    provider: "groq", providers: ["groq", "vosk"],
    groq: { STT_API_KEY: "gsk_••••••••••••••••••••••••7fA2", STT_MODEL: "whisper-large-v3-turbo", language: "ru", temperature: 0, timeout: 60 },
    vosk: { VOSK_MODEL_PATH: "models/vosk-model-small-ru-0.22" }
  },
  llm: {
    provider: "openrouter", providers: ["openrouter", "groq"],
    INTENT_API_KEY: "sk-or-••••••••••••••••••••3dB9",
    INTENT_MODEL: "anthropic/claude-haiku-4.5",
    temperature: 0.8, max_tokens: 4096, MAX_TOOL_ROUNDS: 5, timeout: 300,
    fallbacks: {
      rate_limit: "Сейчас много запросов, попробуй через секунду.",
      empty_after_tools: "Готово.",
      empty: "Я тебя не расслышала, повтори."
    }
  },
  ruaccent: {
    enabled: false,
    model: "tiny", models: ["tiny", "turbo", "big"],
    modelInfo: { tiny: "fast · ~0.3 GB", turbo: "balanced · ~0.8 GB", big: "most accurate · ~2.1 GB" },
    useDict: false, homographs: true,
    dict: [
      { word: "замок", accented: "за́мок" },
      { word: "духи", accented: "духи́" },
      { word: "берегу", accented: "берегу́" }
    ],
    previewIn: "вертолёт замок берегу",
    previewOut: "вертолёт за́мок берегу́"
  },
  tts: {
    backend: "yandex", backends: ["teratts", "piper", "yandex"], timeout: 30,
    teratts: { TTS_BASE_URL: "http://10.0.0.7:8080" },
    piper: { PIPER_VOICE_PATH: "models/ru_RU-ruslan-medium.onnx", TTS_SENTENCE_SILENCE: 0.4 },
    yandex: {
      YANDEX_TTS_API_KEY: "AQVN••••••••••••••••••••••V3q2",
      YANDEX_TTS_VOICE: "zahar",
      voices: ["zahar (m)", "ermil (m)", "filipp (m)", "alena (f)", "jane (f)", "omazh (f)"],
      YANDEX_TTS_EMOTION: "neutral", emotions: ["neutral", "good", "evil"],
      YANDEX_TTS_SPEED: 1.0,
      YANDEX_TTS_FOLDER_ID: "",
      YANDEX_TTS_URL: "tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    }
  },

  // ── MCP servers ─────────────────────────────────────────────────────────
  mcp: [
    {
      id: "m1", name: "node-red.home", url: "http://10.0.0.5:8001/mcp", token: "nr_••••••••5521", enabled: true, status: "online",
      prompt: "Это узлы Node-RED для управления освещением, шторами и сценами в квартире. Используй light.* для света, cover.* для штор. Не выдумывай несуществующие комнаты.",
      tools: [
        { name: "light.set", desc: "Set brightness/on-off for a room light" },
        { name: "light.list", desc: "List available lights and rooms" },
        { name: "cover.set", desc: "Open/close blinds by position %" },
        { name: "scene.activate", desc: "Activate a named scene" },
        { name: "timer.set", desc: "Start a named countdown timer" },
        { name: "media.volume", desc: "Set speaker volume level" },
        { name: "climate.set", desc: "Set target temperature" },
        { name: "sensor.read", desc: "Read a sensor value" },
        { name: "switch.toggle", desc: "Toggle a generic switch" }
      ]
    },
    {
      id: "m2", name: "scenes.local", url: "http://10.0.0.5:8044/mcp", token: "", enabled: true, status: "online",
      prompt: "Дополнительные сцены и плейлисты. media.* управляет музыкой.",
      tools: [
        { name: "media.play", desc: "Play a playlist or station" },
        { name: "media.pause", desc: "Pause playback" },
        { name: "scene.list", desc: "List saved scenes" },
        { name: "scene.save", desc: "Save current state as scene" },
        { name: "notify.push", desc: "Send a push notification" }
      ]
    },
    {
      id: "m3", name: "garden.pi", url: "http://10.0.0.9:8001/mcp", token: "gp_••••••••0090", enabled: false, status: "offline",
      prompt: "", tools: []
    }
  ],

  // ── System prompt ───────────────────────────────────────────────────────
  prompt: {
    path: "data/system_prompt.md",
    chars: 1487,
    text: `Ты — Захар, голосовой ассистент умного дома. Отвечай коротко, по-русски, дружелюбно и с лёгким характером. Никогда не выдумывай факты об устройствах.

Текущий контекст: <<<<<TDW>>>>>

Правила ответа:
1. Для управления домом используй инструменты MCP. Не описывай действие словами, если можешь его выполнить.
2. Ставь ударения знаком «+» перед ударной гласной в любых словах, где ударение неочевидно (напр. «з+амок»).
3. Если запрос непонятен — переспроси одной фразой.
4. Не зачитывай длинные списки; обобщай.

Формат: только текст для озвучивания, без разметки и эмодзи.`
  },

  // ── Context ─────────────────────────────────────────────────────────────
  context: {
    fields: [
      { key: "CONTEXT_MAX_TURNS", label: "Context depth", unit: "turns", type: "int", min: 1, value: 5, hint: "How many recent Q&A pairs are remembered and sent to the model." },
      { key: "CONTEXT_TTL_SECONDS", label: "Dialog TTL", unit: "s", type: "int", min: 0, value: 300, hint: "Idle time before a dialog resets (0 = always fresh)." },
      { key: "CONTEXT_DIR", label: "Context folder", type: "path", value: "data", hint: "Where per-speaker context files live. Usually unchanged." }
    ],
    activeDialogs: [
      { device: "kitchen", turns: 3, idle: "42s" },
      { device: "bedroom", turns: 1, idle: "2m 10s" },
      { device: "office", turns: 5, idle: "8s" }
    ]
  },

  // ── Devices (speakers) ──────────────────────────────────────────────────
  devices: [
    { id: "d1", name: "kitchen", host: "10.0.0.21", psk: "k7Hq••••••••wQ2", status: "online", fw: "HA Voice PE 2025.5", rssi: -52 },
    { id: "d2", name: "bedroom", host: "10.0.0.22", psk: "9bRt••••••••pL8", status: "online", fw: "HA Voice PE 2025.5", rssi: -61 },
    { id: "d3", name: "office", host: "10.0.0.23", psk: "m3Xy••••••••zV1", status: "online", fw: "HA Voice PE 2025.4", rssi: -47 },
    { id: "d4", name: "living room", host: "10.0.0.24", psk: "p0Wn••••••••aK5", status: "offline", fw: "HA Voice PE 2025.5", rssi: null }
  ],
  devicesCommon: { ESPHOME_PORT: 6053, PUBLIC_BASE_URL: "http://10.0.0.10:8200" },

  // ── Network / integrations ──────────────────────────────────────────────
  network: {
    EXTERNAL_PROXY: "socks5://10.0.0.2:1080",
    AUDIO_HOST: "0.0.0.0", AUDIO_PORT: 8200, AUDIO_TTL: 300,
    routes: [
      { stage: "STT", via: "proxy" }, { stage: "LLM", via: "proxy" },
      { stage: "RUAccent", via: "local" }, { stage: "TTS", via: "proxy" },
      { stage: "MCP", via: "local" }
    ]
  },

  // ── Request log (rich) ──────────────────────────────────────────────────
  requests: [
    {
      id: "r1", time: "14:32:07", device: "kitchen", stt: "включи свет на кухне",
      llm: "Готово, включила свет на кухне.", result: "ok", reason: "endpoint", tokens: 384, model: "claude-haiku-4.5",
      audio: { ms: 680, bytes: 41984, fmt: "mp3" },
      t: { vad: 140, stt: 520, llm: 910, ruaccent: 0, tts: 680 },
      rounds: [
        { type: "model", round: 1, note: "decided to call a tool", tokens: 142, calls: [{ name: "light.set", args: { room: "kitchen", on: true }, result: "{ ok: true, brightness: 80 }" }] },
        { type: "model", round: 2, note: "final answer", tokens: 242, calls: [] }
      ]
    },
    {
      id: "r2", time: "14:28:51", device: "bedroom", stt: "какая завтра погода",
      llm: "Завтра в Москве −3°, облачно, без осадков.", result: "ok", reason: "endpoint", tokens: 451, model: "claude-haiku-4.5",
      audio: { ms: 940, bytes: 58210, fmt: "mp3" },
      t: { vad: 160, stt: 610, llm: 1480, ruaccent: 0, tts: 940 },
      rounds: [{ type: "model", round: 1, note: "answered from weather context", tokens: 451, calls: [] }]
    },
    {
      id: "r3", time: "14:21:03", device: "office", stt: "поставь таймер на десять минут",
      llm: "Поставила таймер на десять минут.", result: "tool", reason: "endpoint", tokens: 612, model: "claude-haiku-4.5",
      audio: { ms: 540, bytes: 33120, fmt: "mp3" },
      t: { vad: 130, stt: 480, llm: 1320, ruaccent: 0, tts: 540 },
      rounds: [
        { type: "model", round: 1, note: "tool call", tokens: 168, calls: [{ name: "timer.set", args: { name: "office", seconds: 600 }, result: "{ ok: true, id: 'tmr_4471' }" }] },
        { type: "model", round: 2, note: "final answer", tokens: 444, calls: [] }
      ]
    },
    {
      id: "r4", time: "14:15:40", device: "kitchen", stt: "", llm: "", result: "empty", reason: "no_speech", tokens: 0, model: null,
      audio: null, t: { vad: 8000, stt: 0, llm: 0, ruaccent: 0, tts: 0 }, rounds: []
    },
    {
      id: "r5", time: "14:09:12", device: "living room", stt: "включи музыку погромче",
      llm: "—", result: "error", reason: "tts_timeout", tokens: 208, model: "claude-haiku-4.5",
      error: { stage: "TTS", text: "TTS backend timeout (yandex, 30 s) — no audio produced." },
      audio: null, t: { vad: 150, stt: 560, llm: 870, ruaccent: 0, tts: 0 },
      rounds: [{ type: "model", round: 1, note: "tool call", tokens: 208, calls: [{ name: "media.volume", args: { device: "living room", level: 80 }, result: "{ ok: true }" }] }]
    },
    {
      id: "r6", time: "14:02:55", device: "office", stt: "выключи весь свет",
      llm: "Выключила весь свет в квартире.", result: "tool", reason: "endpoint", tokens: 523, model: "claude-haiku-4.5",
      audio: { ms: 610, bytes: 37880, fmt: "mp3" },
      t: { vad: 145, stt: 505, llm: 1180, ruaccent: 0, tts: 610 },
      rounds: [
        { type: "model", round: 1, note: "tool call", tokens: 180, calls: [{ name: "light.list", args: {}, result: "[ kitchen, office, hall, bedroom ]" }] },
        { type: "model", round: 2, note: "tool call", tokens: 160, calls: [{ name: "scene.activate", args: { name: "all_off" }, result: "{ ok: true }" }] },
        { type: "model", round: 3, note: "final answer", tokens: 183, calls: [] }
      ]
    },
    {
      id: "r7", time: "13:58:10", device: "bedroom", stt: "поставь будильник на семь утра",
      llm: "Поставила будильник на 7:00.", result: "ok", reason: "endpoint", tokens: 290, model: "claude-haiku-4.5",
      audio: { ms: 520, bytes: 31200, fmt: "mp3" },
      t: { vad: 155, stt: 540, llm: 760, ruaccent: 0, tts: 520 }, rounds: [{ type: "model", round: 1, note: "tool call + answer", tokens: 290, calls: [{ name: "timer.set", args: { name: "alarm", at: "07:00" }, result: "{ ok: true }" }] }]
    },
    {
      id: "r8", time: "13:51:33", device: "kitchen", stt: "расскажи анекдот", llm: "Колобок повесился. Ладно, шучу — лучше включу тебе музыку.",
      result: "ok", reason: "endpoint", tokens: 412, model: "claude-haiku-4.5", audio: { ms: 1120, bytes: 70110, fmt: "mp3" },
      t: { vad: 170, stt: 590, llm: 1620, ruaccent: 0, tts: 1120 }, rounds: [{ type: "model", round: 1, note: "answered", tokens: 412, calls: [] }]
    }
  ]
};

export default ZAPP;
