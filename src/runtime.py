"""Mutable runtime holder shared by all pipelines.

The live source of truth at runtime: pipelines read config and backends THROUGH
this holder on every request, so a reconfiguration takes effect by either
mutating ConfigService's document (live fields, surfaced via the properties
below) or by swapping a backend/subsystem reference here (later tiers) — without
rebuilding the pipelines.
"""


class Runtime:
    def __init__(self, svc, *, stt_backend, llm_backend, tts_backend,
                 hub, audio_server, runs_store=None, run_events=None):
        self.svc = svc
        # Swappable runtime objects (rebuilt by the reconfigurator in later tiers).
        self.stt_backend = stt_backend
        self.llm_backend = llm_backend
        self.tts_backend = tts_backend
        self.hub = hub
        self.audio_server = audio_server
        self.runs_store = runs_store
        self.run_events = run_events
        # Set post-construction in app.py; mutated by the Reconfigurator on hot-reload.
        self.manager = None         # DeviceManager (used by Tier 3c device rebuild)
        self.scheduler = None       # ReminderScheduler | None (Tier 3c reminders toggle)
        self.reminders_store = None  # RemindersStore | None (swapped by a reminders toggle)
        self.panel = None           # PanelServer (so a runs-store swap reaches its endpoints)
        self.zc = None              # zeroconf.Zeroconf (Tier 3c device rebuild)

    @property
    def core(self):
        """Current CoreConfig — always reflects the latest applied document."""
        return self.svc.core

    @property
    def llm_cfg(self):
        """Validated config of the selected LLM provider (live fields like
        reply_* / max_tool_rounds are read from here per request)."""
        return self.svc.get("llm")
