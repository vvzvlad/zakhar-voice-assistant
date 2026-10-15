"""Mutable runtime holder shared by all pipelines.

The live source of truth at runtime: pipelines read config and backends THROUGH
this holder on every request, so a reconfiguration takes effect by either
mutating ConfigService's document (live fields, surfaced via the properties
below) or by swapping a backend/subsystem reference here — without
rebuilding the pipelines.
"""


class Runtime:
    def __init__(self, svc, *, vad_backend, stt_backends, llm_backends, tts_backends,
                 hub, audio_server, runs_store=None, run_events=None,
                 prompt_store=None, stress_backend=None, wakeword_backend=None):
        self.svc = svc
        # Swappable runtime objects (rebuilt by the reconfigurator in later tiers).
        self.vad_backend = vad_backend
        # Wakeword verification stage backend, the second-stage gate between VAD and
        # STT. Defaults to None so the pipeline guards a None/passthrough backend and
        # simply accepts (skips the gate) until a real verifier is loaded.
        self.wakeword_backend = wakeword_backend
        # Built STT / handler (llm) backends keyed by provider id: one per provider an
        # enabled recognition chain references (core.chains). The Reconfigurator
        # adds, rebuilds and drops entries as the chains and instances change.
        self.stt_backends = stt_backends
        self.llm_backends = llm_backends
        # Accent (Russian stress) stage backend, between LLM and TTS. Defaults to
        # None so test fixtures that don't pass it keep working (the pipeline guards
        # for a None backend and simply skips the stage).
        self.stress_backend = stress_backend
        # Built TTS backends keyed by provider id: one per enabled variant (tts.variants).
        # The pipeline walks the variants in config order; the Reconfigurator adds,
        # rebuilds and drops entries as the variants and instances change.
        self.tts_backends = tts_backends
        self.hub = hub
        self.audio_server = audio_server
        self.runs_store = runs_store
        self.run_events = run_events
        self.prompt_store = prompt_store  # PromptStore (system-prompt profiles)
        # Set post-construction in app.py; mutated by the Reconfigurator on hot-reload.
        self.manager = None         # DeviceManager (used by Tier 3c device rebuild)
        self.scheduler = None       # ReminderScheduler | None (Tier 3c reminders toggle)
        self.reminders_store = None  # RemindersStore | None (swapped by a reminders toggle)
        self.panel = None           # PanelServer (so a runs-store swap reaches its endpoints)
        self.zc = None              # zeroconf AsyncZeroconf (Tier 3c device rebuild)

    @property
    def core(self):
        """Current CoreConfig — always reflects the latest applied document."""
        return self.svc.core
