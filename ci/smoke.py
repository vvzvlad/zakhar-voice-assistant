"""Smoke gate for the zakhar-voice-assistant image, run on the CI runner against an already-built
image.

Three parts:
* This file runs on the runner as `python3 ci/smoke.py` and drives `docker` against $SMOKE_IMAGE:
  container-level questions (`docker inspect` CMD/WORKDIR, the real command's log, the first-boot
  config on disk, liveness at the end).
* The IMAGE PROBE and the HTTP PROBE are string constants fed to `docker exec -i <name> python -u -`.
  The image probe checks code and files inside the image (version stamp, native VAD runtime,
  frontend build, imports); the HTTP probe checks that the running app serves.

The job runs inside act_runner's container while `docker` drives a daemon outside it, so nothing
here may reach a container by address. No port is published (a published port lands in the outer
daemon's namespace); the probes run inside the container, where 127.0.0.1 means what the
docker-compose.yml healthcheck means by it. `docker exec` propagates the exit status, which tells
"the probe reported failures" from "the probe died". The runner image ships `python3`, the app
image `python`. The outer side uses only the standard library.

Models and check (e)
--------------------
The image carries no models (production mounts `/app/models`). First boot copies
templates/default_config.json, which selects `tts: piper` with
`voice_path: models/ru_RU-ruslan-medium.onnx`. `PiperTtsBackend.__init__` loads it eagerly
(src/plugins/tts/piper.py) and src/app.py builds the TTS backend before `DeviceManager.start()`, so a
bare container logs the startup line and the config creation, then dies with
`FileNotFoundError: ... 'models/ru_RU-ruslan-medium.onnx.json'`. /health and the panel never answer.

So check (e) uses two containers:

  * (e1) `-bareboot`: the image's own command, no volumes, no config, `--network none`. Proves the
    startup marker, the template copy, data/config.json on disk (fetched with `docker cp`), and no
    unexpected traceback. The boot must end either alive and serving /health (the TTS load became
    lazy) or in exactly the missing-model failure, judged on the last traceback of the settled log
    and a non-zero exit code. Any other death fails. A survivor that does not answer /health is
    re-inspected, since a slow ordinary death looks the same at the deadline.

  * (e2) `-serve`: same image and CMD, no volumes, with data/config.json injected before start
    (`docker create` -> `docker cp` -> `docker start`). The config is the image's own template with
    the TTS provider switched to a cloud one (fake api_key, never used) and the stress stage
    switched off (the template's RuAccent would download ~360 MB from HuggingFace). The app comes
    up fully and /health and the panel are checked. Both switches are confirmed from the log: the
    TTS backend line and the `ruaccent/off` line.

If the Piper load becomes lazy, (e1) reports a serving container and (e2)'s TTS switch becomes
removable; the stress switch stays (it is about the network).

Checks:

* (a) the image's declared contract: CMD, WORKDIR, no ENTRYPOINT. Outer half.
* (b) the version really got stamped at build time — not the `0.0.0+unknown` fallback — and it is
      the value the workflow passed. Image probe, plus an end-to-end confirmation from the panel's
      own /api/system in the HTTP probe.
* (c) the native VAD runtime loads: libc++.so.1, libc++abi.so.1, and the bundled libten_vad.so
      located through the ten_vad package itself. Image probe.
* (d) the frontend really was built into the image, assets and all. Image probe.
* (e) the application boots with no volumes — (e1) and (e2) above. Outer half + HTTP probe.
* (f) the admin panel answers. HTTP probe.
* (g) the serving container is still alive at the end, and the log it has by THEN — not the
      snapshot the boot wait returned — carries no traceback and shows the stress stage really came
      up disabled. Outer half.

No credentials, and nothing reaches outside the runner, so the gate can run on pull requests.

Invariants:
* Failures leave through SystemExit, never `assert` (asserts vanish under PYTHONOPTIMIZE=1).
* Every check runs before the run is judged. A check that cannot run reports FAILED, never skipped.
  Probe report lines are parsed back into this gate's report.
"""

import json
import os
import subprocess
import tempfile
import time

# The tag to test. Required: a default could gate some other image that happens to be on the daemon.
IMAGE_ENV = "SMOKE_IMAGE"
# Base name for every container this gate starts; the workflows include the run id so concurrent
# runs on the shared daemon do not collide. Required, like the tag.
NAME_ENV = "SMOKE_NAME"
# The VERSION passed to `docker build`. Required for check (b): the image's stamp is compared with
# it, so the `0.0.0+unknown` fallback cannot pass.
VERSION_ENV = "SMOKE_VERSION"

# The four containers this gate starts, by suffix on $SMOKE_NAME:
#   -template   a one-shot `--rm` run that cats the image's config template back out.
#   -probe      the long-lived one the image probe runs inside, started with a sleeping command
#               rather than the image's own (the real command cannot come up without models).
#   -bareboot   the image's REAL command with no volumes and no config: check (e1).
#   -serve      the image's REAL command with an injected model-free config: checks (e2) and (f).
# Named so they can be removed after a subprocess timeout, which kills the docker client but not the
# container. Kept in step with the suffix list in both workflows' cleanup steps.
TEMPLATE_SUFFIX = "-template"
PROBE_SUFFIX = "-probe"
BAREBOOT_SUFFIX = "-bareboot"
SERVE_SUFFIX = "-serve"
ALL_SUFFIXES = (TEMPLATE_SUFFIX, PROBE_SUFFIX, BAREBOOT_SUFFIX, SERVE_SUFFIX)

# The Dockerfile's WORKDIR and CMD. Hardcoded on purpose: they are the contract with the image, and a
# Dockerfile that moves them must go red here.
APP_DIR = "/app"
EXPECTED_CMD = ["python", "main.py"]
CONFIG_PATH = "/app/data/config.json"
# Read out of the image with `cat`, not from the checkout. src/app.py opens this relative path on
# first boot, so a broken `COPY templates/` must go red here.
TEMPLATE_PATH = "templates/default_config.json"

# The ports the two servers bind inside the container. 8200 comes from the template's
# core.audio.port, 8201 from PANEL_PORT's default in src/app.py. Both are what docker-compose.yml
# publishes, and 8200 is what its healthcheck probes — so these are the numbers production depends
# on, checked here on the same loopback the healthcheck uses.
AUDIO_PORT = 8200
PANEL_PORT = 8201

# main.py's first log line, before asyncio.run(app_main()): proves imports and logging setup
# succeeded, nothing more.
STARTUP_MARKER = "Starting zakhar-voice-assistant"
# src/app.py's receipt for the first-boot template copy.
CONFIG_CREATED_MARKER = "created default config at data/config.json"
# The TTS backend the app built; in the serving container it proves the injected config was read.
TTS_BACKEND_MARKER = "tts backend: {}".format
# The known way a bare container stops: the eager Piper load with no models directory. Both
# fragments are required (the exception type alone would match a missing template or prompt), and
# they are matched in the last traceback of the settled log only (see bareboot_stop_reason).
BAREBOOT_EXPECTED_FRAGMENTS = ("FileNotFoundError", "models/ru_RU-ruslan-medium.onnx")
# Declared once; three verdicts search for it.
TRACEBACK_MARKER = "Traceback (most recent call last)"

# The TTS provider the serving container is switched to, and the key its constructor demands. The
# cloud providers (yandex, fishaudio) raise ValueError on an empty api_key in __init__ but make no
# request there, so a fake key is enough to construct one — and nothing in this gate ever calls it.
# Chosen over piper/silero because those two are exactly the ones that need a model file on disk.
SMOKE_TTS_PROVIDER = "yandex"
SMOKE_TTS_KEY = "smoke-not-a-real-key"

# The stress provider the serving container's config turns off. The template enables `ruaccent`,
# which src/app.py loads right after the panel starts and which pulls ~360 MB from HuggingFace;
# `enabled: false` makes RuAccentProvider.create return a PassthroughAccentizer
# (src/plugins/stress/ruaccent.py). A failed load would not go red (src/reconfig.py logs and carries
# on), so it is switched off rather than left to fail.
SMOKE_STRESS_PROVIDER = "ruaccent"
# Receipt for that switch, checked by (g): the config model ignores unknown keys, so a renamed field
# would make the switch a silent no-op. src/reconfig.py logs `<provider>/off` when the stage is
# disabled and `<provider>/<model>` otherwise.
STRESS_DISABLED_MARKER = "hot-reloaded stress backend -> {}/off".format

# The image-probe container runs this instead of the image's own command, which dies without models.
# 900 s is far beyond the gate's worst case; the container is removed in a `finally` regardless.
IDLE_COMMAND = ["python", "-c", "import time; time.sleep(900)"]

# TZ matches the deployment (docker-compose.yml sets exactly this and nothing else). It is the only
# environment variable production supplies, so supplying it here keeps the container this gate
# judges and the container that runs in production configured the same way.
CONTAINER_ENV = ["TZ=Europe/Moscow"]
# The name under which both probes read the expected version from their container's environment.
# `docker exec` runs with the environment the container was CREATED with, so this is passed to
# `docker run`/`docker create` and inherited by the probe rather than interpolated into its text.
PROBE_VERSION_ENV = "SMOKE_EXPECTED_VERSION"

# Printed by each probe on its last line. Declared again inside the probe text, kept as plain
# strings without `.format()` (their bodies contain braces) so they can be pasted into a container by
# hand; the two copies must agree. A drift causes a false red, never a false green.
IMAGE_PROBE_MARKER = "zakhar image probe ok"
HTTP_PROBE_MARKER = "zakhar http probe ok"
# The same convention for the bare container's liveness probe.
BAREBOOT_HEALTH_MARKER = "zakhar bareboot health ok"

# How many verdicts each probe must produce. Pinned because a probe whose checks silently stopped
# emitting rows would still print its marker and exit 0. Adding a check means updating the number.
EXPECTED_IMAGE_PROBE_TARGETS = 23
EXPECTED_HTTP_PROBE_TARGETS = 5

# The same pin for the outer half, checked in main() before the summary.
# Full run: 3 (image contract) + 1 (template read) + 1 (probe container start) + 23 + 2 (image probe
# plus run_probe's two consistency rows) + 6 (bare boot) + 3 + 2 (serving start and boot) + 5 + 2
# (http probe) + 1 (serving still alive) + 2 (check_serving_final_log).
EXPECTED_TOTAL_TARGETS = 51
# When the template cannot be read, the serving block (3 + 2 + 5 + 2 + 1 + 2 = 15 rows) is replaced
# by two "not attempted" rows. Other failure paths also emit fewer rows; main() only checks the count
# when nothing else failed.
EXPECTED_TEMPLATELESS_TARGETS = EXPECTED_TOTAL_TARGETS - 15 + 2

# The prefix put in front of every row a probe reported, so the merged report says where each
# verdict was decided. Without it a reader of thirty result lines cannot tell which of them were
# taken from inside a container and which from the runner.
IMAGE_PROBE_ROW_PREFIX = "[in-image] "
HTTP_PROBE_ROW_PREFIX = "[in-container] "

# Every docker call is bounded: a hung gate holds the shared runner until the step timeout, and a
# killed step skips its cleanup. Worst-case sum, in the order main() makes the calls (the smoke
# step's `timeout-minutes` must exceed it):
#     30 (inspect the image config)                                                  =  30
#   + 30 (rm template)  + 60 (run --rm: cat the template)                            =  90
#   + 30 (rm probe)     + 60 (run -d: the idle container)                            =  90
#   + 120 (docker exec: the image probe)                                             = 120
#   + 30 (rm bareboot)  + 60 (run -d)   + 90 (marker poll: 60 budget + one 30 s logs)
#                       + 60 (exit poll: 30 budget + one 30 s inspect)
#                       + 30 (re-read the log once it is final)
#                       + 30 (docker exec: the health probe, only reached by a survivor)
#                       + 30 (re-inspect after that probe fails: judge_bare_outcome)
#                       + 30 (re-read the log when that inspect says it has since stopped)
#                       + 30 (docker cp the config out)                                = 390
#   + 30 (rm serve)     + 60 (create)   + 30 (cp the config in) + 60 (start)         = 180
#   + 90 (marker poll on the serving container: 60 budget + one 30 s logs)           =  90
#   + 180 (docker exec: the http probe)                                              = 180
#   + 30 (inspect the serving container)
#   + 45 (settled-log poll: 15 budget + one 30 s logs)                               =  75
#   + 120 (the `finally`: 4 x 30)                                                    = 120
#   = 1365 s, 22 minutes 45 s. Both workflows allow 24.
# Each poll is bounded by its budget in wall clock plus one in-flight docker call. Every container
# is removed by name before it starts, so a UI re-run (same run id) cannot hit "name already in use".
INSPECT_TIMEOUT = 30
REMOVE_TIMEOUT = 30
TEMPLATE_TIMEOUT = 60
START_TIMEOUT = 60
CREATE_TIMEOUT = 60
COPY_TIMEOUT = 30
IMAGE_PROBE_TIMEOUT = 120
# Larger than the image probe's because this probe polls for both servers; its internal
# HTTP_PROBE_BUDGET has to fit inside it.
HTTP_PROBE_TIMEOUT = 180
LOGS_TIMEOUT = 30
# The bare container's liveness probe, wrapping BAREBOOT_HEALTH_BUDGET; only spent on the survivor
# path.
BAREBOOT_HEALTH_TIMEOUT = 30

# The startup-marker poll, bounded in wall clock (each `docker logs` has its own 30 s bound). The
# marker arrives within a second or two; 60 s is the margin for a loaded runner.
BOOT_BUDGET = 60
BOOT_PAUSE = 0.5

# How long the BARE container gets to reach a terminal state after its startup line. The Piper
# failure comes a second or two after the marker (the onnxruntime import first); 30 s is about
# fifteen times that. A container still alive at the deadline may be the hoped-for survivor, a
# deadlock, a retry loop or a slow death; the health probe and judge_bare_outcome() tell them apart.
BAREBOOT_EXIT_BUDGET = 30
# The bare survivor's /health budget inside the probe. Short: by then a working app has been
# listening for tens of seconds. Duplicated as `BUDGET` in the probe text (no `.format()` over a body
# full of braces); move both together.
BAREBOOT_HEALTH_BUDGET = 15

# How long the SERVING container's log is polled for the stress stage's receipt: the deferred load
# is requested only after the panel serves, so the line may land after the HTTP probe.
STRESS_SETTLE_BUDGET = 15

# How much of an unexpected output reaches the log. Container logs can run to thousands of lines
# when something loops, and an unbounded dump would bury the verdict.
EXCERPT_CHARS = 4000


# The program fed to `python -u -` inside the IDLE container. Kept as a string so the gate stays one
# file. `-u` because stdout is a pipe: a probe killed by IMAGE_PROBE_TIMEOUT would otherwise lose its
# buffered output. It takes no parameters; `docker exec` inherits the container's creation
# environment, including SMOKE_EXPECTED_VERSION.
IMAGE_PROBE = r'''
"""Runs INSIDE a container of the zakhar-voice-assistant image, fed to `python -u -` over stdin.

It is HERMETIC: not one line of it opens a socket to anything outside the container. What it
therefore checks is everything about this image that can be decided without a peer — and, for this
image, that is most of what can go wrong:

  * the version stamp, which is the one build-time input that fails SILENTLY. A CI that forgot
    `fetch-depth: 0` or the `--build-arg` produces an image that builds, starts, serves and reports
    `0.0.0+unknown` forever;
  * the native VAD runtime. `libten_vad.so` links against LLVM libc++, which python:3.11-slim does
    not ship; the Dockerfile installs libc++1/libc++abi1 for exactly this. A base-image bump or a
    lost apt layer turns TEN VAD into a crash at CDLL the moment somebody selects it in the panel —
    long after the deploy, with nothing in CI having gone red;
  * the frontend. The two-stage build is the classic place for a stage that "succeeded" and
    produced nothing: the image still builds, the container still starts, and the panel is a blank
    page. Only looking at the files answers it.

Everything is checked before the run is judged, and a check that cannot run reports itself as
FAILED rather than being skipped. Every check function returns a FIXED number of rows whatever
happens inside it, so the outer half's exact-count assertion means what it says. Failures leave
through SystemExit, never `assert`.
"""

import ctypes
import json
import os
import re

# Printed on the last line and nowhere else. This is the outer half's IMAGE_PROBE_MARKER, written
# out again because this text is handed to another interpreter and cannot import anything from the
# file it lives in. If you move one, move the other.
PROBE_MARKER = "zakhar image probe ok"

# The WORKDIR the image declares, checked here as well as in the outer half's `docker inspect` — and
# they are not the same statement. The outer half reads what the image DECLARES; this reads where a
# process actually LANDS, which is what makes every relative path in this app resolve: "data",
# "templates/default_config.json", "frontend/dist" and "models/..." are all relative strings.
APP_DIR = "/app"
DATA_DIR = "/app/data"
CONFIG_PATH = "/app/data/config.json"
MODELS_DIR = "/app/models"
TEMPLATE_PATH = "/app/templates/default_config.json"
FRONTEND_DIR = "/app/frontend/dist"
INDEX_PATH = "/app/frontend/dist/index.html"
STAMP_PATH = "/app/src/_version.txt"

# The fallback src/version.py falls back to when neither git nor the stamp file answers. The whole
# point of check (b) is that this string must NOT be what the image reports.
FALLBACK_VERSION = "0.0.0+unknown"
# Set on the container by the outer half; the value the workflow passed to --build-arg VERSION.
EXPECTED_VERSION_ENV = "SMOKE_EXPECTED_VERSION"

# The two LLVM runtime libraries the Dockerfile installs. Loaded by SONAME exactly the way
# ten_vad's own bundled .so pulls them in, so this fails here for the same reason it would fail
# there.
LIBCXX_SONAMES = ["libc++.so.1", "libc++abi.so.1"]

# The modules imported to prove the wheels really installed and the sources really got copied.
# src.app is the composition root and pulls in nearly everything (aiohttp, soxr, mcp,
# httpx, pydantic, the whole src.plugins registry); the other four are the ones with a direct
# production consequence — the two servers, the config store and the version resolver. Note that
# importing src.plugins does NOT import piper/vosk/ten_vad: those are deliberately lazy, imported
# inside their backends' constructors, which is why (c) has to load the native library by hand.
SRC_MODULES = ["src.version", "src.config_store", "src.device_server", "src.panel_api", "src.app"]


def describe(error):
    return "{}: {}".format(type(error).__name__, error)


def check_working_directory():
    """A process in this container must land in /app, which is what makes relative paths work."""
    target = "the container's working directory is {}".format(APP_DIR)
    cwd = os.getcwd()
    if cwd == APP_DIR:
        return [(target, None)]
    return [(target, (
        "it is {!r}. Every path this app uses is RELATIVE — 'data' for the config and the SQLite "
        "databases, 'templates/default_config.json' for the first-boot copy, 'frontend/dist' for "
        "the panel, 'models/...' for the voices — so a process that starts anywhere else writes "
        "state outside the mounted volume and reads none of its own files".format(cwd)))]


def check_version():
    """(b) The version was really stamped at build time, and it is the one the build passed.

    This is the check that exists because the failure is otherwise SILENT. The Dockerfile's
    `RUN test -n "$VERSION" && echo "$VERSION" > src/_version.txt` only refuses an EMPTY arg; a CI
    that omitted `--build-arg VERSION` entirely gets the ARG's own default, `0.0.0+unknown`, writes
    that into the stamp, and produces an image that builds, starts, serves and misreports its
    version forever. The same happens on a checkout without full history, where `git describe`
    cannot answer.

    Four rows, and each is a different statement: the file is there and non-empty; it is not the
    fallback; it equals what the build passed; and src.version resolves to that same value — which
    is the number the panel shows and the only one an operator ever sees.
    """
    rows = []
    stamp_target = "the build stamp {} exists and is non-empty".format(STAMP_PATH)
    fallback_target = "...and is not the {!r} fallback".format(FALLBACK_VERSION)
    expected = os.environ.get(EXPECTED_VERSION_ENV) or ""
    match_target = "...and equals the version the build passed ({!r})".format(expected)
    resolve_target = "...and src.version.__version__ resolves to that same value"

    try:
        stamped = open(STAMP_PATH, encoding="utf-8").read().strip()
    except Exception as error:
        reason = "reading it raised {}".format(describe(error))
        return [(stamp_target, reason), (fallback_target, "not attempted: " + reason),
                (match_target, "not attempted: " + reason),
                (resolve_target, "not attempted: " + reason)]

    if not stamped:
        rows.append((stamp_target, (
            "it is empty. The Dockerfile's `test -n \"$VERSION\"` is supposed to make that "
            "impossible, so an empty stamp means the stamping step itself changed")))
    else:
        rows.append((stamp_target, None))

    if stamped == FALLBACK_VERSION:
        rows.append((fallback_target, (
            "it is exactly the fallback. That is what ARG VERSION defaults to, so this image was "
            "built with no --build-arg VERSION at all — or from a checkout with no history for "
            "`git describe` to read. It runs perfectly and lies about which commit it is, which is "
            "the one thing an operator needs when a deploy goes wrong")))
    else:
        rows.append((fallback_target, None))

    if not expected:
        rows.append((match_target, (
            "not attempted: {} is not set on this container, so there is nothing to compare "
            "against. The outer half is supposed to refuse to start without it".format(
                EXPECTED_VERSION_ENV))))
    elif stamped != expected:
        rows.append((match_target, (
            "the image says {!r}. The build and this gate disagree about what was built, which "
            "means the tag being smoke-tested is not the image this run produced".format(stamped))))
    else:
        rows.append((match_target, None))

    try:
        from src.version import __version__ as resolved
    except Exception as error:
        rows.append((resolve_target, "importing src.version raised {}".format(describe(error))))
        return rows

    if resolved != stamped:
        rows.append((resolve_target, (
            "it resolves to {!r} while the stamp says {!r}. src/version.py prefers `git describe` "
            "over the stamp, so the usual cause is a .git directory that got into the image past "
            ".dockerignore — which makes the reported version depend on the build host rather than "
            "on the build".format(resolved, stamped))))
    else:
        rows.append((resolve_target, None))
    return rows


def check_native_libs():
    """(c) The native VAD runtime loads: LLVM libc++, and ten_vad's own bundled shared object.

    python:3.11-slim ships no libc++, and ten_vad's wheel bundles a .so that links against it. The
    Dockerfile installs libc++1/libc++abi1 for exactly that reason, and the failure it prevents is
    invisible until somebody switches the VAD provider to TEN in the panel — at which point
    `create()` raises at CDLL, in production, with nothing in CI having gone red.

    The .so is located THROUGH THE PACKAGE rather than by a hardcoded path: the wheel lays its
    binaries out per platform (lib/Linux/x64/libten_vad.so today) and a hardcoded path would keep
    passing by finding nothing to disagree with the moment that layout changed.

    Five rows: two SONAMEs, then locate/load/construct.
    """
    rows = []
    for soname in LIBCXX_SONAMES:
        target = "ctypes.CDLL({!r}) succeeds".format(soname)
        try:
            ctypes.CDLL(soname)
        except Exception as error:
            rows.append((target, (
                "it raised {}. python:3.11-slim does not ship this library; the Dockerfile's "
                "`apt-get install libc++1 libc++abi1` is what puts it there. Losing that layer — a "
                "base-image bump, a reordered Dockerfile, an apt mirror that failed quietly — turns "
                "the TEN VAD provider into a crash the moment it is selected".format(
                    describe(error)))))
        else:
            rows.append((target, None))

    locate_target = "ten_vad's bundled libten_vad.so is present in the installed package"
    load_target = "...and ctypes.CDLL loads it"
    build_target = "...and TenVad() constructs against it"

    try:
        import ten_vad
    except Exception as error:
        reason = "importing ten_vad raised {}".format(describe(error))
        rows.append((locate_target, reason))
        rows.append((load_target, "not attempted: " + reason))
        rows.append((build_target, "not attempted: " + reason))
        return rows

    package_dir = os.path.dirname(os.path.abspath(ten_vad.__file__))
    found = []
    for root, _dirs, files in os.walk(package_dir):
        for name in files:
            if name == "libten_vad.so":
                found.append(os.path.join(root, name))
    if not found:
        reason = (
            "nothing named libten_vad.so under {!r}. The wheel is supposed to bundle it (the repo "
            "never ships it), so either the wheel installed without its binaries or its layout "
            "changed and this search has to follow".format(package_dir))
        rows.append((locate_target, reason))
        rows.append((load_target, "not attempted: " + reason))
        rows.append((build_target, "not attempted: " + reason))
        return rows
    library = sorted(found)[0]
    rows.append(("{} ({})".format(locate_target, library), None))

    try:
        ctypes.CDLL(library)
    except Exception as error:
        rows.append((load_target, (
            "it raised {}. This is the exact failure the libc++ rows above exist to predict: the "
            "file is there and the loader cannot satisfy what it links against".format(
                describe(error)))))
    else:
        rows.append((load_target, None))

    try:
        from ten_vad import TenVad

        TenVad(hop_size=256, threshold=0.5)
    except Exception as error:
        rows.append((build_target, (
            "it raised {}. Loading the library and initialising an engine are different things, and "
            "src/plugins/vad/ten.py does the second one inside create() — so this is what the "
            "panel would surface when somebody selects the TEN provider".format(describe(error)))))
    else:
        rows.append((build_target, None))
    return rows


def check_frontend():
    """(d) The admin panel really was built into the image, with the assets it references.

    The Dockerfile builds the panel in a separate node stage and copies only `dist` across. That is
    the classic place for a stage that runs, exits 0 and produces nothing useful: the image still
    builds, the container still starts, `src/app.py` still finds the directory and serves it, and
    the panel is a blank page in a browser. Nothing fails anywhere.

    So it is not enough to look at index.html. Its asset references are extracted and each one is
    resolved on disk, and the count is asserted to be non-zero — otherwise an index.html that
    referenced nothing at all would sail through the "every referenced asset exists" row by having
    none to check, which is the vacuous pass this check exists to avoid.
    """
    rows = []
    index_target = "{} exists and is non-empty".format(INDEX_PATH)
    refs_target = "...and references at least one local .js asset"
    assets_target = "...and every asset it references exists on disk and is non-empty"

    try:
        size = os.path.getsize(INDEX_PATH)
    except Exception as error:
        reason = "stat raised {}".format(describe(error))
        return [(index_target, reason), (refs_target, "not attempted: " + reason),
                (assets_target, "not attempted: " + reason)]
    if size == 0:
        reason = "it is 0 bytes: the frontend stage produced an empty index.html"
        return [(index_target, reason), (refs_target, "not attempted: " + reason),
                (assets_target, "not attempted: " + reason)]
    rows.append(("{} ({} bytes)".format(index_target, size), None))

    try:
        html = open(INDEX_PATH, encoding="utf-8").read()
    except Exception as error:
        reason = "reading it raised {}".format(describe(error))
        rows.append((refs_target, reason))
        rows.append((assets_target, "not attempted: " + reason))
        return rows

    # Local references only. The favicon is an inline data: URI and any CDN reference would be an
    # absolute URL; neither is a file this image is supposed to contain.
    references = []
    for value in re.findall(r'(?:src|href)="([^"]+)"', html):
        if value.startswith("data:") or "://" in value:
            continue
        references.append(value)

    scripts = [value for value in references if value.split("?")[0].endswith(".js")]
    if not scripts:
        rows.append((refs_target, (
            "it references no local .js file at all. Vite always emits a hashed entry bundle and "
            "links it from index.html, so an index.html without one is a build that produced a "
            "shell — the panel would load and render nothing. References found: {!r}".format(
                references))))
    else:
        rows.append(("{} ({})".format(refs_target, ", ".join(scripts)), None))

    broken = []
    for value in references:
        relative = value.split("?")[0].split("#")[0].lstrip("/")
        path = os.path.join(FRONTEND_DIR, relative)
        try:
            if os.path.getsize(path) == 0:
                broken.append("{} (0 bytes)".format(path))
        except Exception:
            broken.append("{} (missing)".format(path))
    if broken:
        rows.append((assets_target, (
            "these are not usable: {}. index.html was copied into the image but the files it points "
            "at were not, so every request for them 404s and the panel stays blank".format(
                ", ".join(broken)))))
    else:
        rows.append(("{} ({} checked)".format(assets_target, len(references)), None))
    return rows


def check_template():
    """The first-boot config template is in the image and is valid JSON.

    src/app.py opens this exact relative path the first time it starts against an empty data volume
    — i.e. on every NEW deployment and never on an existing one, which is what makes a missing or
    malformed template the kind of fault that is only ever discovered in production.
    """
    target = "{} is in the image and parses as JSON".format(TEMPLATE_PATH)
    try:
        document = json.load(open(TEMPLATE_PATH, encoding="utf-8"))
    except Exception as error:
        return [(target, (
            "it raised {}. src/app.py reads this file on first boot; without it a container started "
            "against a fresh volume dies before it configures anything".format(describe(error))))]
    if not isinstance(document, dict) or "core" not in document or "tts" not in document:
        return [(target, (
            "it parses but does not look like a config document (top-level keys: {!r}). "
            "ConfigDoc(**doc) is what consumes it".format(sorted(document) if isinstance(
                document, dict) else type(document).__name__)))]
    return [(target, None)]


def check_data_dir():
    """The state directory exists and is writable.

    It is the mount point of the production data volume, and the config save path is atomic —
    tempfile + fsync + os.replace in the SAME directory — so a read-only or missing directory means
    the first boot cannot write config.json at all.
    """
    rows = []
    exists_target = "{} exists and is a directory".format(DATA_DIR)
    write_target = "...and is writable, so the first boot can create config.json"

    if not os.path.isdir(DATA_DIR):
        rows.append((exists_target, (
            "it is not there (or is not a directory). The Dockerfile creates it with `RUN mkdir -p "
            "data` and production mounts a volume over it")))
        rows.append((write_target, "not attempted: the directory is not there"))
        return rows
    rows.append((exists_target, None))

    probe_path = os.path.join(DATA_DIR, ".smoke-write-probe")
    try:
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("smoke")
        os.unlink(probe_path)
    except Exception as error:
        rows.append((write_target, (
            "writing a file into it raised {}. config_store.save() writes a temp file into this "
            "directory and os.replace()s it into place, so nothing about the config can be "
            "persisted".format(describe(error)))))
        return rows
    rows.append((write_target, None))
    return rows


def check_no_baked_state():
    """Neither a config nor a model set may be baked into the image.

    A baked data/config.json would be read on first boot INSTEAD of the template, so a new
    deployment would silently inherit whatever configuration — and whatever api keys — happened to
    be on the machine that built the image. .dockerignore lists `data/` for exactly this, and this
    row is what keeps a Dockerfile edit from undoing it without anyone noticing.

    A baked models directory would be the other half of the same mistake: hundreds of megabytes in
    the image, and a container that comes up happily on a host whose models volume was never
    mounted — hiding the misconfiguration until the volume is needed for something else.
    """
    rows = []
    config_target = "no data/config.json is baked into the image"
    models_target = "no models are baked into the image (they are a volume)"

    if os.path.exists(CONFIG_PATH):
        rows.append((config_target, (
            "{!r} exists in the image. src/app.py only copies the template when the config is "
            "ABSENT, so every fresh deployment would boot on this file instead — including any "
            "api keys it carries".format(CONFIG_PATH))))
    else:
        rows.append((config_target, None))

    baked = []
    if os.path.isdir(MODELS_DIR):
        for root, _dirs, files in os.walk(MODELS_DIR):
            for name in files:
                baked.append(os.path.join(root, name))
    if baked:
        rows.append((models_target, (
            "{} file(s) under {}, e.g. {}. Production mounts these as a volume; baking them in "
            "makes the image enormous and lets a container come up on a host where the volume was "
            "never mounted".format(len(baked), MODELS_DIR, sorted(baked)[0]))))
    else:
        rows.append((models_target, None))
    return rows


def check_imports():
    """Every module that matters imports inside the image.

    An import here is a statement about the WHEELS, not about this repo's syntax: src.app pulls in
    aiohttp, soxr, httpx, pydantic, mcp and the whole src.plugins registry, so a
    requirements.txt whose transitive tree failed to resolve on this base image fails right here
    instead of at the first boot on the deployment.
    """
    rows = []
    for name in SRC_MODULES:
        target = "import {}".format(name)
        try:
            __import__(name)
        except Exception as error:
            rows.append((target, (
                "it raised {}. Either the module is missing from the image (the Dockerfile copies "
                "src/ wholesale, so that would be a copy that changed) or one of the wheels it "
                "imports is".format(describe(error)))))
            continue
        rows.append((target, None))
    return rows


def run_check(name, function):
    """Run one check, converting an unexpected exception into a row instead of a dead probe.

    Without this a single unforeseen error would take the whole probe down and cost the run every
    other verdict, in exactly the situation where the rest of the report is what tells somebody
    which change did it.

    The check is passed as a FUNCTION and called in here, not called at the call site and passed as
    a result: calling it in the argument list would run its body BEFORE this function is entered, so
    an exception from it would escape past the `except` below and take the probe down anyway.
    """
    try:
        return function()
    except Exception as error:
        return [("the {} check".format(name), "it raised {}".format(describe(error)))]


def main():
    rows = []

    def stage(name, function):
        produced = run_check(name, function)
        # Printed as each group lands rather than all at once at the end, so a probe that dies
        # half-way still leaves behind everything it had already decided.
        for target, reason in produced:
            if reason is None:
                print("ok   {}".format(target))
            else:
                print("FAIL {} -> {}".format(target, reason))
        rows.extend(produced)

    stage("working directory", check_working_directory)
    stage("version stamp", check_version)
    stage("native VAD libraries", check_native_libs)
    stage("frontend bundle", check_frontend)
    stage("config template", check_template)
    stage("data directory", check_data_dir)
    stage("baked state", check_no_baked_state)
    stage("imports", check_imports)

    failures = [target for target, reason in rows if reason is not None]
    if failures:
        print("probe FAILED: {}/{} targets broken".format(len(failures), len(rows)))
        raise SystemExit(1)

    print("{}: {}/{} targets".format(PROBE_MARKER, len(rows), len(rows)))


if __name__ == "__main__":
    main()
'''


# The program fed to `python -u -` inside the SERVING container. It talks to the app's two servers
# over the container's own loopback, the way docker-compose.yml's healthcheck does (urllib, since
# python:3.11-slim has no curl).
HTTP_PROBE = r'''
"""Runs INSIDE the SERVING container, fed to `python -u -` over stdin.

Answers the two questions that only a running application can answer: is the device server serving
/health, and is the admin panel answering. Both are polled rather than sampled once — the container
has just been started and the panel comes up after the device server — and both are reached over the
container's own loopback with nothing but the standard library, which is precisely what
docker-compose.yml's healthcheck does.

The third question it answers is a cheap one worth having: the version the PANEL reports. The image
probe proves the stamp file is right; this proves the value actually reaches the surface an operator
looks at, through src/version.py and PanelServer(version=...). Those are different failures.

Every check runs before the run is judged, a check that cannot run reports itself as FAILED, and
failures leave through SystemExit rather than `assert`.
"""

import json
import os
import time
import urllib.error
import urllib.request

# Printed on the last line and nowhere else. This is the outer half's HTTP_PROBE_MARKER, written out
# again because this text is handed to another interpreter. If you move one, move the other.
PROBE_MARKER = "zakhar http probe ok"

AUDIO_URL = "http://127.0.0.1:8200/health"
PANEL_SYSTEM_URL = "http://127.0.0.1:8201/api/system"
PANEL_INDEX_URL = "http://127.0.0.1:8201/"

EXPECTED_VERSION_ENV = "SMOKE_EXPECTED_VERSION"

# The poll. The outer half only waited for main.py's first log line, which is printed BEFORE
# asyncio.run() — so at the moment this probe starts, the servers may legitimately not be listening
# yet. 90 s is far beyond the second or two a real boot takes and is sized for a loaded shared
# runner; it has to stay comfortably inside the outer half's HTTP_PROBE_TIMEOUT.
BUDGET = 90
PAUSE = 0.5
# Per-attempt bound, so one hung connect cannot eat the whole budget in a single attempt.
ATTEMPT_TIMEOUT = 5


def describe(error):
    return "{}: {}".format(type(error).__name__, error)


def fetch(url, deadline):
    """Poll one URL until it answers 200 or the deadline passes.

    Returns (status, body, error). A connection refused is the EXPECTED answer early on — the
    server is still binding — so it is retried rather than reported; only the last one survives into
    the verdict.
    """
    last = None
    while True:
        try:
            with urllib.request.urlopen(url, timeout=ATTEMPT_TIMEOUT) as response:
                return response.status, response.read().decode("utf-8", "replace"), None
        except urllib.error.HTTPError as error:
            # An HTTP answer, just not the one wanted: the server IS up, so stop here rather than
            # spending the whole budget re-asking a question that has been answered.
            body = ""
            try:
                body = error.read().decode("utf-8", "replace")
            except Exception:
                pass
            return error.code, body, None
        except Exception as error:
            last = error
        if time.monotonic() >= deadline:
            return None, "", last
        time.sleep(PAUSE)


def main():
    rows = []
    deadline = time.monotonic() + BUDGET
    expected_version = os.environ.get(EXPECTED_VERSION_ENV) or ""

    health_target = "GET {} answers 200 within {} s".format(AUDIO_URL, BUDGET)
    body_target = "...and its body is the device server's own 'ok'"
    status, body, error = fetch(AUDIO_URL, deadline)
    if status == 200:
        rows.append((health_target, None))
        if body.strip() == "ok":
            rows.append((body_target, None))
        else:
            rows.append((body_target, (
                "it answered 200 with {!r}. DeviceManager._handle_health returns the literal 'ok', so "
                "something else is answering on this port".format(body[:200]))))
    else:
        reason = (
            "it answered {!r} (last error: {}). This is the endpoint docker-compose.yml's "
            "healthcheck probes, so an image that fails here is one Docker will report as unhealthy "
            "forever — and, because src/app.py starts the device server only after every backend has "
            "been constructed, the usual cause is a backend that could not be built".format(
                status, describe(error) if error is not None else "none"))
        rows.append((health_target, reason))
        rows.append((body_target, "not attempted: " + reason))

    system_target = "GET {} answers 200".format(PANEL_SYSTEM_URL)
    version_target = "...and the panel reports the version the build stamped ({!r})".format(
        expected_version)
    status, body, error = fetch(PANEL_SYSTEM_URL, deadline)
    if status == 200:
        rows.append((system_target, None))
        try:
            payload = json.loads(body)
        except Exception as exc:
            rows.append((version_target, "the response is not JSON ({}): {!r}".format(
                describe(exc), body[:200])))
        else:
            reported = payload.get("version")
            if not expected_version:
                rows.append((version_target, (
                    "not attempted: {} is not set on this container".format(
                        EXPECTED_VERSION_ENV))))
            elif reported != expected_version:
                rows.append((version_target, (
                    "it reports {!r}. This is the value the admin panel shows and the only version "
                    "an operator ever sees; a mismatch means the number in the UI does not identify "
                    "the running commit".format(reported))))
            else:
                rows.append((version_target, None))
    else:
        reason = (
            "it answered {!r} (last error: {}). /api/system is registered unconditionally in "
            "PanelServer.build_app(), so it is the one panel route that cannot be absent — a "
            "failure here means the panel is not serving at all".format(
                status, describe(error) if error is not None else "none"))
        rows.append((system_target, reason))
        rows.append((version_target, "not attempted: " + reason))

    index_target = "GET {} serves the built single-page app".format(PANEL_INDEX_URL)
    status, body, error = fetch(PANEL_INDEX_URL, deadline)
    if status == 200 and "<div id=\"root\">" in body:
        rows.append((index_target, None))
    else:
        rows.append((index_target, (
            "it answered {!r} and the body does not carry the SPA mount point (last error: {}). "
            "src/app.py passes static_dir='frontend/dist' to PanelServer ONLY when that directory "
            "exists, and silently runs API-only when it does not — so this is what a frontend stage "
            "that produced nothing looks like from the outside: every API call works and the panel "
            "is a blank page".format(status, describe(error) if error is not None else "none"))))

    failures = [target for target, reason in rows if reason is not None]
    for target, reason in rows:
        if reason is None:
            print("ok   {}".format(target))
        else:
            print("FAIL {} -> {}".format(target, reason))

    if failures:
        print("probe FAILED: {}/{} targets broken".format(len(failures), len(rows)))
        raise SystemExit(1)

    print("{}: {}/{} targets".format(PROBE_MARKER, len(rows), len(rows)))


if __name__ == "__main__":
    main()
'''


# The program fed to `python -u -` inside the BARE container, only when it is still alive after
# BAREBOOT_EXIT_BUDGET. An answer on /health proves the eager backend construction got past (the
# device server starts after it); a live process alone could be deadlocked or retrying. Separate from
# the HTTP probe, which needs a version in the environment and checks the panel.
BAREBOOT_HEALTH_PROBE = r'''
"""Runs INSIDE the BARE container when it has outlived the gate's exit budget.

One question: is the device server answering /health? src/app.py starts it only after every eager
backend has been built, so an answer here means the boot got through the load that this container
is not supposed to be able to complete — and that is the only evidence that turns "still running"
into "the eager model load became lazy".
"""

import time
import urllib.error
import urllib.request

# The outer half's BAREBOOT_HEALTH_MARKER, written out again because this text is handed to another
# interpreter and cannot import from the file it lives in. If you move one, move the other.
PROBE_MARKER = "zakhar bareboot health ok"

# The port docker-compose.yml's healthcheck probes, reached on the container's own loopback.
URL = "http://127.0.0.1:8200/health"
# The outer half's BAREBOOT_HEALTH_BUDGET, written out again for the same reason PROBE_MARKER is:
# this text is handed to another interpreter and cannot import from the file it lives in. The outer
# half's timeout arithmetic is written against that name and this loop is bounded by this number, so
# if you move one, move the other.
BUDGET = 15
PAUSE = 0.5
ATTEMPT_TIMEOUT = 5


def main():
    deadline = time.monotonic() + BUDGET
    last = None
    while True:
        try:
            with urllib.request.urlopen(URL, timeout=ATTEMPT_TIMEOUT) as response:
                status, body = response.status, response.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as error:
            # An HTTP answer, just not the one wanted. Something IS listening and speaking HTTP, so
            # retrying is pointless — but it is not a pass either: the verdict below still requires
            # 200 and a body of "ok", and /health has exactly one handler, so anything else means
            # the thing answering on this port is not the device server this check is about. Breaking
            # here is what turns that into a reported status instead of a spent budget.
            status, body = error.code, ""
            break
        except Exception as error:
            last = error
        if time.monotonic() >= deadline:
            status, body = None, ""
            break
        time.sleep(PAUSE)

    if status == 200 and body.strip() == "ok":
        print("{}: {} answered 200 {!r}".format(PROBE_MARKER, URL, body.strip()))
        return

    print("{} answered {!r} with body {!r} within {} s (last error: {})".format(
        URL, status, body[:200], BUDGET,
        "none" if last is None else "{}: {}".format(type(last).__name__, last)))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def excerpt(text):
    """Bound what reaches the log, and say so when something was cut."""
    if text is None:
        return ""
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS] + "\n[... truncated at {} characters]".format(EXCERPT_CHARS)


def docker(args, timeout, stdin_text=None):
    """Run a docker command.

    Returns (status, output) with stderr folded into stdout, because everything here is read by a
    human out of a CI log where the interleaving is the useful part. A status of None means the
    command produced no exit code at all — it timed out, or docker is not there — and `output` then
    explains which.
    """
    argv = ["docker"] + args
    try:
        completed = subprocess.run(
            argv,
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True)
    except FileNotFoundError:
        return None, (
            "`docker` is not on PATH. This gate drives the daemon from the runner, so it cannot run "
            "anywhere the docker CLI is missing")
    except subprocess.TimeoutExpired as error:
        return None, "`{}` did not finish within {} s. Output so far:\n{}".format(
            " ".join(argv), timeout, excerpt(error.output))
    return completed.returncode, completed.stdout or ""


def remove_container(name):
    """Best effort. Never the reason a check fails; the workflow cleans up too."""
    docker(["rm", "-f", name], REMOVE_TIMEOUT)


def environment_flags(pairs):
    """A list of `VAR=value` strings as `-e VAR=value` pairs for a `docker run` argument list."""
    flags = []
    for pair in pairs:
        flags.extend(["-e", pair])
    return flags


def probe_report_rows(output, prefix):
    """Parse a probe's own report lines back into rows of this gate's report.

    Both probes print exactly the same `ok   <target>` / `FAIL <target> -> <reason>` shape this file
    does, so their verdicts can be merged into one list instead of arriving as a single opaque "the
    probe failed". That matters for the same reason every other check here has one row per claim: a
    run that breaks four things should say four things.
    """
    rows = []
    for line in output.splitlines():
        if line.startswith("ok   "):
            rows.append((prefix + line[5:], None))
        elif line.startswith("FAIL "):
            target, _, reason = line[5:].partition(" -> ")
            rows.append((prefix + target, reason or "the probe reported FAIL with no reason"))
    return rows


def wait_for_marker(name, marker, budget):
    """Poll `docker logs` until `marker` appears or the budget runs out.

    Bounded in wall clock, since each `docker logs` has its own 30 s bound. Works on an exited
    container too.
    """
    deadline = time.monotonic() + budget
    status, logs = None, ""
    while True:
        status, logs = docker(["logs", name], LOGS_TIMEOUT)
        if status == 0 and marker in logs:
            return status, logs
        if time.monotonic() >= deadline:
            return status, logs
        time.sleep(BOOT_PAUSE)


def container_state(name):
    """`Running ExitCode` for a container, or None when it cannot be read."""
    status, output = docker(
        ["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name], INSPECT_TIMEOUT)
    if status != 0:
        return None, output
    return output.strip(), output


def wait_for_exit(name, budget):
    """Poll container_state() until the container is no longer running, or the budget runs out.

    Returns (state, output, still_running). `still_running` is True only when the container was
    still up at the deadline. Bounded in wall clock, like wait_for_marker.
    """
    deadline = time.monotonic() + budget
    while True:
        state, output = container_state(name)
        if state is None:
            # Could not be read at all; the caller reports that rather than guessing an outcome.
            return None, output, False
        if not state.startswith("true"):
            return state, output, False
        if time.monotonic() >= deadline:
            return state, output, True
        time.sleep(BOOT_PAUSE)


def check_image_contract(image):
    """(a) What the image DECLARES: its command, where it runs, and that nothing wraps them."""
    rows = []
    target = "docker inspect {}".format(image)
    status, output = docker(["inspect", "--format", "{{json .Config}}", image], INSPECT_TIMEOUT)
    if status is None:
        # No exit code at all: the docker CLI is missing or the call ran out its timeout. The
        # message from docker() already says which, and neither means what a non-zero exit means.
        return [(target, output)]
    if status != 0:
        return [(target, (
            "it exited {} — the tag does not exist on this daemon, so the build step and this step "
            "disagree about what was built. Output:\n{}".format(status, excerpt(output))))]
    try:
        config = json.loads(output)
    except ValueError as error:
        return [("parse the image config of {}".format(image), (
            "docker inspect returned something that is not JSON ({}): {}".format(
                error, excerpt(output))))]

    cmd = config.get("Cmd")
    rows.append((
        "the image's CMD is {}".format(EXPECTED_CMD),
        None if cmd == EXPECTED_CMD else (
            "it is {!r}. Production runs the image's own command and so do the two boot checks "
            "below — a silent change here means the gate and the deployment are describing two "
            "different programs".format(cmd))))

    # An ENTRYPOINT would prepend itself to the CMD above, so `docker run <image>` would no longer be
    # `python main.py` and every boot verdict below would be about something else entirely.
    entrypoint = config.get("Entrypoint")
    rows.append((
        "the image declares no ENTRYPOINT, so its CMD is the whole command line",
        None if not entrypoint else (
            "it declares {!r}, which is prepended to the CMD. Every `docker run` in this gate now "
            "executes something other than what it reports".format(entrypoint))))

    working_dir = config.get("WorkingDir")
    rows.append((
        "the image's WORKDIR is {}".format(APP_DIR),
        None if working_dir == APP_DIR else (
            "it is {!r}, and that is not cosmetic. Every path this app uses is relative — 'data', "
            "'templates/default_config.json', 'frontend/dist', 'models/...' — so a moved working "
            "directory writes state outside the mounted volume and finds none of its own "
            "files".format(working_dir))))
    return rows


def read_template(image, name):
    """Read templates/default_config.json back out of the IMAGE, not the checkout: check (e2) boots
    on the image's own template.

    Returns (rows, document or None).
    """
    target = "the image's {} can be read and parsed".format(TEMPLATE_PATH)
    remove_container(name)
    status, output = docker(
        ["run", "--rm", "--name", name, image, "cat", TEMPLATE_PATH], TEMPLATE_TIMEOUT)
    if status is None:
        return [(target, output)], None
    if status != 0:
        return [(target, "`docker run ... cat {}` exited {}:\n{}".format(
            TEMPLATE_PATH, status, excerpt(output)))], None
    try:
        document = json.loads(output)
    except ValueError as error:
        return [(target, "the file is in the image but is not valid JSON ({}): {}".format(
            error, excerpt(output)))], None
    return [(target, None)], document


def selected_stress_provider(template):
    """Which stress provider the image's template selects, falling back to the one this file was
    written against.

    Shared by the switch that disables the stage and the verdict that reads its receipt, which must
    name the same provider.
    """
    stress = template.get("stress") if isinstance(template, dict) else None
    if isinstance(stress, dict) and stress.get("selected"):
        return stress["selected"]
    return SMOKE_STRESS_PROVIDER


def build_serving_config(template):
    """The template with its TTS provider switched to one that needs no model file, and its stress
    stage switched off.

    The only deviation from production's configuration:
    * TTS: the image has no models and Piper loads its voice eagerly before the device server
      starts. The fake api_key only satisfies the cloud provider's constructor.
    * Stress: the template enables RuAccent, which src/app.py loads in the background once the
      panel is up (a ~360 MB HuggingFace download; a failure is swallowed by src/reconfig.py).
      `enabled: false` on the selected instance makes RuAccentProvider.create return a
      PassthroughAccentizer.
    """
    document = json.loads(json.dumps(template))  # deep copy: the caller keeps the pristine template
    tts = document.setdefault("tts", {})
    tts["selected"] = SMOKE_TTS_PROVIDER
    tts.setdefault("instances", {})[SMOKE_TTS_PROVIDER] = {"api_key": SMOKE_TTS_KEY}
    stress = document.setdefault("stress", {})
    selected_stress = selected_stress_provider(template)
    stress.setdefault("instances", {}).setdefault(selected_stress, {})["enabled"] = False
    return document


def start_probe_container(image, name, version):
    """Start the idle container the image probe runs inside.

    NO PORT IS PUBLISHED — see the module docstring. The image's CMD is REPLACED with a sleeping
    command because the real one cannot survive without models; the two boot checks below cover the
    real command in their own containers, which is where that question belongs.
    """
    target = "`docker run -d` starts an idle probe container of {}".format(image)
    remove_container(name)
    status, output = docker(
        ["run", "-d", "--name", name]
        + environment_flags(CONTAINER_ENV + ["{}={}".format(PROBE_VERSION_ENV, version)])
        + [image] + IDLE_COMMAND,
        START_TIMEOUT)
    if status is None:
        return [(target, output)], False, output
    if status != 0:
        return [(target, "`docker run -d` exited {}:\n{}".format(status, excerpt(output)))], \
            False, output
    return [(target, None)], True, output


def run_probe(name, started, program, timeout, marker, expected_targets, prefix, label):
    """Feed one probe to `docker exec -i <name> python -u -` and merge its verdicts into ours.

    `-i` attaches stdin without asking for a tty (none is needed and none is available on a runner),
    the image's own interpreter runs the program, and `docker exec` propagates the command's exit
    status — which is what lets the two consistency rows below tell "the probe reported failures"
    apart from "the probe died before it could report anything".
    """
    exit_target = "the {} probe's exit status agrees with its own report".format(label)
    marker_target = "the {} probe ran to its end, reporting all {} targets".format(
        label, expected_targets)
    if not started:
        return [("the {} probe".format(label),
                 "not attempted: its container never started")], ""

    status, output = docker(["exec", "-i", name, "python", "-u", "-"], timeout, stdin_text=program)
    if status is None:
        # Timed out, or docker is missing. Either way there are no verdicts to merge.
        return [("the {} probe".format(label), output)], output

    rows = probe_report_rows(output, prefix)
    if not rows:
        # Exit status alone cannot be read here: a probe that printed nothing has told us nothing,
        # whatever it exited with. This is what a `docker exec` that could not start the interpreter
        # looks like, and what a truncated stdin looks like.
        return [("the {} probe".format(label), (
            "it produced no report lines at all, so none of its checks ran. `docker exec` exited "
            "{}. Output:\n{}".format(status, excerpt(output))))], output

    reported_failures = any(reason is not None for _, reason in rows)
    if status != 0 and not reported_failures:
        rows.append((exit_target, (
            "`docker exec` exited {} but every line the probe printed says ok — so it died after "
            "reporting and before finishing, and the report above is incomplete".format(status))))
    elif status == 0 and reported_failures:
        rows.append((exit_target, (
            "the probe printed FAIL lines yet exited 0. Its failures are supposed to leave through "
            "SystemExit(1); an exit of 0 here means they no longer do, and this gate would have "
            "gone green on them")))
    else:
        rows.append((exit_target, None))

    # Exact, not a substring of the marker alone: see the note on the EXPECTED_*_TARGETS constants.
    expected_line = "{}: {}/{} targets".format(marker, expected_targets, expected_targets)
    if expected_line in output:
        rows.append((marker_target, None))
    elif marker not in output:
        rows.append((marker_target, (
            "it never printed {!r}. The marker is on the probe's last line, so its absence means "
            "the program did not run to the end — a truncated stdin, or something that killed the "
            "interpreter mid-report".format(marker))))
    else:
        # It finished, but with a different number of verdicts than this gate expects. That is a
        # gate that has quietly started proving less, which is the failure mode nothing else here
        # can see: every row it DID print says ok, and the exit status is 0.
        summary = [line for line in output.splitlines() if line.startswith(marker)]
        rows.append((marker_target, (
            "it ran to the end but reported {!r} instead of {} targets. Either a check stopped "
            "emitting rows — in which case this gate is now proving less than it says it does and "
            "nothing else would have noticed — or one was added and the EXPECTED_*_TARGETS constant "
            "in this file needs updating".format(
                summary[0] if summary else "(unparseable)", expected_targets))))
    return rows, output


def bareboot_serves(name):
    """Ask a SURVIVING bare container whether its device server actually answers.

    Returns (serving, output). Runs over `docker exec` because the container's loopback is not the
    runner's. Both the marker and a zero exit are required.
    """
    status, output = docker(["exec", "-i", name, "python", "-u", "-"], BAREBOOT_HEALTH_TIMEOUT,
                            stdin_text=BAREBOOT_HEALTH_PROBE)
    return status == 0 and BAREBOOT_HEALTH_MARKER in output, output


def bareboot_stop_reason(state, logs):
    """Why a STOPPED bare container's stop is NOT the one known stop. Returns (reason, exit_code).

    `reason` is None exactly when the stop is the expected one; `exit_code` is returned either way
    because the accepting row names it.

    Three things must hold:
    * there is a traceback at all (otherwise `rsplit` would return the whole log, and a tidy
      `logger.error` plus non-zero exit would match both fragments);
    * the expected fragments are in the LAST traceback, so a logged-and-caught Piper error cannot
      mask a later death;
    * the exit code is non-zero (an exit of 0 means the exception was caught).
    """
    # `Running ExitCode`, e.g. "false 1". Parsed as a fixed pair so an unparseable state does not
    # become an empty string that passes every `!= "0"` test.
    fields = state.split()
    exit_code = fields[1] if len(fields) == 2 else ""

    if TRACEBACK_MARKER not in logs:
        return ("docker reports `Running ExitCode` = {!r} and its log carries NO traceback at all, "
                "so the known Piper stop cannot be what ended it: that stop is an UNCAUGHT "
                "FileNotFoundError, and CPython prints a traceback for those. Something else ended "
                "this boot — a caught error re-reported as a log line and a bare SystemExit, a "
                "signal, an exec that never got that far — and the expected fragments appearing "
                "somewhere in the log would say nothing about which. Full logs:\n{}".format(
                    state, excerpt(logs))), exit_code

    last_traceback = logs.rsplit(TRACEBACK_MARKER, 1)[-1]
    missing = [f for f in BAREBOOT_EXPECTED_FRAGMENTS if f not in last_traceback]
    if missing:
        mismatch = ("its LAST traceback does not carry {} (the whole log may well mention them "
                    "higher up; what ends the process is what counts)".format(
                        " and ".join(repr(f) for f in missing)))
    elif exit_code == "":
        mismatch = ("its last traceback IS the known one, but `docker inspect` did not answer with "
                    "the two fields this gate reads an exit code out of, so whether that exception "
                    "actually ended the process is unknown — and unknown is not a pass")
    elif exit_code == "0":
        mismatch = ("its last traceback IS the known one but the container exited 0, so something "
                    "caught that exception and returned — this stop is not the one the traceback "
                    "looks like")
    else:
        return None, exit_code

    return ("docker reports `Running ExitCode` = {!r} and {}. This image is KNOWN to stop here on "
            "the eager Piper voice load and nothing else; anything else means something new is "
            "broken and is being hidden behind an expected failure. Last traceback:\n{}\n"
            "Full logs:\n{}".format(
                state, mismatch, excerpt(last_traceback), excerpt(logs))), exit_code


def judge_bare_outcome(name, outcome_target, state, state_output, still_running, logs):
    """(e1)'s last verdict: the boot either came up and SERVES, or stopped in the ONE known place.

    Returns (row, still_running, logs): asking the question can change how the boot ended, and the
    other rows in check_bare_boot are written about the final state.

    * Stopped: judged by bareboot_stop_reason() on the settled log.
    * Alive and answering /health: the eager Piper load became lazy; accepted.
    * Alive and not answering: a slow ordinary death (still inside `svc.create("tts")`, before the
      device server starts) looks the same as a deadlock, and the container may have died during the
      probe. So state and log are re-read, and a container that has since stopped is judged as a
      stop.
    """
    if state is None:
        return (outcome_target, "docker inspect failed: {}".format(excerpt(state_output))), \
            False, logs

    if not still_running:
        reason, exit_code = bareboot_stop_reason(state, logs)
        if reason is None:
            return ("{} (stopped on the missing Piper voice, exit {}, as expected)".format(
                outcome_target, exit_code), None), False, logs
        return (outcome_target, reason), False, logs

    serving, probe_output = bareboot_serves(name)
    if serving:
        return ("{} (alive {} s after its startup line AND answering /health: the eager model load "
                "is gone)".format(outcome_target, BAREBOOT_EXIT_BUDGET), None), True, logs

    # The probe said no, which is the answer both remaining cases give. Re-read what the container
    # IS now, then re-read its log if it has stopped — the traceback that decides the verdict is
    # written in the last moments before the exit, so the log taken at the budget predates it.
    state, state_output = container_state(name)
    if state is not None and not state.startswith("true"):
        log_status, settled_logs = docker(["logs", name], LOGS_TIMEOUT)
        if log_status != 0:
            return (outcome_target, (
                "it had already stopped by the time /health was asked — it was still running at "
                "the {} s exit budget and reached its terminal state ({!r}) while the probe ran — "
                "but its log could not be re-read to see WHY: {}. The log taken at the budget is "
                "older than the exit, so it cannot answer that question either. Probe "
                "output:\n{}".format(BAREBOOT_EXIT_BUDGET, state, excerpt(settled_logs),
                                     excerpt(probe_output)))), False, logs
        logs = settled_logs
        reason, exit_code = bareboot_stop_reason(state, logs)
        if reason is None:
            # The ordinary slow death: too slow for the exit budget, dead before the probe gave up.
            # An expected stop, and it says so — no deadlock is claimed, because none was observed.
            return ("{} (stopped on the missing Piper voice, exit {}, as expected — it was still "
                    "inside that load at the {} s exit budget and reached the stop while /health "
                    "was being asked)".format(outcome_target, exit_code, BAREBOOT_EXIT_BUDGET),
                    None), False, logs
        return (outcome_target, reason), False, logs

    # Still up on the second look (or unreadable on it), and not serving. NOW the diagnosis below
    # is earned: the known stop kills the process, and this process is still there.
    second_look = ("it is still running on a second `docker inspect`, taken after the probe gave up"
                   if state is not None else
                   "a second `docker inspect` could not be read at all ({}), so this verdict rests "
                   "on the state at the exit budget".format(excerpt(state_output)))
    return (outcome_target, (
        "the container is still up {} s after its startup line, but its device server never "
        "answered /health from inside it. Alive is not the same as up: src/app.py builds every "
        "eager backend BEFORE it starts that server, so a container that is running and not serving "
        "is one still inside that construction. It is not the known Piper stop running late either "
        "— {} — so what is left is a construction that is deadlocked, retrying, or waiting on a "
        "network timeout, and it is neither of the two endings this branch exists to accept. Probe "
        "output:\n{}\nLogs:\n{}".format(
            BAREBOOT_EXIT_BUDGET, second_look, excerpt(probe_output), excerpt(logs)))), True, logs


def check_bare_boot(image, name):
    """(e1) The image's own command, no volumes, no config: production's genuine first boot.

    See the module docstring: this image cannot come up without the models volume, so this check
    pins down exactly where it stops.

    Started detached and without `--rm`, so `docker cp` and `docker inspect` still work after it
    exits.

    Order: wait for the startup line, wait for a terminal state, re-read the log, decide the
    outcome, then write the rows, so every row describes a boot that is over. A container still
    alive at the deadline is asked whether it serves (bareboot_serves(), judge_bare_outcome())
    rather than taken as "came up".
    """
    rows = []
    start_target = "`docker run -d` starts {} with the image's own command and no volumes".format(
        image)
    marker_target = "...its log carries {!r} (bounded at {} s)".format(STARTUP_MARKER, BOOT_BUDGET)
    created_target = "...and {!r}: the first-boot template copy fired".format(CONFIG_CREATED_MARKER)
    config_target = "...and {} really exists in the container and parses as JSON".format(
        CONFIG_PATH)
    # Deliberately not "nothing before the startup line": which part of the log has to be
    # traceback-free depends on how the boot ended, and the row says which window it used.
    trace_target = "...and no traceback appears where one would mean a NEW failure"
    outcome_target = "...and the boot either comes up and SERVES or stops in the ONE known place"
    every_target = (marker_target, created_target, config_target, trace_target, outcome_target)

    remove_container(name)
    # `--network none`, only for this container: it boots on the image's own template, which enables
    # RuAccent (a ~360 MB HuggingFace download requested once the panel is up). Today the Piper crash
    # comes first, but if the boot ever survives, the empty namespace still keeps the gate offline.
    # Nothing here needs the network: the boot path opens no socket, `logs`/`inspect`/`cp`/`exec`
    # talk to the daemon, and `lo` still exists for the /health probe.
    status, output = docker(
        ["run", "-d", "--name", name, "--network", "none"]
        + environment_flags(CONTAINER_ENV) + [image], START_TIMEOUT)
    if status is None:
        rows.append((start_target, output))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be started"))
        return rows, ""
    if status != 0:
        rows.append((start_target, "`docker run -d` exited {}:\n{}".format(
            status, excerpt(output))))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be started"))
        return rows, ""
    rows.append((start_target, None))

    log_status, logs = wait_for_marker(name, STARTUP_MARKER, BOOT_BUDGET)
    if log_status is None:
        # No exit code at all: the poll's last `docker logs` timed out, or the CLI is gone.
        # docker()'s own message already says which, and neither is what a non-zero exit means —
        # "`docker logs` exited None" describes nothing that happened.
        for target in every_target:
            rows.append((target, logs))
        return rows, logs
    if log_status != 0:
        for target in every_target:
            rows.append((target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs))))
        return rows, logs

    # Nothing below is decided on the log wait_for_marker returned: the marker is main.py's first
    # line, and the Piper failure comes a second or two later (after the onnxruntime import). Wait
    # for a terminal state, re-read the log, and decide from that. Reaching the deadline only says
    # the container is alive; the outcome row then asks whether it serves.
    state, state_output, still_running = wait_for_exit(name, BAREBOOT_EXIT_BUDGET)
    log_status, logs = docker(["logs", name], LOGS_TIMEOUT)
    # The daemon answered `docker logs` for this container moments ago, so a failure here is an
    # anomaly rather than a slow start. Reported instead of quietly falling back to the stale
    # snapshot: a verdict taken from that snapshot is the exact thing this block exists to stop.
    # `status is None` is kept apart from a non-zero exit for the reason it is everywhere else in
    # this file — the two are different events and only one of them has an exit code to report.
    settled_note = ("the container reached its startup line, but re-reading its log once it had "
                    "settled failed: {}. Every verdict below is about the FINAL log, so none of "
                    "them can be decided from the partial one taken at the marker")
    if log_status is None:
        for target in every_target:
            rows.append((target, settled_note.format(logs)))
        return rows, logs
    if log_status != 0:
        for target in every_target:
            rows.append((target, settled_note.format(
                "`docker logs` exited {}: {}".format(log_status, excerpt(logs)))))
        return rows, logs

    # The outcome is decided before the rows are written: judge_bare_outcome() may spend the health
    # probe's budget and re-read state and log, and every row below must describe the boot as it
    # finally ended.
    outcome_row, still_running, logs = judge_bare_outcome(
        name, outcome_target, state, state_output, still_running, logs)

    reached = STARTUP_MARKER in logs
    rows.append((marker_target, None if reached else (
        "it never appeared in {} s. That line is printed by main.py before asyncio.run(), so its "
        "absence means the interpreter did not even get through the imports and the logging setup. "
        "Logs:\n{}".format(BOOT_BUDGET, excerpt(logs)))))

    rows.append((created_target, None if CONFIG_CREATED_MARKER in logs else (
        "the log does not carry it. src/app.py copies templates/default_config.json into "
        "data/config.json only when the config is absent, which is exactly this container's state, "
        "so the copy should have fired and said so. Logs:\n{}".format(excerpt(logs)))))

    # `docker cp` works against a stopped container, which is the only reason this row can exist:
    # the container this reads from is expected to be dead by now. Reading the file back is a
    # stronger statement than the log line above — the log says the app THINKS it wrote a config,
    # this says one is really there and is really a config.
    handle, local_path = tempfile.mkstemp(prefix="smoke-config-", suffix=".json")
    os.close(handle)
    copy_status, copy_output = docker(["cp", "{}:{}".format(name, CONFIG_PATH), local_path],
                                      COPY_TIMEOUT)
    if copy_status is None:
        # No exit code at all: timed out, or docker is gone. docker()'s own message says which, and
        # neither is "the file is not there" — which is what the branch below would claim.
        rows.append((config_target, copy_output))
    elif copy_status != 0:
        rows.append((config_target, (
            "`docker cp` exited {}: {}. On a fresh container this file is the ONLY thing the first "
            "boot is supposed to produce, so its absence means the deployment's very first start "
            "against an empty volume leaves nothing behind".format(
                copy_status, excerpt(copy_output)))))
    else:
        try:
            with open(local_path, encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception as error:
            rows.append((config_target, (
                "it exists but could not be parsed: {}. config_store.save() writes it atomically, "
                "so a truncated or invalid document here means the write path itself is "
                "broken".format(error))))
        else:
            if not isinstance(document, dict) or "core" not in document:
                rows.append((config_target, (
                    "it parses but is not a config document (top-level keys: {!r})".format(
                        sorted(document) if isinstance(document, dict) else
                        type(document).__name__))))
            else:
                rows.append((config_target, None))
    try:
        os.unlink(local_path)
    except OSError:
        pass

    if not reached:
        rows.append((trace_target, "not attempted: the startup line never appeared, so there is "
                                   "nothing to place a traceback before or after"))
    else:
        # A stopped boot is expected to end in the known Piper traceback (judged by the outcome row),
        # so only the part before the startup line is examined here, where an ImportError while
        # loading src.app lands. A surviving boot has no expected traceback, so its whole log is
        # examined: a background task failing after the marker would otherwise pass unnoticed.
        window = logs if still_running else logs.split(STARTUP_MARKER, 1)[0]
        scope = ("its whole log, since a boot that survived has no expected traceback anywhere"
                 if still_running else "everything before the startup line")
        scoped_target = "{} ({})".format(trace_target, scope)
        rows.append((scoped_target, None if TRACEBACK_MARKER not in window else (
            "there is one. {} Traceback in:\n{}".format(
                "This container is still running, so nothing in its log is supposed to have died."
                if still_running else
                "Everything before that line is import and logging setup, so a traceback there is "
                "the image failing before the application begins.",
                excerpt(window)))))

    rows.append(outcome_row)
    return rows, logs


def start_serving_container(image, name, version, config):
    """(e2) Start the image's own command, no volumes, with a model-free config injected.

    `docker create` -> `docker cp` -> `docker start` instead of a bind mount: the job runs inside
    act_runner's container while `docker` drives an outside daemon, so a `-v $(pwd)/...` path would
    not exist on the daemon's side. No port is published and no volume is attached.
    """
    rows = []
    create_target = "`docker create` prepares a serving container of {} with no volumes".format(
        image)
    copy_target = "...and the model-free config is copied into {}".format(CONFIG_PATH)
    start_target = "...and `docker start` starts it"
    every_target = (copy_target, start_target)

    remove_container(name)
    status, output = docker(
        ["create", "--name", name]
        + environment_flags(CONTAINER_ENV + ["{}={}".format(PROBE_VERSION_ENV, version)])
        + [image],
        CREATE_TIMEOUT)
    # `status is None` means there was no exit code at all — the call timed out or docker is
    # missing — and docker()'s message already says which. Kept apart from a non-zero exit the way
    # read_template() and check_bare_boot() keep them apart: folded together, the report reads
    # "`docker create` exited None", which describes nothing that happened.
    if status is None:
        rows.append((create_target, output))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be created"))
        return rows, False
    if status != 0:
        rows.append((create_target, "`docker create` exited {}:\n{}".format(
            status, excerpt(output))))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be created"))
        return rows, False
    rows.append((create_target, None))

    handle, local_path = tempfile.mkstemp(prefix="smoke-serving-config-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    status, output = docker(["cp", local_path, "{}:{}".format(name, CONFIG_PATH)], COPY_TIMEOUT)
    try:
        os.unlink(local_path)
    except OSError:
        pass
    if status is None:
        rows.append((copy_target, output))
        rows.append((start_target, "not attempted: the config was not copied in"))
        return rows, False
    if status != 0:
        rows.append((copy_target, "`docker cp` exited {}:\n{}".format(status, excerpt(output))))
        rows.append((start_target, "not attempted: the config was not copied in"))
        return rows, False
    rows.append((copy_target, None))

    status, output = docker(["start", name], START_TIMEOUT)
    if status is None:
        rows.append((start_target, output))
        return rows, False
    if status != 0:
        rows.append((start_target, "`docker start` exited {}:\n{}".format(
            status, excerpt(output))))
        return rows, False
    rows.append((start_target, None))
    return rows, True


def check_serving_boot(name, started):
    """(e2) The serving container reaches its startup line, on the config it was given.

    Returns (rows, logs). The HTTP probe does the serving checks afterwards.

    Waits for the TTS backend line rather than the startup line: src/app.py builds the ConfigDoc and
    the backends before logging it, so the config verdict needs that later line.

    Tracebacks are not judged here: the returned log is a snapshot taken when the TTS line appeared,
    before the servers and deferred loads start. check_serving_final_log() judges them.
    """
    tts_marker = TTS_BACKEND_MARKER(SMOKE_TTS_PROVIDER)
    # The bound belongs to the line this actually POLLS for, and saying otherwise would be a
    # promise about the wrong line: if the startup line appears and the TTS one never does, the
    # poll spends the whole budget and the startup row then passes carrying a bound it did not use.
    marker_target = "the serving container logs {!r}".format(STARTUP_MARKER)
    config_target = ("...and {!r}: the injected config was really the one it booted on (the log is "
                     "polled for THIS line, bounded at {} s)".format(tts_marker, BOOT_BUDGET))
    every_target = (marker_target, config_target)

    if not started:
        return [(target, "not attempted: the container never started") for target in every_target], \
            ""

    log_status, logs = wait_for_marker(name, tts_marker, BOOT_BUDGET)
    if log_status is None:
        # No exit code at all: the poll's last `docker logs` timed out, or the CLI is gone. Kept
        # apart from a non-zero exit as everywhere else here — docker()'s message says which, and
        # "`docker logs` exited None" would describe neither.
        return [(target, logs) for target in every_target], logs
    if log_status != 0:
        return [(target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs)))
                for target in every_target], logs

    rows = []
    rows.append((marker_target, None if STARTUP_MARKER in logs else (
        "it never appeared while waiting {} s for {!r}. Logs:\n{}".format(
            BOOT_BUDGET, tts_marker, excerpt(logs)))))

    # Without this row the whole of (e2) could pass on a container that had quietly booted on the
    # TEMPLATE — i.e. died on Piper — and the HTTP probe would then simply report a dead server,
    # blaming the application for the gate's own setup. This says which config was consumed.
    rows.append((config_target, None if tts_marker in logs else (
        "the log does not name it. src/app.py logs the backend it built for every stage, so its "
        "absence means the injected data/config.json was not read: either the copy landed "
        "somewhere else or something rewrote it. Logs:\n{}".format(excerpt(logs)))))
    return rows, logs


def check_serving_final_log(name, started, stress_provider):
    """(g) The two verdicts that can only be taken once the serving container has settled.

    Returns (rows, logs). Reads the log after the boot wait, the HTTP probe and the liveness check:
    the device server's start, the panel, the MCP endpoint and the deferred stress/wakeword loads all run
    after the TTS line, and a traceback in a background asyncio task kills neither the process nor
    the probe, so only this log read sees it.

    The log is polled for the stress receipt because the deferred load is queued only after the
    panel starts serving and may land after the HTTP probe finished.
    """
    trace_target = "...and the log it has by THEN carries no traceback"
    stress_target = "...and the stress stage really came up disabled ({!r})".format(
        STRESS_DISABLED_MARKER(stress_provider))
    every_target = (trace_target, stress_target)

    if not started:
        return [(target, "not attempted: the container never started") for target in every_target], \
            ""

    stress_marker = STRESS_DISABLED_MARKER(stress_provider)
    log_status, logs = wait_for_marker(name, stress_marker, STRESS_SETTLE_BUDGET)
    if log_status is None:
        return [(target, logs) for target in every_target], logs
    if log_status != 0:
        return [(target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs)))
                for target in every_target], logs

    rows = []
    # The whole log, unlike the bare container's: this one is given everything it needs, so every
    # stage of its boot is supposed to succeed and a traceback anywhere in it is a finding. The
    # deferred loads that fail for want of model files report themselves with logger.error and no
    # traceback (src/reconfig.py catches them per-category), which is what keeps this row from
    # flapping on a runner that has none.
    rows.append((trace_target, None if TRACEBACK_MARKER not in logs else (
        "there is one. This container is given everything it needs to come up, so a traceback in "
        "its log is the image failing at something that has nothing to do with the missing models. "
        "A traceback raised in a background asyncio task does not stop the process, so neither the "
        "HTTP probe nor the liveness row above would have seen it. Logs:\n{}".format(
            excerpt(logs)))))

    # The receipt for the one switch in this gate that is about the NETWORK. build_serving_config
    # writes `enabled: false` into the selected stress instance, and RuAccentConfig ignores unknown
    # keys — so a renamed field, a different provider or an instance on another schema turns that
    # write into a silent no-op: no exception, no red row, and ~360 MB pulled from HuggingFace by a
    # gate whose documented claim is that it reaches nothing outside the runner. src/reconfig.py
    # prints the backend's own describe() when the deferred load lands, and `<provider>/off` is what
    # a stage that really came up disabled says.
    rows.append((stress_target, None if stress_marker in logs else (
        "the log does not carry it within {} s of the checks above. Either the deferred stress load "
        "never ran, or it came up ENABLED — which means the `enabled: false` this gate injects no "
        "longer reaches the provider (RuAccentConfig ignores unknown keys, so a renamed field or a "
        "new stress provider fails silently) and this run pulled the RuAccent model from "
        "HuggingFace. Logs:\n{}".format(STRESS_SETTLE_BUDGET, excerpt(logs)))))
    return rows, logs


def check_container_alive(name, started):
    """(g) The serving container is still running after everything above.

    A container that is gone either died after serving (production would sit in a restart loop) or
    was killed, e.g. by the OOM killer on the shared runner, which also discredits the HTTP probe.
    """
    target = "the serving container is still running after every check above"
    if not started:
        return [(target, "not attempted: the container never started")]
    state, output = container_state(name)
    if state is None:
        return [(target, "docker inspect failed: {}".format(excerpt(output)))]
    return [(target, None if state.startswith("true") else (
        "docker reports `Running ExitCode` = {!r}. It runs `python main.py`, which blocks on an "
        "event that is never set, so it cannot have finished on its own".format(state)))]


def main():
    image = os.environ.get(IMAGE_ENV)
    if not image:
        print("{} is not set: this gate tests the image it is given and has no default, because a "
              "default would silently gate whichever image happened to be on the daemon".format(
                  IMAGE_ENV))
        raise SystemExit(1)
    name = os.environ.get(NAME_ENV)
    if not name:
        # No default: the workflow's cleanup step builds the same names from this variable, so a
        # fallback would leak containers past it, and two concurrent runs on the shared daemon would
        # collide on (and remove) each other's containers.
        print("{} is not set: this gate names every container it starts after it, and a default "
              "would both hide containers from the workflow's cleanup step and make two concurrent "
              "runs collide on one name".format(NAME_ENV))
        raise SystemExit(1)
    version = os.environ.get(VERSION_ENV)
    if not version:
        # Required for the same reason the other two are, but the failure it guards against is
        # subtler: this is the value check (b) compares the image's stamp against. Defaulting it
        # would mean comparing the image to a constant invented here, and skipping the comparison
        # would leave the gate unable to tell a correctly stamped image from one built without the
        # build arg — which is the exact fault check (b) exists for.
        print("{} is not set: check (b) compares the image's build stamp against the version the "
              "build actually passed, and a gate that cannot make that comparison cannot tell a "
              "stamped image from one silently carrying the 0.0.0+unknown fallback".format(
                  VERSION_ENV))
        raise SystemExit(1)

    rows = []
    transcripts = []
    # How many verdicts this run is supposed to end with; checked against the real count below. The
    # only legitimate reason for it to change is the template-less branch, which says so where it
    # takes it.
    expected_targets = EXPECTED_TOTAL_TARGETS

    rows.extend(check_image_contract(image))

    try:
        template_rows, template = read_template(image, name + TEMPLATE_SUFFIX)
        rows.extend(template_rows)

        probe_start_rows, probe_started, probe_start_output = start_probe_container(
            image, name + PROBE_SUFFIX, version)
        transcripts.append(("`docker run -d` of the idle probe container", probe_start_output))
        rows.extend(probe_start_rows)

        image_probe_rows, image_probe_output = run_probe(
            name + PROBE_SUFFIX, probe_started, IMAGE_PROBE, IMAGE_PROBE_TIMEOUT,
            IMAGE_PROBE_MARKER, EXPECTED_IMAGE_PROBE_TARGETS, IMAGE_PROBE_ROW_PREFIX, "image")
        transcripts.append(("the image probe, from inside the container", image_probe_output))
        rows.extend(image_probe_rows)

        bare_rows, bare_logs = check_bare_boot(image, name + BAREBOOT_SUFFIX)
        transcripts.append(("the image started bare: no volumes, no config", bare_logs))
        rows.extend(bare_rows)

        if template is None:
            # Without the template there is no document to inject, and inventing one here would
            # test a configuration this image has never seen. Reported as failures rather than
            # skipped, because the checks below are the ones that prove the service serves at all.
            reason = "not attempted: the image's config template could not be read"
            for target in ("the serving container", "the http probe"):
                rows.append((target, reason))
            # Two rows where the serving block's fourteen would have been.
            expected_targets = EXPECTED_TEMPLATELESS_TARGETS
        else:
            serve_rows, serve_started = start_serving_container(
                image, name + SERVE_SUFFIX, version, build_serving_config(template))
            rows.extend(serve_rows)

            boot_rows, serve_logs = check_serving_boot(name + SERVE_SUFFIX, serve_started)
            rows.extend(boot_rows)

            http_rows, http_output = run_probe(
                name + SERVE_SUFFIX, serve_started, HTTP_PROBE, HTTP_PROBE_TIMEOUT,
                HTTP_PROBE_MARKER, EXPECTED_HTTP_PROBE_TARGETS, HTTP_PROBE_ROW_PREFIX, "http")
            rows.extend(http_rows)

            # After the probe on purpose: "the container is still up" is only worth anything once
            # everything else has had its turn at it.
            rows.extend(check_container_alive(name + SERVE_SUFFIX, serve_started))

            # LAST, and the reason it is last is the reason it exists: everything above has now run
            # against this container, so the log it has at this point is the whole boot rather than
            # the snapshot the boot wait returned. The traceback verdict and the stress receipt are
            # both taken from it — see check_serving_final_log. The transcript shows that same log,
            # so what a reader sees is what the verdicts were decided on.
            final_rows, final_logs = check_serving_final_log(
                name + SERVE_SUFFIX, serve_started, selected_stress_provider(template))
            rows.extend(final_rows)
            transcripts.append(("the serving container", final_logs or serve_logs))
            transcripts.append(("the http probe, from inside the serving container", http_output))
    finally:
        # ALL FOUR, the `--rm` template container included. `--rm` disposes of a container that
        # EXITED, and the case cleanup exists for is the one that did not: when a timeout fires
        # inside docker() it kills the docker CLIENT on the runner, not the container on the daemon,
        # so a hung run survives its own `--rm` and would sit here pinning the image until somebody
        # noticed. The workflow removes the same four again under `if: always()`, which covers the
        # case where this whole script is killed by the step timeout and never reaches this block.
        for suffix in ALL_SUFFIXES:
            remove_container(name + suffix)

    # The transcripts first, the verdicts last: in a CI log the verdicts are what somebody scrolls
    # to the bottom for, and the containers' own output is what turns a one-line verdict into a
    # diagnosis.
    for label, text in transcripts:
        print("")
        print("--- {} ---".format(label))
        print(excerpt(text).rstrip() or "(no output)")
    print("")
    print("--- results ---")

    # The number of verdicts is itself checked: a check_* that silently stopped emitting rows would
    # otherwise leave an all-green report with a smaller N. Only on a run where nothing else failed,
    # because failing checks legitimately emit fewer rows and a count row there would misdirect.
    if len(rows) != expected_targets and not any(reason is not None for _, reason in rows):
        rows.append((
            "this gate produced all {} of the verdicts it is supposed to".format(expected_targets),
            "it produced {}, and every one of them says ok. Either a check stopped emitting rows — "
            "in which case this gate is now proving less than it says it does and nothing else here "
            "would have noticed — or one was added or removed and EXPECTED_TOTAL_TARGETS (or, on "
            "the template-less path, EXPECTED_TEMPLATELESS_TARGETS) needs updating".format(
                len(rows))))

    failures = []
    for target, reason in rows:
        if reason is None:
            print("ok   {}".format(target))
        else:
            print("FAIL {} -> {}".format(target, reason))
            failures.append(target)

    if failures:
        print("")
        print("smoke FAILED: {}/{} targets broken:".format(len(failures), len(rows)))
        for target in failures:
            print("  - {}".format(target))
        raise SystemExit(1)

    print("")
    print("smoke ok: {}/{} targets".format(len(rows), len(rows)))


if __name__ == "__main__":
    main()
