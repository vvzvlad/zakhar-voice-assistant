"""Smoke gate for the zakhar-voice-assistant image, run on the CI runner against an already-built
image.

It is split in three, and the split is the thing to understand before changing anything here.

The OUTER half is this file: a plain `python3 ci/smoke.py` on the runner that drives `docker`
against the tag it is given in $SMOKE_IMAGE. It answers the questions that are about the container
as an object — what `docker inspect` says its CMD and WORKDIR are, what the image writes to its log
when started with its REAL command, whether the first-boot config really lands on disk, whether a
container is still alive when everything else is done. None of those can be answered from a process
already running inside it.

The two INNER halves are PROBES: programs that live here as string constants and are fed to
`docker exec -i <name> python -u -`. The IMAGE PROBE answers everything about the code and the
files INSIDE the image — that the version stamp is real, that the native VAD runtime loads, that
the frontend was actually built into the image, that every module imports. The HTTP PROBE answers
whether the running application is SERVING, from inside the container's own network namespace.

They run inside for two reasons. The obvious one is that those objects only exist in there. The
other is the one that shapes this whole file: **this job and the docker daemon are not in the same
network namespace.** Gitea's act_runner executes the job inside its own job container while the
`docker` CLI it provides drives a daemon that lives outside it, so nothing here may assume it can
reach a container by address.

NO PORT IS PUBLISHED by anything in this file, and that is not a stylistic choice: this service
does listen (8200 for audio, 8201 for the panel), so publishing would look like the obvious way to
check it — and it would silently check nothing. A published port lands in the HOST daemon's network
namespace; this job's loopback is a different loopback, so the gate would be connecting to its own
empty port and would have to be written to tolerate that. Everything internal therefore goes
through `docker exec`, which puts the probe in the container's own namespace, where 127.0.0.1 means
what the healthcheck in docker-compose.yml means by it.

`docker exec -i <name> python -u -` is the mechanism the fleet already uses: `-i` attaches stdin
without asking for a tty, the image's own interpreter runs the program, and `docker exec`
propagates the command's exit status — which is what lets the outer half tell "the probe reported
failures" from "the probe died before it could report anything".

The runner's job image (`docker.gitea.com/runner-images:ubuntu-latest`) ships /usr/bin/python3;
note that it ships `python3` and NOT a bare `python`, which is why both workflows invoke this as
`python3 ci/smoke.py`. Inside the container it is the other way round — the image's own interpreter
is `python` — and this file invokes it accordingly. Nothing beyond the standard library is imported
on the outer side.

THE MODELS PROBLEM, AND WHY CHECK (e) HAS TWO CONTAINERS
--------------------------------------------------------
This is the one thing about this gate that cannot be guessed from the Dockerfile, so it is written
down here rather than left to be rediscovered.

The image deliberately carries NO models: they are hundreds of megabytes and live on a separate
`/app/models` volume in production. The first boot on an empty data volume copies
templates/default_config.json into data/config.json, and that template selects `tts: piper` with
`voice_path: models/ru_RU-ruslan-medium.onnx`.

`PiperTtsBackend.__init__` loads that file EAGERLY (src/plugins/tts/piper.py, `PiperVoice.load`),
and src/app.py calls `svc.create("tts")` at line 92 — BEFORE `audio_server.start()` at line 134.
So an image started with no models at all reaches exactly this far and no further:

    INFO  __main__:main:16 - Starting zakhar-voice-assistant
    INFO  src.app:load_or_create_config:38 - created default config at data/config.json
    Traceback (most recent call last):
      ...
      File "src/app.py", line 92, in main
        tts_backend = svc.create("tts")
      File "src/plugins/tts/piper.py", line 40, in __init__
        voice = PiperVoice.load(voice_path, voice_path + ".json")
    FileNotFoundError: [Errno 2] No such file or directory: 'models/ru_RU-ruslan-medium.onnx.json'

/health therefore never answers in a bare container, and neither does the panel. Mounting an empty
models directory would not change that by one line — the file would still not be there.

The check is split rather than weakened, and neither half is allowed to be vacuous:

  * check (e1), container `-bareboot`: the image's own command, NO volumes, NO config — production's
    genuine first boot. It proves what is really provable in that state: the startup marker is
    logged, the template copy fires, data/config.json really lands on disk (fetched back out with
    `docker cp` and parsed), and nothing before the marker is a traceback. Its last verdict is the
    one that keeps this honest: the boot must end EITHER still running (a future in which the TTS
    load became lazy) OR in exactly the missing-model failure above. Any OTHER way of dying — a
    missing wheel, a broken libc++ layer, an import error — is a FAIL. That is the difference
    between "we know why it stops here" and "it stops here, we stopped looking".

  * check (e2), container `-serve`: the same image, still NO volumes, still the image's own CMD,
    but with data/config.json injected before start (`docker create` -> `docker cp` -> `docker
    start`). The injected document is the IMAGE'S OWN TEMPLATE, read back out of the image, with
    one field changed: the TTS provider is switched to a cloud one that needs no local model file.
    That is the single thing production supplies via a volume and an image can never contain. With
    it the whole application really comes up — audio server, panel, MCP endpoint — and /health and
    the panel are then checked from inside, for real.

If the eager Piper load is ever made lazy, (e1) will report the container still running and (e2)'s
config injection becomes removable. Until then this is the shape of the truth, and it is stated in
both directions so that neither half can quietly become a formality.

Every check is written against a way this image has a realistic chance of shipping broken:

* (a) the image's declared contract: CMD, WORKDIR, no ENTRYPOINT. Outer half.
* (b) the version really got stamped at build time — not the `0.0.0+unknown` fallback — and it is
      the value the workflow passed. Image probe, plus an end-to-end confirmation from the panel's
      own /api/system in the HTTP probe.
* (c) the native VAD runtime loads: libc++.so.1, libc++abi.so.1, and the bundled libten_vad.so
      located through the ten_vad package itself. Image probe.
* (d) the frontend really was built into the image, assets and all. Image probe.
* (e) the application boots with no volumes — (e1) and (e2) above. Outer half + HTTP probe.
* (f) the admin panel answers. HTTP probe.
* (g) the serving container is still alive at the end and its boot log carries no traceback.
      Outer half.

Nothing here holds a credential or reaches an external service. The one key it invents is a fake
TTS api_key, needed only because the cloud provider's constructor refuses an empty one; no request
is ever made with it. A gate that needed a real key could not run on a pull request at all.

Two properties matter and are easy to lose, so they are stated where they can be checked:

* Failures leave through SystemExit, never `assert` — on all three sides of the split. Asserts
  vanish under PYTHONOPTIMIZE=1, which would silently turn this gate permanently green.
* Every check runs before the run is judged, so one run shows the full extent of the breakage
  instead of only the first broken thing. A check that CANNOT run reports itself as FAILED; it is
  never quietly skipped, which is the classic way a gate keeps reporting success while proving less
  and less. Each probe's own report lines are parsed back into this gate's report, so a failure
  inside the container is one row in the same list as a failure outside it.
"""

import json
import os
import subprocess
import tempfile
import time

# The tag to test. Required rather than defaulted: a default would let a mistyped `env:` block in a
# workflow silently gate some other image that happens to be on the daemon.
IMAGE_ENV = "SMOKE_IMAGE"
# Base name for every container this gate starts. The workflows put the run id in it, because the
# runner has a single docker daemon shared by every repository in the fleet and two concurrent runs
# must not collide on a name. Required, exactly like the tag above — see the message in main().
NAME_ENV = "SMOKE_NAME"
# The version string the workflow passed to `docker build --build-arg VERSION=...`. Required too,
# and that is the whole point of check (b): the failure it exists to catch is a build that forgot
# the full git history or the build arg, produced an image stamped with the `0.0.0+unknown`
# fallback, and shipped it looking perfectly healthy. A gate that merely checked "some version is
# present" would pass on the fallback; a gate that defaulted this variable would compare the image
# against a constant this file made up. So it is compared against what the build actually used, and
# an absent value is a hard stop rather than a skipped check.
VERSION_ENV = "SMOKE_VERSION"

# The four containers this gate starts, by suffix on $SMOKE_NAME:
#   -template   a one-shot `--rm` run that cats the image's config template back out.
#   -probe      the long-lived one the image probe runs inside, started with a sleeping command
#               rather than the image's own (the real command cannot come up without models).
#   -bareboot   the image's REAL command with no volumes and no config: check (e1).
#   -serve      the image's REAL command with an injected model-free config: checks (e2) and (f).
# All four are NAMED rather than left to docker's random name generator, and the reason is the one
# case that matters: `subprocess` hitting its timeout kills the docker CLIENT on the runner, not the
# container on the daemon. With no name nobody could ever remove the survivor — not
# remove_container() here, not the workflow's `if: always()` step — and it would go on pinning the
# image, so the `docker rmi` at the end of the job would fail too, be swallowed by its `|| true`,
# and leave both a container and a few hundred MB of image on the daemon the whole fleet shares.
# Kept in step with the suffix list in both workflows' cleanup steps: if you move one, move the
# others.
TEMPLATE_SUFFIX = "-template"
PROBE_SUFFIX = "-probe"
BAREBOOT_SUFFIX = "-bareboot"
SERVE_SUFFIX = "-serve"
ALL_SUFFIXES = (TEMPLATE_SUFFIX, PROBE_SUFFIX, BAREBOOT_SUFFIX, SERVE_SUFFIX)

# The Dockerfile's WORKDIR and CMD, and the paths that resolve against that WORKDIR. Hardcoded
# rather than read back out of the image: these are the contract between this repo and its image, so
# a Dockerfile that quietly moves them has to go red here and be looked at, not be politely
# followed.
APP_DIR = "/app"
EXPECTED_CMD = ["python", "main.py"]
CONFIG_PATH = "/app/data/config.json"
# Read out of the image with `cat`, not from the checkout: the question this gate asks is what is IN
# THE IMAGE. src/app.py opens this exact relative path on first boot, so a Dockerfile whose
# `COPY templates/` slipped produces a container that dies on an empty volume — production's exact
# upgrade path — while the file sits perfectly fine in the repository.
TEMPLATE_PATH = "templates/default_config.json"

# The ports the two servers bind inside the container. 8200 comes from the template's
# core.audio.port, 8201 from PANEL_PORT's default in src/app.py. Both are what docker-compose.yml
# publishes, and 8200 is what its healthcheck probes — so these are the numbers production depends
# on, checked here on the same loopback the healthcheck uses.
AUDIO_PORT = 8200
PANEL_PORT = 8201

# main.py's first log line, before asyncio.run(app_main()). It is a receipt for "the interpreter got
# through every import and the logging setup" and NOTHING more — be precise about that, because the
# temptation is to read it as "the app started". Everything interesting happens after it.
STARTUP_MARKER = "Starting zakhar-voice-assistant"
# src/app.py's receipt for the first-boot template copy. Its absence on a bare container means the
# template was not found or not copied, which is the failure that only ever shows up on a fresh
# volume — i.e. on a new deployment, never in testing against an existing one.
CONFIG_CREATED_MARKER = "created default config at data/config.json"
# The log line naming the TTS backend the app actually built. In the serving container it is the
# proof that the config this gate injected was really read: without it the app would have built
# Piper from the template and died, so seeing the injected provider here is what separates "the app
# came up" from "the app came up for the reason we think".
TTS_BACKEND_MARKER = "tts backend: {}".format
# The known, understood way a bare container stops: the eager Piper load hitting a models directory
# that is not there. BOTH fragments must appear — the exception type alone would also match a
# missing template or a missing prompt file, and the path alone would match a log line that merely
# mentioned it. See the module docstring for the full trace.
BAREBOOT_EXPECTED_FRAGMENTS = ("FileNotFoundError", "models/ru_RU-ruslan-medium.onnx")

# The TTS provider the serving container is switched to, and the key its constructor demands. The
# cloud providers (yandex, fishaudio) raise ValueError on an empty api_key in __init__ but make no
# request there, so a fake key is enough to construct one — and nothing in this gate ever calls it.
# Chosen over piper/silero because those two are exactly the ones that need a model file on disk.
SMOKE_TTS_PROVIDER = "yandex"
SMOKE_TTS_KEY = "smoke-not-a-real-key"

# The command the image-probe container runs INSTEAD of the image's own. The image's own command
# cannot survive without models (see the module docstring), so a probe container started with it
# would be racing a process on its way out. A sleeping container is a stable place to `docker exec`
# into, and the real command is covered separately, in its own two containers, which is where that
# question belongs.
# 900 s is far beyond this gate's own worst case (see the timeout arithmetic below); the container
# is removed in a `finally` regardless, and the workflow removes it again under `if: always()`.
IDLE_COMMAND = ["python", "-c", "import time; time.sleep(900)"]

# TZ matches the deployment (docker-compose.yml sets exactly this and nothing else). It is the only
# environment variable production supplies, so supplying it here keeps the container this gate
# judges and the container that runs in production configured the same way.
CONTAINER_ENV = ["TZ=Europe/Moscow"]
# The name under which both probes read the expected version from their container's environment.
# `docker exec` runs with the environment the container was CREATED with, so this is passed to
# `docker run`/`docker create` and inherited by the probe rather than interpolated into its text.
PROBE_VERSION_ENV = "SMOKE_EXPECTED_VERSION"

# Printed by each probe on its last line and nowhere else. DECLARED TWICE — here and again inside
# the probe text — and the two have to keep agreeing. The duplication is deliberate: interpolating
# them into the probes would mean `.format()` on bodies that contain braces of their own, which
# would quietly turn each probe into a template that no longer runs as-is — and being able to paste
# either block into a container by hand and have it work is most of why they are plain strings. A
# drift costs a false RED (the outer half reports a probe that never finished), never a false green.
IMAGE_PROBE_MARKER = "zakhar image probe ok"
HTTP_PROBE_MARKER = "zakhar http probe ok"

# How many verdicts each probe is expected to produce. Pinned rather than merely counted, because
# the marker alone is a substring match: a probe whose checks silently stopped emitting rows — a
# loop over a list that became empty, a check that started returning `[]` instead of a row — would
# still print the marker, still exit 0, and still look like a full pass. Requiring the exact `N/N`
# turns "fewer things were checked" into a red build. Adding a check is supposed to fail this once,
# loudly, and be answered by updating the number.
EXPECTED_IMAGE_PROBE_TARGETS = 23
EXPECTED_HTTP_PROBE_TARGETS = 5

# The prefix put in front of every row a probe reported, so the merged report says where each
# verdict was decided. Without it a reader of thirty result lines cannot tell which of them were
# taken from inside a container and which from the runner.
IMAGE_PROBE_ROW_PREFIX = "[in-image] "
HTTP_PROBE_ROW_PREFIX = "[in-container] "

# Bounds. Every docker call gets one, because a gate that hangs is worse than a gate that fails: it
# holds a slot on a runner the whole fleet queues for until the step timeout kills it, and a killed
# step never runs its cleanup.
# The numbers are generous against the real cost — this image starts in a second or two — and are
# sized for a shared runner under load right after a build. Their worst-case SUM is what the smoke
# step's `timeout-minutes` in both workflows has to exceed, so it is spelled out here, one line per
# call, in the order main() actually makes them:
#     30 (inspect the image config)                                                  =  30
#   + 30 (rm template)  + 60 (run --rm: cat the template)                            =  90
#   + 30 (rm probe)     + 60 (run -d: the idle container)                            =  90
#   + 120 (docker exec: the image probe)                                             = 120
#   + 30 (rm bareboot)  + 60 (run -d)   + 90 (marker poll: 60 budget + one 30 s logs)
#                       + 30 (docker cp the config out) + 30 (inspect its state)     = 240
#   + 30 (rm serve)     + 60 (create)   + 30 (cp the config in) + 60 (start)         = 180
#   + 90 (marker poll on the serving container: 60 budget + one 30 s logs)           =  90
#   + 180 (docker exec: the http probe)                                              = 180
#   + 30 (inspect the serving container) + 30 (one last `docker logs` for the log)   =  60
#   + 120 (the `finally`: 4 x 30)                                                    = 120
#   = 1200 s, exactly 20 minutes. Both workflows allow 22.
# A marker poll is bounded at BOOT_BUDGET in WALL CLOCK plus, in the worst case, one `docker logs`
# that started just before the deadline — hence 60 + 30 rather than 60.
# Four of these `rm`s are pre-run cleanups: every container this gate starts is removed by name
# before it is started, so a re-run from the UI — which keeps the same run id, hence the same
# $SMOKE_NAME — cannot die on "name already in use".
INSPECT_TIMEOUT = 30
REMOVE_TIMEOUT = 30
TEMPLATE_TIMEOUT = 60
START_TIMEOUT = 60
CREATE_TIMEOUT = 60
COPY_TIMEOUT = 30
IMAGE_PROBE_TIMEOUT = 120
# Larger than the image probe's because this one POLLS: it waits inside the container for the audio
# server and the panel to answer, and its own internal budget (HTTP_PROBE_BUDGET, declared in the
# probe text) has to fit inside it with room to spare for a slow `docker exec` on a loaded daemon.
HTTP_PROBE_TIMEOUT = 180
LOGS_TIMEOUT = 30

# The startup-marker poll, bounded in WALL CLOCK rather than in attempts: each attempt shells out to
# `docker logs`, whose own timeout is 30 s, so an attempt-counted bound would multiply into minutes
# the moment the daemon got slow. The marker is main.py's first log line, so it arrives within a
# second or two of the container starting; 60 s is roughly thirty times that, which is the margin a
# shared runner right after a build needs.
BOOT_BUDGET = 60
BOOT_PAUSE = 0.5

# How much of an unexpected output reaches the log. Container logs can run to thousands of lines
# when something loops, and an unbounded dump would bury the verdict.
EXCERPT_CHARS = 4000


# The program fed to `python -u -` inside the IDLE container. It lives here as a string rather than
# as a second file so that this gate stays one artifact: the workflows already pass it nothing but
# an image tag and a container name, and a second file is a second thing to forget to copy when this
# repo's CI is cloned into the next one.
#
# `-u` is not decoration: this stdout is a pipe, and CPython block-buffers those. A normal exit
# flushes (SystemExit included, so a self-reported failure is never lost), but a probe killed by
# IMAGE_PROBE_TIMEOUT would otherwise take every line it had already printed down with it — in
# exactly the run whose output is the only thing left to diagnose from.
#
# It takes NO parameters. Everything it needs is already in the container's environment, because
# `docker exec` runs with the environment the container was created with — including
# SMOKE_EXPECTED_VERSION, which start_probe_container() passes to `docker run`.
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
# src.app is the composition root and pulls in nearly everything (zeroconf, aioesphomeapi, mcp,
# httpx, pydantic, the whole src.plugins registry); the other four are the ones with a direct
# production consequence — the two servers, the config store and the version resolver. Note that
# importing src.plugins does NOT import piper/vosk/ten_vad: those are deliberately lazy, imported
# inside their backends' constructors, which is why (c) has to load the native library by hand.
SRC_MODULES = ["src.version", "src.config_store", "src.audio_server", "src.panel_api", "src.app"]


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
    zeroconf, aioesphomeapi, httpx, pydantic, mcp and the whole src.plugins registry, so a
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


# The program fed to `python -u -` inside the SERVING container. Same mechanism as the image probe
# above and the same reasons for it; what differs is that this one is not hermetic in the same
# sense — it talks to the two servers the application itself is running, over the container's own
# loopback.
#
# That is the whole reason it exists as a probe rather than as a curl on the runner: 127.0.0.1 in
# here is the application's loopback, and 127.0.0.1 on the runner is a different one. It is also
# exactly how docker-compose.yml's healthcheck reaches /health — urllib from the image's own
# interpreter, no curl, because python:3.11-slim ships none — so this checks the deployment's real
# liveness mechanism rather than an approximation of it.
HTTP_PROBE = r'''
"""Runs INSIDE the SERVING container, fed to `python -u -` over stdin.

Answers the two questions that only a running application can answer: is the audio server serving
/health, and is the admin panel answering. Both are polled rather than sampled once — the container
has just been started and the panel comes up after the audio server — and both are reached over the
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
    body_target = "...and its body is the audio server's own 'ok'"
    status, body, error = fetch(AUDIO_URL, deadline)
    if status == 200:
        rows.append((health_target, None))
        if body.strip() == "ok":
            rows.append((body_target, None))
        else:
            rows.append((body_target, (
                "it answered 200 with {!r}. AudioServer._handle_health returns the literal 'ok', so "
                "something else is answering on this port".format(body[:200]))))
    else:
        reason = (
            "it answered {!r} (last error: {}). This is the endpoint docker-compose.yml's "
            "healthcheck probes, so an image that fails here is one Docker will report as unhealthy "
            "forever — and, because src/app.py starts the audio server only after every backend has "
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
            "PanelServer.start(), so it is the one panel route that cannot be absent — a failure "
            "here means the panel is not serving at all".format(
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

    Bounded in WALL CLOCK rather than in attempts, because each attempt shells out to `docker logs`
    with its own 30 s bound and an attempt-counted loop would multiply into minutes on a slow
    daemon. `docker logs` works just as well on a container that has already exited, which is what
    makes this usable for the bare-boot container that is EXPECTED to die.
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
    """Read templates/default_config.json back out of the IMAGE.

    Out of the image rather than off the checkout, because what is being built here is the config
    the SERVING container will boot on, and the point of check (e2) is that it boots on the image's
    own template with one field changed. Reading the repo's copy instead would quietly test a
    document the image might not contain.

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


def build_serving_config(template):
    """The template with its TTS provider switched to one that needs no model file.

    THIS IS THE ONE PLACE THIS GATE DEVIATES FROM PRODUCTION'S CONFIGURATION, and the module
    docstring explains why at length: the image contains no models by design, the template selects
    Piper, and Piper loads its voice file eagerly during boot — before the audio server is started.
    Production supplies that file on the /app/models volume; a gate cannot.

    Exactly one key is changed, and the rest of the document is production's own template, so
    everything else about this boot — the audio server, the panel, the MCP endpoint, the prompt
    store, the reminders scheduler, the device manager — is configured the way a real first boot
    configures it. The api_key is required only because the cloud provider's constructor refuses an
    empty one; no request is ever made with it.
    """
    document = json.loads(json.dumps(template))  # deep copy: the caller keeps the pristine template
    tts = document.setdefault("tts", {})
    tts["selected"] = SMOKE_TTS_PROVIDER
    tts.setdefault("instances", {})[SMOKE_TTS_PROVIDER] = {"api_key": SMOKE_TTS_KEY}
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


def check_bare_boot(image, name):
    """(e1) The image's own command, no volumes, no config: production's genuine first boot.

    What this proves and what it deliberately does NOT is set out in the module docstring; the short
    version is that this image cannot come up without the models volume, so this check establishes
    the boundary precisely rather than pretending the boundary is not there.

    The container is started detached and WITHOUT `--rm`, so its filesystem and its state can still
    be read after it has exited — `docker cp` and `docker inspect` both work on a stopped container,
    which is what makes the config-on-disk row possible at all.
    """
    rows = []
    start_target = "`docker run -d` starts {} with the image's own command and no volumes".format(
        image)
    marker_target = "...its log carries {!r} (bounded at {} s)".format(STARTUP_MARKER, BOOT_BUDGET)
    created_target = "...and {!r}: the first-boot template copy fired".format(CONFIG_CREATED_MARKER)
    config_target = "...and {} really exists in the container and parses as JSON".format(
        CONFIG_PATH)
    trace_target = "...and nothing before the startup line is a traceback"
    outcome_target = "...and the boot either survives or stops in the ONE known place"
    every_target = (marker_target, created_target, config_target, trace_target, outcome_target)

    remove_container(name)
    status, output = docker(
        ["run", "-d", "--name", name] + environment_flags(CONTAINER_ENV) + [image], START_TIMEOUT)
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
    if log_status != 0:
        for target in every_target:
            rows.append((target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs))))
        return rows, logs

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
    if copy_status != 0:
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
        rows.append((trace_target, "not attempted: the startup line never appeared, so there is no "
                                   "'before' to look at"))
    else:
        # Narrow by construction, and worth saying so: the marker is main.py's FIRST log line, so
        # "before" is import and logging setup only. It is still the right place to look — an
        # ImportError raised while loading src.app lands exactly there — and the outcome row below
        # is what covers everything after it.
        before = logs.split(STARTUP_MARKER, 1)[0]
        rows.append((trace_target, None if "Traceback (most recent call last)" not in before else (
            "there is one. Everything before that line is import and logging setup, so a traceback "
            "there is the image failing before the application begins. "
            "Before the marker:\n{}".format(excerpt(before)))))

    state, state_output = container_state(name)
    if state is None:
        rows.append((outcome_target, "docker inspect failed: {}".format(excerpt(state_output))))
    elif state.startswith("true"):
        # A future in which the eager Piper load became lazy. Not a failure — an improvement — and
        # this row deliberately accepts it so that fixing the app does not redden the gate. When
        # this starts happening, the config injection in check (e2) can be dropped.
        rows.append(("{} (still running: the eager model load appears to be gone)".format(
            outcome_target), None))
    elif all(fragment in logs for fragment in BAREBOOT_EXPECTED_FRAGMENTS):
        rows.append(("{} (stopped on the missing Piper voice, as expected)".format(outcome_target),
                     None))
    else:
        rows.append((outcome_target, (
            "docker reports `Running ExitCode` = {!r} and the log does not carry {}. This image is "
            "KNOWN to stop here on the eager Piper voice load and nothing else; a different failure "
            "means something new is broken — a wheel that did not install, the libc++ layer, a "
            "missing source file — and is being hidden behind an expected one. Logs:\n{}".format(
                state, " and ".join(repr(f) for f in BAREBOOT_EXPECTED_FRAGMENTS),
                excerpt(logs)))))
    return rows, logs


def start_serving_container(image, name, version, config):
    """(e2) Start the image's own command, no volumes, with a model-free config injected.

    `docker create` -> `docker cp` -> `docker start` rather than a bind mount, and that is not a
    detour: this job runs inside act_runner's own container while `docker` drives a daemon outside
    it, so a `-v $(pwd)/...` would ask that daemon for a path that exists in a different filesystem
    — it would mount an empty directory, or a stale one, and the container would boot on the
    template again. Copying into a created-but-not-started container puts the file where the
    application looks for it with nothing shared between the two filesystems.

    NO PORT IS PUBLISHED and no volume is attached: the whole point is that the image comes up on
    its own.
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
    if status != 0:
        rows.append((copy_target, "`docker cp` exited {}:\n{}".format(status, excerpt(output))))
        rows.append((start_target, "not attempted: the config was not copied in"))
        return rows, False
    rows.append((copy_target, None))

    status, output = docker(["start", name], START_TIMEOUT)
    if status != 0:
        rows.append((start_target, "`docker start` exited {}:\n{}".format(
            status, excerpt(output))))
        return rows, False
    rows.append((start_target, None))
    return rows, True


def check_serving_boot(name, started):
    """(e2)/(g) The serving container reaches its startup line, on the config it was given.

    Returns (rows, logs). The HTTP probe runs after this and does the actual serving checks; this
    establishes that there is something to probe and that it is the program we think it is.
    """
    marker_target = "the serving container logs {!r} (bounded at {} s)".format(
        STARTUP_MARKER, BOOT_BUDGET)
    tts_marker = TTS_BACKEND_MARKER(SMOKE_TTS_PROVIDER)
    config_target = "...and {!r}: the injected config was really the one it booted on".format(
        tts_marker)
    trace_target = "...and its boot log carries no traceback"
    every_target = (marker_target, config_target, trace_target)

    if not started:
        return [(target, "not attempted: the container never started") for target in every_target], \
            ""

    log_status, logs = wait_for_marker(name, STARTUP_MARKER, BOOT_BUDGET)
    if log_status != 0:
        return [(target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs)))
                for target in every_target], logs

    rows = []
    rows.append((marker_target, None if STARTUP_MARKER in logs else (
        "it never appeared in {} s. Logs:\n{}".format(BOOT_BUDGET, excerpt(logs)))))

    # Without this row the whole of (e2) could pass on a container that had quietly booted on the
    # TEMPLATE — i.e. died on Piper — and the HTTP probe would then simply report a dead server,
    # blaming the application for the gate's own setup. This says which config was consumed.
    rows.append((config_target, None if tts_marker in logs else (
        "the log does not name it. src/app.py logs the backend it built for every stage, so its "
        "absence means the injected data/config.json was not read: either the copy landed "
        "somewhere else or something rewrote it. Logs:\n{}".format(excerpt(logs)))))

    # The whole log this time, not only the part before the marker. In THIS container the marker is
    # still main.py's first line, but everything that follows — building the backends, starting the
    # audio server, starting the panel — is a boot that is supposed to succeed, so a traceback
    # anywhere in it is a finding. The deferred stress/wakeword loads that run after the panel is up
    # report their failures with logger.error and no traceback (src/reconfig.py catches them
    # per-category), which is what keeps this row from flapping on a runner with no model files.
    rows.append((trace_target, None if "Traceback (most recent call last)" not in logs else (
        "there is one. This container is given everything it needs to come up, so a traceback in "
        "its boot log is the image failing at something that has nothing to do with the missing "
        "models. Logs:\n{}".format(excerpt(logs)))))
    return rows, logs


def check_container_alive(name, started):
    """(g) The serving container is still running after everything above.

    It was started with the image's own long-running command and given a config it can boot on, so
    there is exactly one correct answer. A container that is gone means either the application died
    after serving its first requests — the restart loop production would then sit in — or something
    killed it, the daemon's OOM killer being the realistic candidate on a runner shared with every
    other repository. In the second case every verdict the HTTP probe just produced was collected
    from a process that was being torn down underneath it.
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
        # No default here either, and this one is worth spelling out because a default would not
        # fail — it would work, quietly and wrongly, in two ways at once.
        # First, the cleanup: the workflow's `if: always()` step removes the four suffixed names,
        # and it builds them from the SAME variable. A dropped or misspelled `env:` entry would
        # leave this script naming its containers after the fallback while the cleanup step went
        # looking for the run-id ones, found nothing, and swallowed the miss in its `|| true` — the
        # gate stays green and the leak is permanent.
        # Second, collisions: the runner runs one docker daemon for every repository in the fleet,
        # so a constant fallback means two concurrent runs pick the SAME names — and
        # remove_container() below would then delete the other run's live container out from under
        # it.
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
            # everything else has had its turn at it. The log is re-read here rather than reused
            # from above so the transcript shows everything the boot produced, probe included.
            rows.extend(check_container_alive(name + SERVE_SUFFIX, serve_started))
            _, final_logs = docker(["logs", name + SERVE_SUFFIX], LOGS_TIMEOUT)
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
