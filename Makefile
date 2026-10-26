# Makefile — single entry point for every repeated action in this project.
# Run `make` (or `make help`) to list the available targets.
#
# All routine commands (environment setup, tests, run, docker build/push) live
# here so they stay documented, consistent and hard to get wrong. Prefer adding
# a target over writing a one-off command in the shell or in CI.

# --- Configuration -----------------------------------------------------------
VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Frontend (Vite/React admin panel). The backend serves its built output from
# $(FRONTEND_DIST) (see src/app.py), so the dist must be built before `make run`.
FRONTEND_DIR  ?= frontend
FRONTEND_DIST := $(FRONTEND_DIR)/dist

.DEFAULT_GOAL := help

# Remove a target file/dir whose recipe fails or is interrupted, so a partial
# download/extraction is never mistaken for a finished artifact on the next run.
.DELETE_ON_ERROR:

# --- Help --------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- Environment -------------------------------------------------------------
# The project ALWAYS runs inside a local .venv. Every Python target depends on
# the virtualenv, so it is created automatically on first use and reused after —
# the system Python is never used directly.
.PHONY: venv
venv: $(VENV)/bin/python ## Create the local virtualenv (.venv) if missing

# Python 3.11, as in the app image (Dockerfile) and CI: the pinned onnxruntime
# ships no wheels for newer interpreters than it was built for.
PYTHON ?= python3.11

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

# Sentinel: dependencies are (re)installed only when a requirements file changes,
# not on every `make test` / `make run`.
$(VENV)/.deps-installed: requirements-dev.txt requirements.txt | $(VENV)/bin/python
	$(PIP) install -r requirements-dev.txt
	touch $@

.PHONY: install
install: $(VENV)/.deps-installed ## Create .venv (if missing) and install dev/test deps

.PHONY: config
config: ## Create data/config.json from the template if it does not exist
	@test -f data/config.json || (mkdir -p data && cp templates/default_config.json data/config.json && echo "created data/config.json")

# --- Models ------------------------------------------------------------------
# In-process STT/TTS/VAD models, downloaded into $(MODELS_DIR). They are large
# binaries and are NOT committed (see .gitignore). Each artifact is a file/dir
# target, so Make's own up-to-date check gives "skip if already present" for free
# — re-running is cheap and idempotent.
MODELS_DIR ?= models
PIPER_BASE := https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium

# Vosk: small Russian model (16 kHz, CPU). The target is the extracted dir; the
# recipe downloads the zip, unzips it (with a python3 -m zipfile fallback when
# `unzip` is absent), then removes the zip.
$(MODELS_DIR)/vosk-model-small-ru-0.22:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Vosk model..."
	curl -fsSL "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip" -o "$(MODELS_DIR)/vosk-model-small-ru-0.22.zip"
	@if command -v unzip >/dev/null 2>&1; then unzip -q "$(MODELS_DIR)/vosk-model-small-ru-0.22.zip" -d "$(MODELS_DIR)"; else python3 -m zipfile -e "$(MODELS_DIR)/vosk-model-small-ru-0.22.zip" "$(MODELS_DIR)"; fi
	rm -f "$(MODELS_DIR)/vosk-model-small-ru-0.22.zip"

# Piper: Russian voice (ru_RU-ruslan-medium) — model weights + config json.
$(MODELS_DIR)/ru_RU-ruslan-medium.onnx:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Piper voice (onnx)..."
	curl -fsSL "$(PIPER_BASE)/ru_RU-ruslan-medium.onnx?download=true" -o "$@"

$(MODELS_DIR)/ru_RU-ruslan-medium.onnx.json:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Piper voice (config json)..."
	curl -fsSL "$(PIPER_BASE)/ru_RU-ruslan-medium.onnx.json?download=true" -o "$@"

# Silero VAD: bare ONNX model (~2 MB, run via onnxruntime, no torch). Pinned to
# the v6.0 release tag (NOT master) so a fresh download is reproducible and can't
# silently pick up a future model with a changed input interface; the [1, 576]
# context-prefix interface is verified against this exact tag by the integration
# tests in tests/test_vad_silero.py.
$(MODELS_DIR)/silero_vad.onnx:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Silero VAD model..."
	curl -fsSL "https://github.com/snakers4/silero-vad/raw/v6.0/src/silero_vad/data/silero_vad.onnx" -o "$@"

# Silero TTS: Russian v4 multi-speaker model (.pt, torch.package; ~60-100 MB).
# Needs PyTorch at RUNTIME — torch is an OPTIONAL dependency (NOT in
# requirements.txt / the Docker image); install it separately to use this voice.
$(MODELS_DIR)/silero_tts_v4_ru.pt:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading Silero TTS model..."
	curl -fsSL "https://models.silero.ai/models/tts/ru/v4_ru.pt" -o "$@"

.PHONY: models models-vosk models-piper models-silero-vad models-silero-tts
models-vosk: $(MODELS_DIR)/vosk-model-small-ru-0.22 ## Download the Vosk RU STT model
models-piper: $(MODELS_DIR)/ru_RU-ruslan-medium.onnx $(MODELS_DIR)/ru_RU-ruslan-medium.onnx.json ## Download the Piper RU TTS voice
models-silero-vad: $(MODELS_DIR)/silero_vad.onnx ## Download the Silero VAD model
models-silero-tts: $(MODELS_DIR)/silero_tts_v4_ru.pt ## Download the Silero TTS model (needs torch at runtime)
models: models-vosk models-piper models-silero-vad models-silero-tts ## Download Vosk + Piper + Silero (TTS & VAD) models into models/
	@echo "Models ready in $(MODELS_DIR)/"

# --- Frontend ----------------------------------------------------------------
# The admin panel is a Vite/React app. The backend serves the built dist, so it
# must exist before the app runs. Both steps are incremental (sentinel-gated):
#   * npm deps are reinstalled only when package.json / package-lock.json change;
#   * the dist is rebuilt only when a frontend source file or manifest changes.
FRONTEND_SRC := $(shell find $(FRONTEND_DIR)/src -type f 2>/dev/null) \
                $(FRONTEND_DIR)/index.html $(FRONTEND_DIR)/vite.config.js

# Sentinel: a clean, reproducible install keyed on the lockfile.
$(FRONTEND_DIR)/node_modules/.installed: $(FRONTEND_DIR)/package.json $(FRONTEND_DIR)/package-lock.json
	cd $(FRONTEND_DIR) && npm ci
	touch $@

# Build output target: rebuilds when deps or any source/manifest file changes.
$(FRONTEND_DIST)/index.html: $(FRONTEND_DIR)/node_modules/.installed $(FRONTEND_SRC)
	cd $(FRONTEND_DIR) && npm run build

.PHONY: frontend
frontend: $(FRONTEND_DIST)/index.html ## Build the admin panel frontend (Vite) into dist/

# --- Develop -----------------------------------------------------------------
.PHONY: test
test: install ## Run the test suite (auto-creates .venv if missing)
	$(PYTEST)

.PHONY: run
run: install frontend ## Run the app (auto-creates .venv, builds the frontend)
	$(PY) main.py

# --- Firmware ----------------------------------------------------------------
# Host test of the zakhar_link component's pure core (no ESP-IDF). The core uses IDF's json
# component; here cJSON is fetched at the version ESP-IDF 5.5.4 ships and compiled next to it.
FW_LINK := firmware/components/zakhar_link
FW_HOST := firmware/build-host
CJSON_V := 1.7.19
CJSON   := https://raw.githubusercontent.com/DaveGamble/cJSON/v$(CJSON_V)
CJSON_D := $(FW_HOST)/cJSON-$(CJSON_V)

$(CJSON_D)/cJSON.c $(CJSON_D)/cJSON.h:
	@mkdir -p $(CJSON_D)
	curl -fsSL "$(CJSON)/$(notdir $@)" -o "$@"

.PHONY: fw-host-test
fw-host-test: $(CJSON_D)/cJSON.c $(CJSON_D)/cJSON.h ## Host-test the firmware zakhar_link core
	cc -std=gnu17 -Wall -Wextra -Werror -I $(FW_LINK)/include -I $(FW_LINK) -I $(CJSON_D) \
		$(FW_LINK)/host_test/test_zl_core.c $(FW_LINK)/zl_core.c $(CJSON_D)/cJSON.c \
		-lm -o $(FW_HOST)/test_zl_core
	$(FW_HOST)/test_zl_core

# The Voice PE firmware (firmware/, plain ESP-IDF). idf.py runs in espressif/idf:v5.5.4, the IDF
# ESPHome built the speaker's bootloader with, on the cirunner docker host, which cannot see this
# laptop's paths: the sources go in as a tar on stdin (no wake-word model: the speaker gets its
# models from the server), the flashable artefacts come back as a tar on stdout into
# firmware/build/, the build log goes to stderr.
# fw-build-seed is the one-time image for the switch from ESPHome: only it takes
# firmware/sdkconfig.secrets (gitignored: a Wi-Fi network) and stores that network on a speaker
# that has none. fw-build leaves the file on the laptop; its image joins over BLE provisioning.
# The container is thrown away, its /root/.cache is not: the cirunner volume zakhar-fw-cache keeps
# ccache and the component manager's downloads, shared by every branch and session (the sources
# always sit at /w/firmware, so ccache hits across them).
# The container is not pinned to CPUs, so parallel builds spread over all 72 of cirunner's; the
# 24-compile cap that keeps a cold build inside its 19 GB lives in firmware/CMakeLists.txt.
FW_DOCKER := docker -c cirunner
FW_IDF    := espressif/idf:v5.5.4
FW_OUT    := build/voicepe.bin build/bootloader/bootloader.bin build/partition_table/partition-table.bin \
             build/ota_data_initial.bin build/flash_args dependencies.lock
FW_SEED    = $(if $(filter fw-build-seed,$@),1,0)

.PHONY: fw-build fw-build-seed fw-flash fw-ota fw-logs fw-voicepe-host-test
fw-build: ## Build the Voice PE firmware on cirunner into firmware/build/
fw-build-seed: ## Build the one-time ESPHome switch image, with the Wi-Fi of firmware/sdkconfig.secrets
fw-build fw-build-seed:
	set -o pipefail; COPYFILE_DISABLE=1 tar -cf - --exclude firmware/build --exclude firmware/build-host \
		--exclude firmware/managed_components --exclude firmware/sdkconfig --exclude firmware/sdkconfig.old \
		$(if $(filter 1,$(FW_SEED)),,--exclude firmware/sdkconfig.secrets) \
		firmware \
	| $(FW_DOCKER) run --rm -i -v zakhar-fw-cache:/root/.cache --entrypoint bash $(FW_IDF) -c 'set -e; mkdir /w; cd /w; tar -xf -; \
		cd firmware; . $$IDF_PATH/export.sh >&2; \
		idf.py -DZAKHAR_GIT_VERSION=$(shell git describe --always --dirty) -DZAKHAR_WIFI_SEED=$(FW_SEED) \
			build size >&2; tar -cf - $(FW_OUT)' \
	| tar -C firmware -xf -

fw-flash: ## Flash firmware/build/ over USB: make fw-flash PORT=/dev/cu.usbmodem...
	@test -n "$(PORT)" || { echo "PORT=<serial port> is required" >&2; exit 1; }
	cd firmware/build && esptool.py --chip esp32s3 -p $(PORT) -b 460800 --before default_reset \
		--after hard_reset write_flash @flash_args

fw-ota: ## Send firmware/build/voicepe.bin to a running speaker: make fw-ota HOST=<address>
	@test -n "$(HOST)" || { echo "HOST=<device address> is required" >&2; exit 1; }
	curl -fsS --data-binary @firmware/build/voicepe.bin -H 'Content-Type: application/octet-stream' \
		http://$(HOST)/ota

fw-logs: ## Stream a running speaker's log over Wi-Fi: make fw-logs HOST=<address>
	@test -n "$(HOST)" || { echo "HOST=<device address> is required" >&2; exit 1; }
	nc $(HOST) 23

fw-voicepe-host-test: ## Host-test the Voice PE firmware's pure cores (audio, indication, dial, button)
	@mkdir -p $(FW_HOST)
	cc -std=gnu17 -Wall -Wextra -Werror -I firmware/main firmware/main/host_test/test_audio_core.c \
		firmware/main/audio_core.c -lm -o $(FW_HOST)/test_audio_core
	cc -std=gnu17 -Wall -Wextra -Werror -I firmware/main firmware/main/host_test/test_ui_core.c \
		firmware/main/ui_core.c -o $(FW_HOST)/test_ui_core
	$(FW_HOST)/test_audio_core
	$(FW_HOST)/test_ui_core

# --- Yandex Station Lite -----------------------------------------------------
# station-lite/: the zakhar device client for the Yandex Station Lite board, one static armhf
# binary in place of the board's stock audio software. It compiles the firmware's zakhar_link,
# wake_word, audio_core and ui_core unchanged over station-lite/compat/'s ESP-IDF and FreeRTOS shims.
# The build runs on the cirunner docker host like fw-build: the sources in as a tar on stdin, the
# binary back as a tar on stdout into station-lite/build/, the log on stderr. The toolchain is
# dockcross's musl one: the glibc of dockcross/linux-armv7 refuses to start on the board's 4.9
# kernel ("FATAL: kernel too old", it wants 5.4), and its -lts images want 4.19.
# station-lite/build.sh fetches the third-party code at pinned versions; the cirunner volume
# zakhar-station-cache keeps the downloads and the built TFLite Micro and webrtc-audio-processing.
STATION_IMAGE := dockcross/linux-armv7l-musl:20260928-b67e766
STATION_OUT   := station-lite/build
STATION_SSH    = ssh -p 2222 root@$(HOST)

.PHONY: station-build station-host-test station-run station-install
station-build: ## Build the Station Lite client on cirunner into station-lite/build/zakhar-station
	@mkdir -p $(STATION_OUT)
	set -o pipefail; COPYFILE_DISABLE=1 tar -cf - station-lite/build.sh station-lite/compat station-lite/main \
		firmware/components/zakhar_link firmware/components/wake_word firmware/main/audio_core.c \
		firmware/main/audio_core.h firmware/main/music_ring.h firmware/main/ui_core.c firmware/main/ui_core.h \
	| $(FW_DOCKER) run --rm -i -v zakhar-station-cache:/cache --entrypoint bash $(STATION_IMAGE) -c 'set -e; \
		mkdir /w; cd /w; tar -xf -; station-lite/build.sh $(shell git describe --always --dirty) >&2; \
		tar -cf - -C station-lite/build zakhar-station' \
	| tar -C $(STATION_OUT) -xf -
	file $(STATION_OUT)/zakhar-station

station-host-test: ## Host-test the Station Lite client's pure parts (WebSocket frames, mDNS packets, volume)
	@mkdir -p $(STATION_OUT)/host
	cc -std=gnu17 -Wall -Wextra -Werror -I station-lite/compat station-lite/host_test/test_ws_frame.c \
		station-lite/compat/ws_frame.c -o $(STATION_OUT)/host/test_ws_frame
	cc -std=gnu17 -Wall -Wextra -Werror -I station-lite/compat station-lite/host_test/test_mdns_packet.c \
		station-lite/compat/mdns_packet.c -o $(STATION_OUT)/host/test_mdns_packet
	cc -std=gnu17 -Wall -Wextra -Werror -I station-lite/main station-lite/host_test/test_volume.c \
		station-lite/main/volume.c -lm -o $(STATION_OUT)/host/test_volume
	$(STATION_OUT)/host/test_ws_frame
	$(STATION_OUT)/host/test_mdns_packet
	$(STATION_OUT)/host/test_volume

# The stock software stops with the board's own helper (`q start` brings it back, as does a
# reboot) and no maind may be left: it holds the microphones. A zakhar-station still running goes
# too; the binary lands in /tmp (the rootfs is read-only) and runs detached, its log in
# /tmp/zakhar-station.log, its stored volume in /tmp/volume.
station-run: ## Start station-lite/build/zakhar-station on a board: make station-run HOST=<address>
	@test -n "$(HOST)" || { echo "HOST=<board address> is required" >&2; exit 1; }
	$(STATION_SSH) -n '/system/vendor/quasar/q stop; \
		! pidof maind >/dev/null || { echo "maind is still running" >&2; exit 1; }; \
		killall zakhar-station 2>/dev/null; while pidof zakhar-station >/dev/null; do sleep 1; done'
	$(STATION_SSH) 'cat >/tmp/zakhar-station && chmod +x /tmp/zakhar-station && \
		{ nohup /tmp/zakhar-station </dev/null >/tmp/zakhar-station.log 2>&1 & }' <$(STATION_OUT)/zakhar-station

# The board as a zakhar device from every boot on: the binary (and the volume it stores next to
# itself) in /data/opt/zakhar, /data being the board's writable partition; station-lite/S99zakhar
# in /etc/init.d, and the stock S99quasar there renamed to quasar.disabled so the boot skips it
# (renaming it back brings quasar back). The rootfs is read-only: remounted for the script and the
# rename, synced before it goes back. The stock software and any running client stop first, then
# the installed client starts.
station-install: ## Install the Station Lite client in place of quasar, from every boot: make station-install HOST=<address>
	@test -n "$(HOST)" || { echo "HOST=<board address> is required" >&2; exit 1; }
	$(STATION_SSH) -n '/system/vendor/quasar/q stop; \
		! pidof maind >/dev/null || { echo "maind is still running" >&2; exit 1; }; \
		killall zakhar-station 2>/dev/null; while pidof zakhar-station >/dev/null; do sleep 1; done; \
		mkdir -p /data/opt/zakhar'
	$(STATION_SSH) 'cat >/data/opt/zakhar/zakhar-station && chmod +x /data/opt/zakhar/zakhar-station && sync' \
		<$(STATION_OUT)/zakhar-station
	$(STATION_SSH) 'mount -o remount,rw / && cat >/etc/init.d/S99zakhar && chmod +x /etc/init.d/S99zakhar && \
		{ [ ! -f /etc/init.d/S99quasar ] || mv /etc/init.d/S99quasar /etc/init.d/quasar.disabled; } && sync; \
		mount -o remount,ro / && /etc/init.d/S99zakhar start' <station-lite/S99zakhar

# --- Housekeeping ------------------------------------------------------------
.PHONY: clean
clean: ## Remove the venv and Python caches
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
