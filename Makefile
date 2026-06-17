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

$(VENV)/bin/python:
	python3 -m venv $(VENV)

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

# --- Housekeeping ------------------------------------------------------------
.PHONY: clean
clean: ## Remove the venv and Python caches
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
