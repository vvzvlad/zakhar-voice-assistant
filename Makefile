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

.PHONY: models
models: ## Download Vosk + Piper models into models/
	bash scripts/download_models.sh

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
