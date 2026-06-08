#!/usr/bin/env bash
# Download the in-process STT/TTS models (Vosk RU small + Piper RU ruslan) into
# ./models. Models are large binaries and are NOT committed (see .gitignore).
# Existing files are kept (skip-if-present), so re-running is cheap and idempotent.
set -euo pipefail

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

# --- Vosk: small Russian model (16 kHz, CPU) ---------------------------------
VOSK_ZIP_URL="https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
VOSK_ZIP="$MODELS_DIR/vosk-model-small-ru-0.22.zip"
VOSK_DIR="$MODELS_DIR/vosk-model-small-ru-0.22"

if [ -d "$VOSK_DIR" ]; then
    echo "Vosk model already present: $VOSK_DIR (skip)"
else
    echo "Downloading Vosk model..."
    curl -fsSL "$VOSK_ZIP_URL" -o "$VOSK_ZIP"
    echo "Unzipping Vosk model..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$VOSK_ZIP" -d "$MODELS_DIR"
    else
        python -m zipfile -e "$VOSK_ZIP" "$MODELS_DIR"
    fi
    rm -f "$VOSK_ZIP"
fi

# --- Piper: Russian voice (ru_RU-ruslan-medium) -------------------------------
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium"
PIPER_ONNX="$MODELS_DIR/ru_RU-ruslan-medium.onnx"
PIPER_JSON="$MODELS_DIR/ru_RU-ruslan-medium.onnx.json"

if [ -f "$PIPER_ONNX" ]; then
    echo "Piper voice already present: $PIPER_ONNX (skip)"
else
    echo "Downloading Piper voice (onnx)..."
    curl -fsSL "$PIPER_BASE/ru_RU-ruslan-medium.onnx?download=true" -o "$PIPER_ONNX"
fi

if [ -f "$PIPER_JSON" ]; then
    echo "Piper config already present: $PIPER_JSON (skip)"
else
    echo "Downloading Piper voice (config json)..."
    curl -fsSL "$PIPER_BASE/ru_RU-ruslan-medium.onnx.json?download=true" -o "$PIPER_JSON"
fi

echo "Models ready in $MODELS_DIR/"
