#!/usr/bin/env bash
# Download the in-process STT/TTS/VAD models (Vosk RU small + Piper RU ruslan +
# Silero TTS RU v4 + Silero VAD onnx) into ./models. Models are large binaries and
# are NOT committed (see .gitignore).
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

# --- Silero VAD: bare ONNX model (~2 MB, run via onnxruntime, no torch) -------
# Pinned to the v6.0 release tag (NOT master) so a fresh download is reproducible
# and can't silently pick up a future model with a changed input interface; the
# [1, 576] context-prefix interface is verified against this exact tag by the
# integration tests in tests/test_vad_silero.py.
SILERO_URL="https://github.com/snakers4/silero-vad/raw/v6.0/src/silero_vad/data/silero_vad.onnx"
SILERO_ONNX="$MODELS_DIR/silero_vad.onnx"

if [ -f "$SILERO_ONNX" ]; then
    echo "Silero VAD model already present: $SILERO_ONNX (skip)"
else
    echo "Downloading Silero VAD model..."
    curl -fsSL "$SILERO_URL" -o "$SILERO_ONNX"
fi

# --- Silero TTS: Russian v4 multi-speaker model (.pt, torch.package) ----------
# ~60-100 MB. Needs PyTorch at RUNTIME — torch is an OPTIONAL dependency (NOT in
# requirements.txt / the Docker image); install it separately to use this voice.
SILERO_TTS_URL="https://models.silero.ai/models/tts/ru/v4_ru.pt"
SILERO_TTS_PT="$MODELS_DIR/silero_tts_v4_ru.pt"

if [ -f "$SILERO_TTS_PT" ]; then
    echo "Silero TTS model already present: $SILERO_TTS_PT (skip)"
else
    echo "Downloading Silero TTS model..."
    curl -fsSL "$SILERO_TTS_URL" -o "$SILERO_TTS_PT"
fi

echo "Models ready in $MODELS_DIR/"
