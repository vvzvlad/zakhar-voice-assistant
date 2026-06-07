#!/usr/bin/env python3
"""Write a microWakeWord v2 JSON manifest for the trained tflite model.

Usage: generate_manifest.py <tflite> <out_json> <model_filename> <cutoff> <window> [arena]
"""
import sys, json, os

tflite_path = sys.argv[1]
out_json = sys.argv[2]
model_filename = sys.argv[3]
cutoff = float(sys.argv[4])
window = int(sys.argv[5])
arena = int(sys.argv[6]) if len(sys.argv) > 6 else None

# Estimate a safe tensor arena size if not provided. The okay_nabu v2 reference
# (a comparable streaming mixednet) uses 26080; we pick a slightly generous value
# scaled by the tflite size. ESPHome reports if it is insufficient; it can also be
# overridden in YAML. Voice PE has ample PSRAM.
if arena is None:
    sz = os.path.getsize(tflite_path)
    arena = max(30000, int(sz * 1.2 / 16) * 16)  # round-ish, generous floor

manifest = {
    "type": "micro",
    "wake_word": "Захар",
    "author": "autopilot (OHF-Voice/micro-wake-word pipeline)",
    "website": "",
    "model": model_filename,
    "trained_languages": ["ru"],
    "version": 2,
    "micro": {
        "probability_cutoff": round(cutoff, 4),
        "sliding_window_size": window,
        "feature_step_size": 10,
        "tensor_arena_size": arena,
        "minimum_esphome_version": "2024.7.0",
    },
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("wrote", out_json)
