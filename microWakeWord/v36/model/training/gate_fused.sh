#!/bin/bash
# v36 gate checks for one model, called by run_gate.sh with FUSED=1. Clean Fish TTS clips stream at gain 1
# (no x4: they are not raw ch1); the studio background is already ch1 x4, so it streams at gain 1 too.
#  (a) held-out voices' fused phrases, whole and uncropped -> detected or not (device_faithful_stream.py)
#  (b) the same voices' isolated «захар»;  (d) their «сахар…»-type phrases with a following word
#  (c) false accepts per hour on the held-out studio background chunk (bg_faph_gain.py)
# Usage: gate_fused.sh <eval_dir> <label> <tflite>
set -uo pipefail
W=/home/claude/zakhar-mww/v36; E=$1; lab=$2; tfl=$3; L=$(echo "$lab" | tr a-z A-Z); I=$W/eval_inputs
for k in fused_ho fused_ho_nospace isolated_ho confus_ho; do
  python "$W/recipe/device_faithful_stream.py" "$tfl" "$lab" "$I/$k.txt" 1 2>/dev/null | grep '^DFJSON' | sed 's/^DFJSON //' > "$E/${L}_$k.dfjson"
done
python "$W/bg_faph_gain.py" "$tfl" "${lab}_sbg" "$I/studiobg_heldout" "0.5,0.6,0.7,0.8,0.85,0.9,0.95" 1 2>/dev/null \
  | grep '^ROOMFAPH' | sed 's/^ROOMFAPH //' > "$E/${L}_studiobg_ho.json"
