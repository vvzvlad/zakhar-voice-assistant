#!/bin/bash
# Rebuild the five feature blocks of cfg_v35 that lived in /tmp and were lost on the node reboot,
# with the recovered scripts and arguments of the runs that made them (v31-v35), from the same sources.
# cwd is /home/claude/zakhar-mww: build_features_v29dev.py resolves its augmenter dirs (device_ir,
# device_noise, fma_16k) relative to it, as in the v31 run.
set -euo pipefail
ROOT=/home/claude/zakhar-mww; W=$ROOT/v36; P=$ROOT/venv/bin/python
cd "$ROOT"
export PYTHONPATH=$ROOT/micro-wake-word PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16
mkdir -p "$W/prep" "$W/features"

echo "=== [1/5] features_real_pos (v31): prep_single_voice.py 18 4 -> build_features_v29dev.py 0 2"
$P "$W/recipe/prep_single_voice.py" device_captures/operator_ch1_20261016 "$W/prep/run1" 18 4
$P build_features_v29dev.py "$W/prep/run1/real_pos_train" "$W/features/features_real_pos" 0 2 2>&1 | grep -E "DONE|Error|Traceback" || true

echo "=== [2/5] features_hardneg_speech (v32): seg_hardneg windows x4 -> build_features_hardneg.py 2"
$P "$W/seg_windows.py" "$W/prep/run2/hardneg_clips" 145427 "$ROOT/device_captures/room_bg_ch1_20261016/ambient_ch1_*.wav" 4
$P "$W/recipe/build_features_hardneg.py" "$W/prep/run2/hardneg_clips" "$W/features/features_hardneg_speech" 2 2>&1 | grep -E "DONE|Error|Traceback" || true

echo "=== [3/5] features_panel_pos (v33): panel_prep.py -> build_features_hardneg.py 40"
$P "$W/recipe/panel_prep.py" "$W/prep/panel_run"
$P "$W/recipe/build_features_hardneg.py" "$W/prep/panel_run/panel_pos_train" "$W/features/features_panel_pos" 40 2>&1 | grep -E "DONE|Error|Traceback" || true

echo "=== [4/5] features_studio_pos (v34): studio_prep.py -> build_features_hardneg.py 40"
$P "$W/recipe/studio_prep.py" "$W/data/take.wav" "$W/prep/studio"
$P "$W/recipe/build_features_hardneg.py" "$W/prep/studio/studio_pos_train" "$W/features/features_studio_pos" 40 2>&1 | grep -E "DONE|Error|Traceback" || true

echo "=== [5/5] features_studio2_pos (v35): segment_clips.py + recorded verdicts -> build_features_hardneg.py 40"
mkdir -p "$W/prep/studio2"
$P "$W/recipe/segment_clips.py" "$W/data/beep_raw" "$W/prep/studio2/sort" > "$W/prep/studio2/segment.out"
tail -1 "$W/prep/studio2/segment.out"
$P "$W/studio2_rebuild.py" "$W/data/beep_raw" "$W/prep/studio2/sort" "$W/logs_recovered/v35_sort_split_prep.out" "$W/prep/studio2"
$P "$W/recipe/build_features_hardneg.py" "$W/prep/studio2/studio2_pos_train" "$W/features/features_studio2_pos" 40 2>&1 | grep -E "DONE|Error|Traceback" || true

echo "=== spec counts"
$P - <<'PY'
import os
from mmap_ninja.ragged import RaggedMmap
F = "/home/claude/zakhar-mww/v36/features"
for b in ("features_real_pos", "features_hardneg_speech", "features_panel_pos", "features_studio_pos", "features_studio2_pos"):
    n = {s: len(RaggedMmap(os.path.join(F, b, s, "wakeword_mmap"))) for s in ("training", "validation", "testing")}
    print(f"{b:26s} {n}")
PY
echo PREP_RECOVERED_DONE
