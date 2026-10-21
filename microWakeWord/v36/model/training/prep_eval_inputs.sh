#!/bin/bash
# Gate inputs that lived in /tmp, rebuilt as the v31-v35 runs made them:
#  - sahar_eval.txt: `ls confusables_neg/*сахар*.wav` with absolute paths (v31, 2 clips);
#  - room_eval_heldout/: a symlink to the held-out room file ambient_ch1_20261016_145427.wav (v32);
#  - the beep-cued session take, its beep-free copies and their session take (v35 commands).
# pos_real_heldout.txt, panel windows, studio manifest and heldout_clips.csv come from prep_recovered.sh.
set -euo pipefail
ROOT=/home/claude/zakhar-mww; W=$ROOT/v36; R=$W/recipe; P=$ROOT/venv/bin/python; S2=$W/prep/studio2
cd "$ROOT"; export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$W/eval_inputs/room_eval_heldout"
ls confusables_neg/*сахар*.wav | sed "s#^#$PWD/#" > "$W/eval_inputs/sahar_eval.txt"
ln -sfn "$ROOT/device_captures/room_bg_ch1_20261016/ambient_ch1_20261016_145427.wav" "$W/eval_inputs/room_eval_heldout/"
$P "$R/session_take.py" "$W/data/beep_raw" "$S2/sort_manifest.csv" "$S2"
$P "$R/make_nobeep.py" "$W/data/beep_raw" "$S2/sort_manifest.csv" "$S2/heldout_clips.csv" "$S2/raw_nobeep"
$P "$R/session_take.py" "$S2/raw_nobeep" "$S2/sort_manifest.csv" "$S2/raw_nobeep"
cmp "$S2/session_manifest.csv" "$S2/raw_nobeep/session_manifest.csv" && echo manifest-identical
cat "$W/eval_inputs/sahar_eval.txt"; ls -l "$W/eval_inputs/room_eval_heldout/"
echo EVAL_INPUTS_DONE
