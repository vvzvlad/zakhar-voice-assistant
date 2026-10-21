#!/bin/bash
# Ship-gate @0.80: recipe/run_shipgate_v35.sh with the models as arguments and the inputs rebuilt under v36/.
#  0) beep-cued studio clips x4: held-out per clip (heldout_clips_eval.py) and the session rebuilt as one
#     take (session_take.py -> studio_eval.py), as recorded and beep-free (make_nobeep.py)
#  1) studio 2-min take x4, timeline method: v34 held-out words + all 53 segments
#  2) panel held-out 19 words x1 (eval_panel_heldout.py)
#  3) speaker 18 home takes x4, room held-out FA/h x4, «сахар» x4 (device_faithful_stream / room_bg_faph2)
# With FUSED=1 also the v36 checks (gate_fused.sh). Models run in parallel, one process per model and check.
# Usage: run_gate.sh <eval_dir> <label:tflite> ...
set -uo pipefail
ROOT=/home/claude/zakhar-mww; W=$ROOT/v36; R=$W/recipe
E=$1; shift; MODELS=("$@")
POS=$W/prep/run1/pos_real_heldout.txt; SAHAR=$W/eval_inputs/sahar_eval.txt; ROOMHELD=$W/eval_inputs/room_eval_heldout
S2=$W/prep/studio2; NB=$S2/raw_nobeep
cd "$ROOT"; source venv/bin/activate
export PYTHONPATH=$ROOT/micro-wake-word OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$E"

one_model() {
  local lab=$1 tfl=$2 L; L=$(echo "$lab" | tr a-z A-Z)
  python "$R/heldout_clips_eval.py" "$S2/heldout_clips.csv" "$lab:$tfl" 2>/dev/null | grep '^HOJSON' | sed 's/^HOJSON //' > "$E/$L.heldout_new.json"
  python "$R/studio_eval.py" "$S2/session_take.wav" "$S2/session_manifest.csv" "$lab:$tfl" 2>/dev/null | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/$L.session.json"
  python "$R/heldout_clips_eval.py" "$NB/heldout_clips.csv" "$lab:$tfl" 2>/dev/null | grep '^HOJSON' | sed 's/^HOJSON //' > "$E/$L.heldout_new_nobeep.json"
  python "$R/studio_eval.py" "$NB/session_take.wav" "$NB/session_manifest.csv" "$lab:$tfl" 2>/dev/null | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/$L.session_nobeep.json"
  python "$R/studio_eval.py" "$W/data/take.wav" "$W/prep/studio/studio_split_manifest.csv" "$lab:$tfl" 2>/dev/null | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/$L.studio.json"
  python "$W/eval_panel_heldout.py" "$lab:$tfl" 2>/dev/null | grep -vE "cudart|XNNPACK" > "$E/$L.panel.out"
  python "$R/device_faithful_stream.py" "$tfl" "$lab" "$POS" 4 2>/dev/null | grep '^DFJSON' | sed 's/^DFJSON //' > "$E/${L}_spk18.dfjson"
  python "$R/room_bg_faph2.py" "$tfl" "${lab}_rh" "$ROOMHELD" 2>/dev/null | grep '^ROOMFAPH' | sed 's/^ROOMFAPH //' > "$E/${L}_roomheld.json"
  python "$R/device_faithful_stream.py" "$tfl" "$lab" "$SAHAR" 4 2>/dev/null | grep '^DFJSON' | sed 's/^DFJSON //' > "$E/${L}_sahar.dfjson"
  if [ "${FUSED:-0}" = 1 ]; then bash "$W/gate_fused.sh" "$E" "$lab" "$tfl"; fi
  echo "model $lab done"
}
for M in "${MODELS[@]}"; do one_model "${M%%:*}" "${M#*:}" & done
wait
python "$W/summarize_gate.py" "$E" "${MODELS[@]}" | tee "$E/summary.txt"
echo "GATE_DONE"
