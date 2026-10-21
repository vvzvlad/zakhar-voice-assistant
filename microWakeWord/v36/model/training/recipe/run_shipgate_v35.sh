#!/bin/bash
# v35 ship-gate @0.80 against v34 and v33. EVAL ONLY, no flash/deploy. The v34 gate (run_shipgate.sh) plus
# the held-out beep-cued clips and the export's built-in streaming ROC:
#  0) held-out beep-cued studio clips x4: each clip streamed alone from a fresh state (heldout_clips_eval.py),
#     and the 40 clips rebuilt as one take (session_take.py) through studio_eval.py, the timeline method;
#     both as recorded and on beep-free copies (make_nobeep.py, run once before the gate)
#  1) studio take x4, timeline method: v34 held-out words + all 53 segments (v34 studio_split_manifest.csv)
#  2) panel held-out 19 words x1 (eval_panel_heldout.py)
#  3) speaker 18 home takes x4, room held-out FAPH x4, «сахар» x4 (device_faithful_stream / room_bg_faph2)
#  4) summary incl. the export's tflite_streaming_roc.txt of each model
set -uo pipefail
ROOT=/home/claude/zakhar-mww
SP=/tmp/claude-1000/-home-claude/71fbe714-4eba-4120-a077-75d50a00bed2/scratchpad   # read-only inputs
V=/tmp/zstudio/v35; E=$V/eval
V33=/tmp/zstudio/v33.tflite; V34=/tmp/zstudio/v34/results/zakhar_v34.tflite; V35=$V/results/zakhar_v35.tflite
POS=$SP/t8/run1/pos_real_heldout.txt; SAHAR=$SP/t8/run1/sahar_eval.txt; ROOMHELD=$SP/t8/run2/room_eval_heldout
ROC33=$SP/models/zakhar_v33/tflite_stream_state_internal_quant/tflite_streaming_roc.txt
ROC34=/tmp/zstudio/v34/models/zakhar_v34/tflite_stream_state_internal_quant/tflite_streaming_roc.txt
ROC35=$V/models/zakhar_v35/tflite_stream_state_internal_quant/tflite_streaming_roc.txt
cd "$ROOT"; source venv/bin/activate
export PYTHONPATH=$ROOT/micro-wake-word OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$E"

echo "=== [0/4] held-out beep-cued clips (x4): per clip, and the session rebuilt as one take ==="
python "$V/heldout_clips_eval.py" "$V/heldout_clips.csv" "v33:$V33" "v34:$V34" "v35:$V35" 2>/dev/null \
  | grep '^HOJSON' | sed 's/^HOJSON //' > "$E/heldout_new.jsonl"
python "$V/studio_eval.py" "$V/session_take.wav" "$V/session_manifest.csv" "v33:$V33" "v34:$V34" "v35:$V35" 2>/dev/null \
  | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/session.jsonl"
# the same two checks on beep-free copies (make_nobeep.py: chime replaced by the clip's own tail noise)
NB=$V/raw_nobeep
python "$V/heldout_clips_eval.py" "$NB/heldout_clips.csv" "v33:$V33" "v34:$V34" "v35:$V35" 2>/dev/null \
  | grep '^HOJSON' | sed 's/^HOJSON //' > "$E/heldout_new_nobeep.jsonl"
python "$V/studio_eval.py" "$NB/session_take.wav" "$NB/session_manifest.csv" "v33:$V33" "v34:$V34" "v35:$V35" 2>/dev/null \
  | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/session_nobeep.jsonl"

echo "=== [1/4] studio take (x4, timeline method) ==="
python "$V/studio_eval.py" /tmp/zstudio/take.wav /tmp/zstudio/v34/studio_split_manifest.csv "v33:$V33" "v34:$V34" "v35:$V35" 2>/dev/null \
  | grep '^STUDIOJSON' | sed 's/^STUDIOJSON //' > "$E/studio.jsonl"

echo "=== [2/4] panel held-out (x1) ==="
python "$V/eval_panel_heldout.py" 2>/dev/null | grep -vE "cudart|XNNPACK" > "$E/panel.out"

echo "=== [3/4] speaker 18 takes, room held-out FAPH, сахар (x4) ==="
for M in "V33:$V33" "V34:$V34" "V35:$V35"; do
  lab=${M%%:*}; tfl=${M#*:}
  python "$V/device_faithful_stream.py" "$tfl" "$lab" "$POS" 4 2>/dev/null | grep '^DFJSON' | sed 's/^DFJSON //' > "$E/${lab}_spk18.dfjson"
  python "$V/room_bg_faph2.py" "$tfl" "${lab}_rh" "$ROOMHELD" 2>/dev/null | grep '^ROOMFAPH' | sed 's/^ROOMFAPH //' > "$E/${lab}_roomheld.json"
  python "$V/device_faithful_stream.py" "$tfl" "$lab" "$SAHAR" 4 2>/dev/null | grep '^DFJSON' | sed 's/^DFJSON //' > "$E/${lab}_sahar.dfjson"
done
echo "=== [4/4] summary ==="
python "$V/summarize_v35.py" "$E" "v33:$ROC33" "v34:$ROC34" "v35:$ROC35" | tee "$E/summary.txt"
echo "SHIPGATE_V35_DONE"
