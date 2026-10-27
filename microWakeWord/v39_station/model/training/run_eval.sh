#!/bin/bash
# Step-3 eval: run session_check.py (firmware rule dev_cont + VAD@0.30) on each station set as ONE
# continuous stream, for v38 AND v39_station. Recall sets + room FA + (for info) owner v38-gate sessions.
set -uo pipefail
cd /home/claude/zakhar-mww; source venv/bin/activate
export PYTHONPATH=$PWD/micro-wake-word OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
V=/home/claude/zakhar-mww/v39_station
SESS=$V/eval/sessions; OUT=$V/eval; VAD=/home/claude/zakhar-mww/v37/data/vad/vad.tflite
V38=/home/claude/zakhar-mww/v38/results/zakhar_v38.tflite
V39=$V/results/zakhar_v39_station.tflite
SC=/home/claude/zakhar-mww/v37/session_check.py
[ -f "$V39" ] || { echo "no v39 tflite at $V39"; exit 2; }

echo "=== RECALL sets (continuous stream, dev_cont+VAD) ==="
for s in sess_test sess_train sess_laptop; do
  python "$SC" "$SESS/$s.wav" "$SESS/$s.tsv" - "$VAD" "$OUT/${s}_eval.json" v38:"$V38" v39:"$V39"
done
printf 'start_s\tend_s\tsource\n' > "$OUT/empty_labels.tsv"
echo "=== ROOM FA (~11.5min; v39 trained on it = IN-SAMPLE) ==="
for f in day_1234utc_quiet night_0222utc room_1249_1254utc room_1254_1259utc; do
  python "$SC" "/home/claude/station_2710/neg_station_room_2710/$f.wav" "$OUT/empty_labels.tsv" - "$VAD" \
    "$OUT/room_${f}_eval.json" v38:"$V38" v39:"$V39"
done
echo "=== HELD-OUT BG-HOUR FA (1.0h; held-out for BOTH v38 and v39) ==="
for f in $(cd /home/claude/station_2710/neg_station_bg_2710 && ls *.wav | sed 's/.wav//'); do
  python "$SC" "/home/claude/station_2710/neg_station_bg_2710/$f.wav" "$OUT/empty_labels.tsv" - "$VAD" \
    "$OUT/bg_${f}_eval.json" v38:"$V38" v39:"$V39"
done
echo "EVAL_RUN_DONE"
