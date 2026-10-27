#!/bin/bash
set -uo pipefail
cd /home/claude/zakhar-mww; source venv/bin/activate
export PYTHONPATH=$PWD/micro-wake-word OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
V=/home/claude/zakhar-mww/v39_station
echo "=== build station_pos (rep40) ==="
python v38/recipe/build_features_hardneg.py "$V/prep/station_pos_train" "$V/features/features_station_pos" 40 2>&1 | grep -viE "warn|tensorflow|cuda|tf-trt|xla" | tail -3
echo "=== build station_neg (rep2) ==="
python v38/recipe/build_features_hardneg.py "$V/prep/station_neg_train" "$V/features/features_station_neg" 2 2>&1 | grep -viE "warn|tensorflow|cuda|tf-trt|xla" | tail -3
echo "FEATS_DONE"
