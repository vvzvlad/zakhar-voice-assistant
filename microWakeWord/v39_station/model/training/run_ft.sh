#!/bin/bash
set -uo pipefail
cd /home/claude/zakhar-mww; source venv/bin/activate
export PYTHONPATH=$PWD/micro-wake-word OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
V=/home/claude/zakhar-mww/v39_station
python v38/recipe/finetune_realpos.py "$V/cfg_v39_station.yaml" "$V/models/zakhar_v39_station" 6000 2>&1 \
  | grep -viE "warn|tensorflow|cudart_stub|cuInit|cuda_platform|tf-trt|xla|StreamExecutor|rebuild TensorFlow"
echo "FT_EXIT $?"
