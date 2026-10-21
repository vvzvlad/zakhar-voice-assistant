#!/bin/bash
# Export the finetuned v36 model dir -> INT8 streaming tflite with the EXACT v28 deploy arch flags,
# verify 0 dequant/rsqrt/square strings, copy to v36/results. recipe/export_v35.sh with the v36 dir; the full
# model_train_eval output is also kept in logs/export_v36_full.log. No deploy/commit.
set -euo pipefail
ROOT=/home/claude/zakhar-mww
V=/home/claude/zakhar-mww/v36
MODEL=${1:?usage: export_v36.sh <model_name> <cfg.yaml>}
CFG=${2:?usage: export_v36.sh <model_name> <cfg.yaml>}
MODELDIR=$V/models/$MODEL
OUT=$V/results/${MODEL}.tflite
[ -f "$MODELDIR/last_weights.weights.h5" ] || { echo "no weights in $MODELDIR"; exit 2; }
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists — bump MODEL"; exit 2; }
cd "$ROOT"; source venv/bin/activate
export PYTHONPATH=$ROOT/micro-wake-word OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
python -m microwakeword.model_train_eval --training_config="$CFG" \
  --train 0 --test_tflite_streaming_quantized 1 --use_weights last_weights \
  mixednet --pointwise_filters "64,64,64,64,64" --repeat_in_block "1, 1, 1, 1, 1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [17,23], [29]' --residual_connection "0,0,0,0,0" \
  --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3 2>&1 | tee "$V/logs/export_v36_full.log" | tail -4
TFL="$MODELDIR/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
[ -f "$TFL" ] || { echo "FAIL: no tflite produced at $TFL"; exit 3; }
H=$(strings "$TFL" | grep -icE 'dequant|rsqrt|square' || true)
[ "$H" = "0" ] || { echo "FAIL: $H dequant/rsqrt/square strings — NOT deployable"; exit 3; }
mkdir -p "$V/results"; cp "$TFL" "$OUT"
echo "TFLITE_OK: $OUT  md5=$(md5sum "$OUT" | cut -d' ' -f1)  (0 dequant, deployable)"
