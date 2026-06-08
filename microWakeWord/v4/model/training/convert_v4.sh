#!/bin/bash
cd /home/claude/zakhar-mww/micro-wake-word
export PYTHONPATH=/home/claude/zakhar-mww/micro-wake-word OMP_NUM_THREADS=50
/home/claude/zakhar-mww/venv/bin/python -m microwakeword.model_train_eval --training_config='/home/claude/zakhar-mww/training_parameters_v4.yaml' \
  --train 0 --test_tflite_streaming_quantized 1 --use_weights "last_weights" \
  mixednet --pointwise_filters "64,64,64,64,64" --repeat_in_block "1, 1, 1, 1, 1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [17,23], [29]' --residual_connection "0,0,0,0,0" \
  --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3
echo "CONVERT_EXIT=$?"
