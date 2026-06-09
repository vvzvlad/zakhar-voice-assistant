#!/bin/bash
# Convert an AM-softmax angular-head candidate (int8 + float streaming), measure quant-gap,
# run device-eval + strat_eval. Usage: gate_am.sh <TAG>
set -e
TAG="$1"; ROOT=/home/claude/zakhar-mww
TD=$ROOT/trained_models/zakhar_v10_$TAG
YAML=$ROOT/training_parameters_v10_$TAG.yaml
ARCH='mixednet --angular_head 1 --pointwise_filters 64,64,64,64,64 --repeat_in_block 1,1,1,1,1 --mixconv_kernel_sizes [5],[7,11],[9,15],[17,23],[29] --residual_connection 0,0,0,0,0 --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3'
cd $ROOT/micro-wake-word
export PYTHONPATH=$ROOT/micro-wake-word OMP_NUM_THREADS=8
echo "=== convert int8 $TAG ==="
/home/claude/zakhar-mww/venv/bin/python -m microwakeword.model_train_eval --training_config=$YAML --train 0 --test_tflite_streaming_quantized 1 --use_weights last_weights $ARCH >/dev/null 2>&1
echo "=== convert float $TAG ==="
/home/claude/zakhar-mww/venv/bin/python -m microwakeword.model_train_eval --training_config=$YAML --train 0 --test_tflite_streaming 1 --use_weights last_weights $ARCH >/dev/null 2>&1
I8=$TD/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
FL=$TD/tflite_stream_state_internal/stream_state_internal.tflite
cp $I8 $ROOT/v10/cand_$TAG.tflite
echo "int8=$(stat -c%s $I8) float=$(stat -c%s $FL) bytes"
cd $ROOT
echo "=== QUANT-GAP (float vs int8 streaming) $TAG ==="
/home/claude/zakhar-mww/venv/bin/python - "$FL" "$I8" <<'PY' 2>&1 | grep -vE "tensorflow|cuda|Warning|XNNPACK|InitializeLog|absl|optimized|rebuild|ring_buffer|Could not|delegate"
import sys,glob,numpy as np,soundfile as sf
sys.path.insert(0,"/home/claude/zakhar-mww/micro-wake-word")
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
fl,i8=sys.argv[1],sys.argv[2]; POS=sorted(glob.glob("v8/dev_heldout_pos/*.wav"))
def load(p):
    d,sr=sf.read(p,dtype="int16"); d=d[:,0] if d.ndim>1 else d; return d.astype(np.float32)/32768.0
def mx(m,c):
    pr=np.asarray(m.predict_clip(c,step_ms=10),dtype=np.float32); return float(sliding_window_view(pr,3).mean(-1).max()) if len(pr)>=3 else 0.0
mf,mi=Model(fl,stride=3),Model(i8,stride=3)
pf=np.array([mx(mf,load(p)) for p in POS]); pi=np.array([mx(mi,load(p)) for p in POS])
for c in [0.5,0.7,0.8,0.9,0.95]:
    print(f"cut {c}: float_FRR {float((pf<c).mean()):.3f}  int8_FRR {float((pi<c).mean()):.3f}  gap {float((pi<c).mean()-(pf<c).mean()):+.3f}")
print(f"mean prob float {pf.mean():.3f} int8 {pi.mean():.3f}")
PY
echo "=== DEVICE-EVAL $TAG ==="
/home/claude/zakhar-mww/venv/bin/python v8/evaluate_device.py v10/cand_$TAG.tflite 2>&1 | grep -vE "tensorflow|cuda|Warning|XNNPACK|InitializeLog|absl|optimized|rebuild|ring_buffer|Could not|delegate"
echo "=== STRAT $TAG ==="
/home/claude/zakhar-mww/venv/bin/python v9/strat_eval.py v10/cand_$TAG.tflite v10/strat_$TAG.json 2>&1 | grep -aE "^clean|^reverb|^music|^babble|^muffled|^lombard|WORST"
echo "=== MUSIC/SILENCE FAPH $TAG (decisive for FAPH gate) ==="
/home/claude/zakhar-mww/venv/bin/python v9/silence_eval2.py v10/cand_$TAG.tflite $TAG 2>&1 | grep -aE "REAL_ambient_music|REAL_device"
echo "GATE_DONE_$TAG"
