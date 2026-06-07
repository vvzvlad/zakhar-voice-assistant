#!/bin/bash
set -e
cd /home/claude/zakhar-mww
echo "[$(date +%H:%M:%S)] extracting fma"
mkdir -p backgrounds_src/fma && unzip -q -o backgrounds_src/fma_xs.zip -d backgrounds_src/fma
mkdir -p fma_16k
echo "[$(date +%H:%M:%S)] converting fma mp3 -> 16k wav (parallel)"
find backgrounds_src/fma -name '*.mp3' | xargs -P 40 -I {} bash -c '
  f="{}"; b=$(basename "$f" .mp3)
  ffmpeg -nostdin -v error -y -i "$f" -ar 16000 -ac 1 -sample_fmt s16 "fma_16k/${b}.wav" 2>/dev/null || true
'
echo "[$(date +%H:%M:%S)] fma_16k: $(ls fma_16k/*.wav 2>/dev/null | wc -l) files"
echo "[$(date +%H:%M:%S)] downloading MIT RIR via datasets"
/home/claude/zakhar-mww/venv/bin/python - <<'PY'
import datasets, scipy.io.wavfile, os, numpy as np
os.makedirs("mit_rirs", exist_ok=True)
ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
n=0
for row in ds:
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join("mit_rirs", name), 16000, (row['audio']['array']*32767).astype(np.int16))
    n+=1
print("RIR files:", n)
PY
echo "[$(date +%H:%M:%S)] BACKGROUNDS DONE: fma_16k=$(ls fma_16k/*.wav|wc -l) mit_rirs=$(ls mit_rirs/*.wav|wc -l)"
