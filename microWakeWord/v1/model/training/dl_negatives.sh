#!/bin/bash
set -e
cd /home/claude/zakhar-mww
root="https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"
for f in dinner_party_eval.zip dinner_party.zip no_speech.zip speech.zip; do
  echo "[$(date +%H:%M:%S)] downloading $f"
  wget -q -O "negative_datasets/$f" "$root/$f"
  echo "[$(date +%H:%M:%S)] extracting $f"
  unzip -q -o "negative_datasets/$f" -d negative_datasets/
  rm -f "negative_datasets/$f"
done
echo "[$(date +%H:%M:%S)] downloading fma_xs.zip"
wget -q -O backgrounds_src/fma_xs.zip "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
echo "[$(date +%H:%M:%S)] ALL NEGATIVE DOWNLOADS DONE"
ls -la negative_datasets/
