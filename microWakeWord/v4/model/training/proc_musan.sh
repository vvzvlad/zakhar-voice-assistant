#!/bin/bash
# Wait for MUSAN download, then: split music into train-negative + a DISJOINT clean
# held-out test; build music & noise negative mmaps; make clean_music_test.wav.
set -e
cd /home/claude/zakhar-mww
export PYTHONPATH=/home/claude/zakhar-mww/micro-wake-word
PY=/home/claude/zakhar-mww/venv/bin/python
L(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1. wait for download to finish (wget process gone AND tar size stable)
while pgrep -f "wget.*musan.tar.gz" >/dev/null; do sleep 20; done
L "download finished: $(du -h v4/downloads/musan.tar.gz | cut -f1)"

# 2. extract
L "extracting musan"
tar -xzf v4/downloads/musan.tar.gz -C v4/downloads/
MUS=v4/downloads/musan

# 3. split music 75/25 (train-neg / clean-test), deterministic
$PY - <<PYEOF
import glob, random, os
m=sorted(glob.glob("v4/downloads/musan/music/**/*.wav", recursive=True))
random.seed(123); random.shuffle(m)
k=int(len(m)*0.25)
test=m[:k]; train=m[k:]
os.makedirs("v4/lists",exist_ok=True)
open("v4/lists/musan_music_train.txt","w").write("\n".join(train))
open("v4/lists/musan_music_test.txt","w").write("\n".join(test))
noise=sorted(glob.glob("v4/downloads/musan/noise/**/*.wav", recursive=True))
open("v4/lists/musan_noise.txt","w").write("\n".join(noise))
print("music:",len(m),"-> train",len(train),"test",len(test),"; noise",len(noise))
PYEOF

# 4. convert to 16k (train music, test music, noise)
conv(){ # listfile outdir
  mkdir -p "$2"
  cat "$1" | xargs -P 40 -I {} bash -c 'f="{}"; b=$(basename "$f" .wav); ffmpeg -nostdin -v error -y -i "$f" -ar 16000 -ac 1 -sample_fmt s16 "'"$2"'/${b}.wav" 2>/dev/null || true'
}
L "converting music(train/test) + noise to 16k"
conv v4/lists/musan_music_train.txt v4/musan_music_train_16k
conv v4/lists/musan_music_test.txt  v4/musan_music_test_16k
conv v4/lists/musan_noise.txt       v4/musan_noise_16k
L "16k: music_train=$(ls v4/musan_music_train_16k/*.wav|wc -l) music_test=$(ls v4/musan_music_test_16k/*.wav|wc -l) noise=$(ls v4/musan_noise_16k/*.wav|wc -l)"

# 5. build training negative mmaps (music + noise), long spectrograms, cap 20s, sharded
build_neg(){ # indir outdir(training) nshards cap
  ls "$1"/*.wav > "${2}.list"
  mkdir -p "$2"
  split -n l/"$3" -d --additional-suffix=.txt "${2}.list" "${2}/_sh_"
  for s in "$2"/_sh_*.txt; do n=$(basename "$s" .txt); $PY build_neg_mmap.py "$s" "${2}/${n}_mmap" "$4" > "${2}/${n}.log" 2>&1 & done
  wait; rm -f "$2"/_sh_*.txt "${2}.list"
}
L "building musan music train negatives"
mkdir -p v4/neg_musan_music/training
build_neg v4/musan_music_train_16k v4/neg_musan_music/training 12 20
L "building musan noise negatives"
mkdir -p v4/neg_musan_noise/training
build_neg v4/musan_noise_16k v4/neg_musan_noise/training 8 20

# 6. also build a MUSIC ambient set (val+test) from a slice of TRAIN music, for best-weights
#    selection on music-FAPH (split truncation). Use first 120 train-music tracks.
L "building musan music ambient (val/test) for best-weights selection"
head -120 <(ls v4/musan_music_train_16k/*.wav) > v4/lists/musan_amb.txt
mkdir -p v4/musan_amb/validation_ambient v4/musan_amb/testing_ambient
$PY build_neg_mmap.py v4/lists/musan_amb.txt v4/musan_amb/validation_ambient/mus_val_mmap 60 > v4/musan_amb_val.log 2>&1
$PY build_neg_mmap.py v4/lists/musan_amb.txt v4/musan_amb/testing_ambient/mus_test_mmap 60 > v4/musan_amb_test.log 2>&1

# 7. clean held-out music test wav (the 25% NEVER trained on), cap ~90 min
L "building clean_music_test.wav"
$PY - <<PYEOF
import glob, numpy as np, soundfile as sf
files=sorted(glob.glob("v4/musan_music_test_16k/*.wav")); out=[]; tot=0; cap=90*60*16000
for f in files:
    try: d,sr=sf.read(f,dtype="int16")
    except: continue
    if d.ndim>1: d=d[:,0]
    if sr!=16000: continue
    out.append(d); tot+=len(d)
    if tot>=cap: break
amb=np.concatenate(out)[:cap]
sf.write("v4/clean_music_test.wav",amb,16000,subtype="PCM_16")
print("clean_music_test.wav:",round(len(amb)/16000/60,1),"min from",len(out),"MUSAN tracks (disjoint from training)")
PYEOF

# 8. cleanup raw
L "cleanup raw musan"
rm -rf v4/downloads/musan v4/downloads/musan.tar.gz v4/musan_music_train_16k v4/musan_music_test_16k v4/musan_noise_16k
L "MUSAN_PROC_DONE"
