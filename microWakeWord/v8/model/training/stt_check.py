#!/usr/bin/env python3
"""Transcribe wav clips with Vosk ru-small and classify each as «захар»-like (keep) or
other-word (garbage/drop). Used for (a) auditing existing synthetic sets and (b)
filtering new positives before training.

Usage:
  stt_check.py audit  <dir> <sample_n>            # print % garbage + sample transcripts
  stt_check.py filter <dir> <kept_list.txt>       # write paths that pass to kept_list
"""
import sys, os, glob, json, re, random
import numpy as np, soundfile as sf, soxr
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

MODEL_DIR = sorted(glob.glob("/home/claude/zakhar-mww/v8/vosk-model-small-ru-*"))[0]
model = Model(MODEL_DIR)
# lenient «захар»-like: zahar / zaha / zaka / zahaa (STT noise on very slow speech)
KEEP = re.compile(r"заха|зака|захар|сахар")  # сахар included: с/з confusion on slow tts is acceptable-positive-ish? -> handle below

def transcribe(path):
    try:
        d, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if d.ndim > 1:
        d = d[:, 0]
    if sr != 16000:
        d = soxr.resample(d, sr, 16000)
    pcm = (np.clip(d, -1, 1) * 32767).astype(np.int16).tobytes()
    rec = KaldiRecognizer(model, 16000)
    rec.AcceptWaveform(pcm)
    res = json.loads(rec.FinalResult())
    return res.get("text", "").strip()

def is_zahar(t):
    if not t:
        return None  # empty -> ambiguous (very slow speech can decode empty); treat as keep-ish
    # Lenient «захар»-like: any token starting зах/зак (захар, заха, захаа, закар).
    # Excludes the garbage Piper produces on broken spellings (дорога, врага, драка,
    # два года, страх, граф, сбер, да...). сахар is NOT kept (wrong word / confusable).
    # Strip spaces: Vosk often splits drawn-out «за-ха-р» into 'за ха' — that is a REAL
    # positive, not garbage. After joining, real ones contain 'заха'/'зах'/'зак';
    # genuine wrong words (товар, два, трава, дорога, за охраны->заохраны) do not.
    j = t.replace(" ", "")
    if re.search(r"зах|зак", j):
        return True
    return False

def list_wavs(d):
    return sorted(glob.glob(os.path.join(d, "*.wav")))

mode = sys.argv[1]
if mode == "audit":
    d, n = sys.argv[2], int(sys.argv[3])
    files = list_wavs(d)
    random.seed(1); random.shuffle(files)
    files = files[:n]
    keep = drop = empty = 0
    examples_bad = []
    by_voice = {}
    for f in files:
        t = transcribe(f)
        bn = os.path.basename(f)
        voice = bn.split("_")[1] if "_" in bn else "?"
        by_voice.setdefault(voice, [0, 0])
        z = is_zahar(t)
        if t == "":
            empty += 1; keep += 1; by_voice[voice][0] += 1  # empty counted as keep (ambiguous)
        elif z:
            keep += 1; by_voice[voice][0] += 1
        else:
            drop += 1; by_voice[voice][1] += 1
            if len(examples_bad) < 12:
                examples_bad.append(f"{voice}: '{t}'")
    tot = len(files)
    print(f"AUDIT {d}: n={tot}  keep={keep} ({100*keep/tot:.0f}%)  GARBAGE={drop} ({100*drop/tot:.0f}%)  empty={empty}")
    print("  by voice (keep/garbage): " + ", ".join(f"{v}:{a}/{b}" for v,(a,b) in sorted(by_voice.items())))
    print("  garbage examples: " + " | ".join(examples_bad))
elif mode == "filter":
    # filter <filelist_or_dir> <kept_out> <stats_out_json>
    src, out = sys.argv[2], sys.argv[3]
    stats_out = sys.argv[4] if len(sys.argv) > 4 else None
    if os.path.isdir(src):
        files = list_wavs(src)
    else:
        files = [l.strip() for l in open(src) if l.strip()]
    kept = []
    bv = {}
    for f in files:
        bn = os.path.basename(f); voice = bn.split("_")[1] if "_" in bn else "?"
        bv.setdefault(voice, [0, 0])
        t = transcribe(f)
        if t == "" or is_zahar(t):
            kept.append(f); bv[voice][0] += 1
        else:
            bv[voice][1] += 1
    open(out, "w").write("\n".join(kept))
    if stats_out:
        json.dump(bv, open(stats_out, "w"))
    print(f"FILTER {src}: kept {len(kept)}/{len(files)} -> {out}")
