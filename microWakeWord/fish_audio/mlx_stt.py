#!/usr/bin/env python3
"""Fast STT on Apple Silicon via mlx-whisper (GPU / Neural Engine).

Use THIS on the Mac instead of faster-whisper: faster-whisper runs large-v3 on the
CPU with cpu_threads=N, pegs all cores to 100% and is slow (~3-6 s/clip) — it looks
like it "hangs at launch". mlx runs on the M-series GPU: ~1 s/clip, CPU stays free.
First clip in a process pays a one-time warmup (model load + MLX compile); the rest
are ~1 s each, so feed many clips in ONE call.

Heavy/full STT runs still belong on .128/.226 (Linux nodes have no mlx) — this is for
quick local checks on the laptop.

Usage:
  python mlx_stt.py file1.wav file2.wav ...
  python mlx_stt.py somedir/            # all *.wav in dir
  python mlx_stt.py --model small somedir/   # smaller/faster model
"""
import sys, os, glob, time
import mlx_whisper

MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",  # default: fast + accurate
    "large": "mlx-community/whisper-large-v3-mlx",
    "small": "mlx-community/whisper-small-mlx",        # lighter/faster, lower quality
}


def collect(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            out += sorted(glob.glob(os.path.join(a, "*.wav")))
        elif a.endswith(".wav"):
            out.append(a)
    return out


def main():
    argv = sys.argv[1:]
    repo = MODELS["turbo"]
    if "--model" in argv:
        i = argv.index("--model")
        repo = MODELS.get(argv[i + 1], argv[i + 1])
        del argv[i:i + 2]
    files = collect(argv)
    if not files:
        print("usage: mlx_stt.py <wav-or-dir> [...] [--model turbo|large|small]", file=sys.stderr)
        return 2
    print(f"[mlx] {len(files)} clip(s), model={repo} (first clip warms up, rest ~1s)\n", flush=True)
    for f in files:
        t = time.time()
        r = mlx_whisper.transcribe(f, path_or_hf_repo=repo, language="ru", fp16=True)
        print(f"{time.time() - t:6.2f}s  {r['text'].strip()!r:32}  {os.path.basename(f)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
