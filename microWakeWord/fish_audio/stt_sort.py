#!/usr/bin/env python3
"""Transcribe fish clips with faster-whisper large-v3 (the SAME model .128 will use to
auto-filter the full batch) and SORT them into folders by whether the STT output looks
like «захар»-family — so the operator can audition each bucket and decide whether STT
can be TRUSTED as the auto-filter.

  recognized/      — exactly ONE word, «заха...р» (захар / захаар / захаааар)
  not_recognized/  — anything else (garbage, extra words, wrong word, emotion noise)

The raw STT transcription is embedded in each copied filename AND in manifest.tsv, so
auditing not_recognized/ shows whether STT genuinely misheard or the match rule was too
strict. drawn vs short stays visible in the original filename (..._aN_... / ..._short_...).

BACKENDS:
  --backend mlx    (DEFAULT) mlx-whisper on the Apple GPU/ANE: ~1 s/clip, CPU stays free.
                   Use this on the Mac. --model turbo|large|small (turbo is cached).
  --backend faster faster-whisper large-v3 on CPU: pegs all cores, ~3-6 s/clip. This is
                   what .128 runs — use it THERE (Linux, has cooling), NOT on the laptop.

Usage:
  python stt_sort.py <dir-or-wavs...> --out <sorted-dir>                 # mlx/turbo (Mac)
  python stt_sort.py <dirs...> --out <dir> --backend faster --threads 8  # large-v3 (.128)
"""
import sys, os, glob, re, shutil, argparse, time, wave

MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",  # cached, fast + accurate
    "large": "mlx-community/whisper-large-v3-mlx",
    "small": "mlx-community/whisper-small-mlx",
}


def is_zakhar(text):
    """One word, «заха...р» family (захар / захаар / захаааар). Rejects сахар, захаров,
    захарка, multi-word and empty."""
    t = [w for w in re.sub(r"[^а-яё ]", " ", text.lower()).split() if w]
    return len(t) == 1 and t[0].startswith("заха") and t[0].endswith("р")


def safe(s):
    """Filesystem-safe, cyrillic-preserving slug of the transcription (for the filename)."""
    s = re.sub(r"[^а-яёa-z0-9]+", "_", s.lower()).strip("_")
    return (s or "EMPTY")[:40]


def collect(paths):
    out = []
    for a in paths:
        if os.path.isdir(a):
            out += glob.glob(os.path.join(a, "*.wav"))
        elif a.endswith(".wav"):
            out.append(a)
    return sorted(out)


def transcribe_mlx(repo, path):
    """Apple GPU/ANE via mlx-whisper — does NOT peg the CPU. Takes a file path directly."""
    import mlx_whisper
    r = mlx_whisper.transcribe(path, path_or_hf_repo=repo, language="ru", fp16=True)
    return r["text"].strip()


def transcribe_fw(model, path):
    """faster-whisper (CPU) — for the Linux nodes (.128/.226), pegs cores on the Mac."""
    import numpy as np
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    segs, _ = model.transcribe(a, language="ru", beam_size=1, temperature=0.0,
                               condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def main():
    ap = argparse.ArgumentParser(description="Sort fish clips into recognized/not_recognized by STT.")
    ap.add_argument("inputs", nargs="+", help="wav files and/or directories")
    ap.add_argument("--out", required=True, help="output dir (recognized/ + not_recognized/ created inside)")
    ap.add_argument("--backend", choices=["mlx", "faster"], default="mlx", help="mlx=Apple GPU (Mac), faster=CPU (.128)")
    ap.add_argument("--model", default="turbo", help="mlx model: turbo | large | small")
    ap.add_argument("--threads", type=int, default=8, help="faster-whisper cpu_threads")
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        print("no wavs found", file=sys.stderr)
        return 2
    rec = os.path.join(args.out, "recognized")
    non = os.path.join(args.out, "not_recognized")
    os.makedirs(rec, exist_ok=True)
    os.makedirs(non, exist_ok=True)

    if args.backend == "mlx":
        repo = MLX_MODELS.get(args.model, args.model)
        print(f"[stt-sort] backend=mlx model={repo} (Apple GPU/ANE, CPU stays free)", flush=True)
        recognize = lambda p: transcribe_mlx(repo, p)
    else:
        from faster_whisper import WhisperModel
        print(f"[stt-sort] backend=faster-whisper large-v3 (cpu, {args.threads} threads)", flush=True)
        fw = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=args.threads)
        recognize = lambda p: transcribe_fw(fw, p)
    man = open(os.path.join(args.out, "manifest.tsv"), "w", encoding="utf-8")
    man.write("verdict\ttranscription\tfile\n")
    n_ok = n_no = 0
    print(f"[stt-sort] {len(files)} clips...", flush=True)
    for f in files:
        t0 = time.time()
        txt = recognize(f)
        ok = is_zakhar(txt)
        base = os.path.basename(f)
        shutil.copy2(f, os.path.join(rec if ok else non, f"{safe(txt)}__{base}"))
        man.write(f"{'OK' if ok else 'NO'}\t{txt}\t{base}\n")
        n_ok += int(ok)
        n_no += int(not ok)
        print(f"{time.time() - t0:5.1f}s [{'OK' if ok else 'NO'}] {txt!r:30} {base}", flush=True)
    man.close()

    print(f"\n[stt-sort] recognized     {n_ok} -> {rec}")
    print(f"[stt-sort] NOT recognized  {n_no} -> {non}")
    print(f"[stt-sort] manifest -> {os.path.join(args.out, 'manifest.tsv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
