#!/usr/bin/env python3
"""Generate drawn-out «захааар» positives via Yandex SpeechKit v3 (PAID API).

Yandex gives real speaker diversity (many RU voices) + emotion/amplua roles — a
diversity multiplier the local Piper/Silero voices lack. We synthesize the CORRECT
word «Захар» (stress-marked «Зах+ар»; NO broken elongated spellings — that bug bit
Piper), then ELONGATE ONLY THE LAST «а» VOWEL in the audio domain (захар -> захааар,
consonants kept normal) by a randomized factor, STT-verify (Vosk), and keep only
«захар»-like clips. Output: 16 kHz mono WAV.

Cost: this calls a PAID API once per generated clip (count = voices×roles×per-combo).
Use --per-combo / --voices to bound it; smoke-test with `--per-combo 1 --voices zahar`.

Run from the repo root (needs .venv: vosk + numpy + the repo's src on the path):
  .venv/bin/python microWakeWord/gen_yandex_positives.py --per-combo 1 --voices zahar   # smoke
  .venv/bin/python microWakeWord/gen_yandex_positives.py --per-combo 8                  # full
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Repo root on the path so we can reuse the project's Yandex v3 helpers/catalog.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.plugins.tts.yandex import YANDEX_V3_VOICES, _decode_v3_audio  # noqa: E402

V3_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
# Correct word, stress on the 2nd syllable in Yandex "+vowel" notation (the «+»
# goes BEFORE the stressed vowel). A couple of variants for variety.
STRESS_VARIANTS = ["Зах+ар", "Захар"]
RU_DEFAULT_VOICES = [  # core ru-RU set (skip the *_ru accented ones by default)
    "alena", "filipp", "ermil", "jane", "omazh", "zahar", "dasha", "julia",
    "lera", "masha", "marina", "alexander", "kirill", "anton",
]
# Vowel elongation factor range (×length of the held «а»); operator preferred
# ~3.0 (male) .. ~4.5 (female), so randomize across that band per clip.
VOWEL_FACTOR_MIN, VOWEL_FACTOR_MAX = 3.0, 4.5


def api_key() -> str:
    k = os.environ.get("YANDEX_TTS_API_KEY")
    if k:
        return k
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("YANDEX_TTS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def rng_seq(seed: int):
    """Deterministic float stream in [0,1) (no Math.random-style globals)."""
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        yield x / 0x7FFFFFFF


def synth(key: str, voice: str, role: str, speed: float, text: str) -> bytes:
    """One v3 synthesis -> MP3 bytes (raises with Yandex's diagnostic on error)."""
    import urllib.error
    import urllib.request

    hints = [{"voice": voice}, {"speed": round(speed, 3)}]
    if role:
        hints.insert(1, {"role": role})
    payload = json.dumps({
        "text": text,
        "hints": hints,
        "outputAudioSpec": {"containerAudio": {"containerAudioType": "MP3"}},
        "loudnessNormalizationType": "LUFS",
    }).encode("utf-8")
    req = urllib.request.Request(
        V3_URL, data=payload, method="POST",
        headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return _decode_v3_audio(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Yandex v3 {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e


# --- audio helpers (16 kHz mono) ---

def _load16k(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ar", "16000",
                          "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def _save16k(path: str, sig: np.ndarray) -> None:
    sig = np.clip(sig, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(sig.tobytes())


def mp3_to_wav16k(mp3: bytes, out_path: str) -> bool:
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ar", "16000",
                        "-ac", "1", "-sample_fmt", "s16", out_path], input=mp3, capture_output=True)
    return p.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _atempo_chain(r: float) -> str:
    """ffmpeg atempo chain achieving rate r (<1 = slower/longer); each stage in [0.5,2]."""
    parts = []
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.4f}")
    return ",".join(parts)


def _stretch_slice(sl: np.ndarray, factor: float) -> np.ndarray:
    """Stretch a slice by `factor` (>1 = longer), pitch-preserved, via ffmpeg atempo."""
    ti, to = tempfile.mktemp(suffix=".wav"), tempfile.mktemp(suffix=".wav")
    _save16k(ti, sl)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ti, "-filter:a",
                    _atempo_chain(1.0 / factor), "-ar", "16000", "-ac", "1", to], check=True)
    out = _load16k(to)
    os.unlink(ti)
    os.unlink(to)
    return out


def hold_vowel(in_path: str, factor: float) -> np.ndarray | None:
    """Elongate ONLY the last «а» vowel (захар -> захааар), consonants unchanged.

    The vowel is the last sustained high-energy / low-zero-crossing run; it is
    time-stretched by `factor` and spliced back between the surrounding consonants.
    Returns None if no vowel run is found.
    """
    sig = _load16k(in_path)
    n = len(sig)
    fl, hop = 160, 80
    starts = list(range(0, n - fl, hop))
    if not starts:
        return None
    rms = np.array([np.sqrt((sig[i:i + fl] ** 2).mean() + 1) for i in starts])
    zcr = np.array([((sig[i:i + fl][:-1] * sig[i:i + fl][1:]) < 0).mean() for i in starts])
    voiced = (rms > 0.30 * rms.max()) & (zcr < 0.18)  # high energy + low ZCR = vowel
    runs, s = [], None
    for k, v in enumerate(voiced):
        if v and s is None:
            s = k
        if (not v) and s is not None:
            runs.append((s, k))
            s = None
    if s is not None:
        runs.append((s, len(voiced)))
    runs = [(a, b) for a, b in runs if b - a >= 4]   # >= ~40 ms
    if not runs:
        return None
    a, b = runs[-1]                                   # last vowel run = «а» before «р»
    s_smp, e_smp = starts[a], min(starts[b - 1] + fl, n)
    return np.concatenate([sig[:s_smp], _stretch_slice(sig[s_smp:e_smp], factor), sig[e_smp:]])


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate drawn-out «захааар» positives via Yandex SpeechKit v3.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "positive_samples_yandex"))
    ap.add_argument("--voices", default="default",
                    help="'default' (core ru), 'all', or comma-separated voice names")
    ap.add_argument("--per-combo", type=int, default=8, help="clips per (voice,role) before STT filter")
    ap.add_argument("--no-verify", action="store_true", help="skip the Vosk STT filter (keep all)")
    args = ap.parse_args()

    key = api_key()
    if not key:
        print("error: YANDEX_TTS_API_KEY not set (env or .env)", file=sys.stderr)
        return 2

    if args.voices == "all":
        voices = list(YANDEX_V3_VOICES.keys())
    elif args.voices == "default":
        voices = RU_DEFAULT_VOICES
    else:
        voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec_model = None
    if not args.no_verify:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        rec_model = Model(str(REPO / "models" / "vosk-model-small-ru-0.22"))

    def is_zakhar(wav_path: str) -> bool:
        from vosk import KaldiRecognizer
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", wav_path, "-ar", "16000",
                              "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
        rec = KaldiRecognizer(rec_model, 16000)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        t = json.loads(rec.FinalResult()).get("text", "").strip().lower()
        return ("захар" in t) or ("захаа" in t) or t in ("зах", "заха", "зака")

    combos = [(v, r) for v in voices for r in (YANDEX_V3_VOICES.get(v, []) or [""])]
    print(f"{len(combos)} voice/role combos | {args.per_combo}/combo | out={out}")
    rnd = rng_seq(42)
    billed = kept = failed = novowel = 0
    for v, r in combos:
        for i in range(args.per_combo):
            speed = 0.9 + 0.2 * next(rnd)                          # ~0.9..1.1 (clean base word)
            text = STRESS_VARIANTS[0] if next(rnd) < 0.7 else STRESS_VARIANTS[1]
            factor = VOWEL_FACTOR_MIN + (VOWEL_FACTOR_MAX - VOWEL_FACTOR_MIN) * next(rnd)
            tag = f"{v}_{r or 'def'}_{i:02d}"
            try:
                mp3 = synth(key, v, r, speed, text)
                billed += 1
            except Exception as e:
                print(f"  FAIL synth {tag}: {e}")
                failed += 1
                continue
            tmp = tempfile.mktemp(suffix=".wav")
            if not mp3_to_wav16k(mp3, tmp):
                failed += 1
                continue
            held = hold_vowel(tmp, factor)               # «захар» -> «захааар» (vowel only)
            os.unlink(tmp)
            if held is None:
                novowel += 1
                continue
            dst = str(out / f"yx_{tag}_v{factor:.1f}.wav")
            _save16k(dst, held)
            if not args.no_verify and not is_zakhar(dst):
                os.remove(dst)
                continue
            kept += 1
        print(f"  {v}/{r or 'default'}: running kept={kept}")
    total = len(glob.glob(str(out / '*.wav')))
    print(f"\nDone: billed calls {billed} | KEPT (захар-verified, vowel-held) {kept} | "
          f"failed {failed} | no-vowel {novowel}")
    print(f"total wav in {out}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
