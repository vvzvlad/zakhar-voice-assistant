#!/usr/bin/env python3
"""Generate «захааар» (drawn-out POSITIVE) and «захар» (short NEGATIVE) via fish.audio.

WHY: the recall floor (~21% FRR on held-out UNFAMILIAR voices) is a generalization
problem — the model keys on the «заха» onset, which is voice-specific. #E showed even
cloned device-tract synthetic positives HELP recall. fish.audio gives 2,000,000+
DISTINCT community voices for pennies («захааар» ≈ 14 UTF-8 bytes; $15 per 1M bytes),
so it's the cheapest shot at real voice diversity to lift the floor — INCLUDING
children's voices (a known weak spot). The SAME voices also give the SHORT «захар»
hard-negatives we need for duration-awareness (#A) — drawn vs short from one voice is
exactly the duration-discrimination signal.

VOICE VETTING: ~1/3 of community voices mispronounce / say garbage, and it's
voice-dependent. So each candidate voice gets ONE short «захар» probe that is checked
with Vosk; voices it does not recognize are dropped ENTIRELY before any clips are made.
--no-verify disables this.

TEXT TRICKS (from operator):
  - drawn-out length comes from the NUMBER of «а» in the text: «зах»+«а»*N+«рр».
  - a DOUBLE «рр» at the end helps the trailing «р» render.
  - emotion tags work well, written in brackets BEFORE the text: «[sobbing] Захаааарр».
    Supported: emphasis | angry | laughing | sobbing (plus neutral). DRAWN positives get
    ONE clip per tag (voice × every tag) for intonation variety; SHORT negatives are
    NEUTRAL only — sobbing/laughing stretch «захар» so it stops being short.
  - `normalize` is OFF so «ааа» / «рр» / «[tag]» are preserved verbatim.

IMPORTANT — these are CLEAN studio wavs. Raw synthetic HURTS (v11b); only device-tract
helps (#E). So this output MUST be playback-recaptured through the device before
training. drawn -> positives, short -> negatives.

API (verified against src/plugins/tts/fishaudio.py):
  POST https://api.fish.audio/v1/tts   headers: Authorization: Bearer KEY, model: s1
       body {text, reference_id, format:"pcm", sample_rate:16000, ...}  -> raw 16k mono PCM
       (fish "wav" carries a streaming placeholder header -> bogus duration; use PCM)
  GET  https://api.fish.audio/model?language=ru&visibility=public&sort_by=task_count&page_size&page_number
       optional: tag=..., title=...   ->  {total, items:[{_id,title,type,languages,state}], has_more}

Usage:
  export FISH_API_KEY=...   (or it is read from data/config.json tts.instances.fishaudio.api_key)
  python gen_fish_positives.py --dry-run                          # preview voices + cost, spend nothing
  python gen_fish_positives.py --mode both --voices 400           # vetted drawn positives + short negatives
  python gen_fish_positives.py --title детск --mode both          # target children's voices
"""
import argparse, json, os, sys, time, wave, urllib.request, urllib.parse, urllib.error

API = "https://api.fish.audio"
REPO = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)  # fish_audio/ -> microWakeWord/ -> repo root
PRICE_PER_MB = 15.0  # USD per 1,000,000 UTF-8 bytes of input text
TAGS = ["", "emphasis", "angry", "laughing", "sobbing"]  # "" = neutral
VOSK_DIR = os.path.join(REPO, "models", "vosk-model-small-ru-0.22")


def build_text(n_a, tag):
    """«зах»+«а»*n_a+«р», optional [tag] prefix. n_a=1 => short «захар».
    (Single trailing «р» + moderate «а» — double «рр» and many «а» sounded unnatural.)"""
    word = "зах" + "а" * n_a + "р"
    return (f"[{tag}] {word}" if tag else word), word


def api_key(arg):
    """--api-key > $FISH_API_KEY > data/config.json (tts.instances.fishaudio.api_key)."""
    if arg:
        return arg
    if os.environ.get("FISH_API_KEY"):
        return os.environ["FISH_API_KEY"]
    try:
        cfg = json.load(open(os.path.join(REPO, "data", "config.json"), encoding="utf-8"))

        def find(o):
            if isinstance(o, dict):
                if isinstance(o.get("fishaudio"), dict) and o["fishaudio"].get("api_key"):
                    return o["fishaudio"]["api_key"]
                for v in o.values():
                    r = find(v)
                    if r:
                        return r
            return None
        return find(cfg)
    except Exception:
        return None


def _get(url, key, timeout=30):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def iter_catalog(key, language, sort_by, tag, title, page_size=50):
    """Lazily yield (voice_id, title, task_count) across catalog pages: public,
    trained TTS voices, deduped by id."""
    seen, page = set(), 1
    while True:
        q = {"visibility": "public", "sort_by": sort_by, "page_size": page_size, "page_number": page}
        if language:
            q["language"] = language
        if tag:
            q["tag"] = tag
        if title:
            q["title"] = title
        try:
            data = _get(f"{API}/model?{urllib.parse.urlencode(q)}", key)
        except urllib.error.HTTPError as e:
            print(f"[voices] HTTP {e.code} page {page}: {e.read()[:200]!r}", file=sys.stderr)
            return
        items = data.get("items", [])
        if not items:
            return
        for it in items:
            vid = it.get("_id")
            if vid and vid not in seen and it.get("type") == "tts" and it.get("state") == "trained":
                seen.add(vid)
                yield vid, (it.get("title") or "").strip(), it.get("task_count", 0)
        if not data.get("has_more"):
            return
        page += 1


def iter_file(path):
    """Yield (voice_id, title, task_count) from a curated TSV (id\\tgender\\tage\\ttask_count\\ttitle),
    e.g. fish_voices_clean.tsv produced by fish_dedup.py --export-clean."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("#") or (i == 0 and line.lower().startswith("id\t")):
                continue
            p = line.rstrip("\n").split("\t")
            if p and p[0]:
                tc = int(p[3]) if len(p) > 3 and p[3].isdigit() else 0
                yield p[0], (p[4] if len(p) > 4 else ""), tc


def tts(key, text, reference_id, model, normalize, retries=3):
    """POST /v1/tts -> raw 16 kHz mono PCM bytes. Retries on transient errors."""
    body = json.dumps({
        "text": text, "reference_id": reference_id,
        "format": "pcm", "sample_rate": 16000,
        "normalize": normalize, "latency": "normal",
        "temperature": 0.3, "top_p": 0.7, "chunk_length": 200,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "model": model}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{API}/v1/tts", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return to_pcm(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last or "tts failed")


def to_pcm(b):
    """Raw 16-bit PCM. If fish returned a WAV (RIFF) anyway, strip to the data chunk."""
    if b[:4] == b"RIFF":
        i = b.find(b"data")
        if i != -1:
            return b[i + 8:]
    return b


def pcm_dur(pcm):
    return len(pcm) / 2 / 16000  # s16 mono @ 16 kHz


def save_wav(dst, pcm):
    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)


def load_asr():
    """Load faster-whisper large-v3 for voice vetting — accurate on clean studio audio
    (Vosk-small was too noisy: hallucinated extra words, confused с/з). None -> vet OFF."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=10)
    except Exception as e:
        print(f"[vet] faster-whisper unavailable ({e}) -> vetting OFF", file=sys.stderr)
        return None


def vet_text(asr, pcm):
    """Transcribe clean PCM (16k mono s16) with large-v3 -> lowercased text."""
    import numpy as np
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    segs, _ = asr.transcribe(a, language="ru", beam_size=1, temperature=0.0,
                             condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip().lower()


def vet_ok(text):
    """STRICT: exactly ONE word and it is «захар(а/у)» — з not с, no extra words
    («сахар»/«загор»/«захарыч»/«захар тебя» are all rejected)."""
    import re
    t = [w for w in re.sub(r"[^а-яё ]", " ", text).split() if w]
    return len(t) == 1 and t[0].startswith("захар") and len(t[0]) <= 6


def collect_voices(key, args, want, asr):
    """Pull voices from the catalog; if asr is set, VET each with one short «захар»
    probe (large-v3 + strict match) and keep only clean voices. -> (voices, tried, rejected)."""
    cat = iter_file(args.voices_file) if args.voices_file else \
        iter_catalog(key, args.language, args.sort_by, args.tag, args.title)
    if asr is None:
        good = []
        for v in cat:
            good.append(v)
            if len(good) >= want:
                break
        return good, len(good), 0
    good, tried, rejected = [], 0, 0
    max_c = args.max_candidates or want * 5
    for vid, title, tc in cat:
        if len(good) >= want or tried >= max_c:
            break
        tried += 1
        try:
            pcm = tts(key, build_text(1, "")[0], vid, args.model, args.normalize)  # short probe
        except Exception:
            rejected += 1
            continue
        if 0.3 < pcm_dur(pcm) < 30 and vet_ok(vet_text(asr, pcm)):
            good.append((vid, title, tc))
        else:
            rejected += 1
        time.sleep(args.sleep)
        if tried % 25 == 0:
            print(f"  [vet] tried {tried}, kept {len(good)}/{want}", flush=True)
    return good, tried, rejected


def synth_one(key, args, vid, k, n_a, tag, out_dir, form):
    """Synthesize one clip, wrap PCM in a correct WAV, save; 1 saved / 0 else."""
    text = build_text(n_a, tag)[0]
    name = f"fish_{vid[:8]}_{k:02d}_{form}_{tag or 'neu'}.wav"
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst):
        return 0
    try:
        pcm = tts(key, text, vid, args.model, args.normalize)
    except Exception as e:
        print(f"  FAIL {name}: {e}", flush=True)
        return 0
    if not (0.3 < pcm_dur(pcm) < 30):
        print(f"  bad-dur {name}: {pcm_dur(pcm):.1f}s", flush=True)
        return 0
    save_wav(dst, pcm)
    return 1


def main():
    ap = argparse.ArgumentParser(description="Generate «захааар»/«захар» via fish.audio.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), os.pardir, "samples", "positive_fish_src"))
    ap.add_argument("--out-short", default=None, help="short NEGATIVES dir (default: <out>_short)")
    ap.add_argument("--mode", choices=["drawn", "short", "both"], default="both")
    ap.add_argument("--voices", type=int, default=400, help="how many GOOD (vetted) voices")
    ap.add_argument("--voices-file", default=None, help="curated voice TSV (fish_voices_clean.tsv); use instead of pulling the catalog")
    ap.add_argument("--per-voice", type=int, default=1, help="drawn tag-sweeps per voice (each sweep = 1 clip per tag)")
    ap.add_argument("--short-per-voice", type=int, default=1, help="short tag-sweeps per voice (each sweep = 1 clip per tag)")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--tag", default=None, help="voice TAG filter (e.g. for children)")
    ap.add_argument("--title", default=None, help="voice TITLE search (e.g. детск / child)")
    ap.add_argument("--sort-by", default="task_count", help="score | task_count | created_at")
    ap.add_argument("--model", default="s2-pro", help="fish synth model: s2-pro (v2, says «захар» well) | s1 (v1 — mangles «захар», do NOT use)")
    ap.add_argument("--a-min", type=int, default=2, help="min «а» count for drawn (a2..a9 = full drawl spread; a2 overlaps short «захар» in length)")
    ap.add_argument("--a-max", type=int, default=9, help="max «а» count for drawn (a3..a9 all sound drawn-out by ear)")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip Vosk voice vetting (keep all)")
    ap.add_argument("--max-candidates", type=int, default=0, help="cap vetted candidates (0 = voices*5)")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true", help="list voices + cost, synth NOTHING (no vetting)")
    args = ap.parse_args()

    key = api_key(args.api_key)
    if not key:
        print("error: no fish.audio key (--api-key, $FISH_API_KEY, or data/config.json)", file=sys.stderr)
        return 2

    out_short = args.out_short or (args.out.rstrip("/") + "_short")
    tags = [""] if args.no_tags else TAGS  # drawn positives: every emotion (variety)
    short_tags = [""]  # short negatives: NEUTRAL only — sobbing/laughing stretch «захар» => not short
    a_counts = list(range(args.a_min, args.a_max + 1))

    asr = None if (args.dry_run or args.no_verify) else load_asr()
    print(f"[fish] collecting {args.voices} {args.language or 'any'} voices "
          f"(vet={'large-v3' if asr else 'OFF'}, tag={args.tag}, title={args.title})...", flush=True)
    voices, tried, rejected = collect_voices(key, args, args.voices, asr)
    print(f"[fish] kept {len(voices)} voices (tried {tried}, rejected {rejected}).", flush=True)
    if not voices:
        return 1

    n_d = len(voices) * args.per_voice * len(tags) if args.mode in ("drawn", "both") else 0
    n_s = len(voices) * args.short_per_voice * len(short_tags) if args.mode in ("short", "both") else 0
    sample = build_text(a_counts[len(a_counts) // 2], "sobbing")[0]
    est = (n_d + n_s) * len(sample.encode("utf-8")) / 1_000_000 * PRICE_PER_MB
    print(f"[fish] plan: {len(voices)} voices -> drawn {n_d} + short {n_s}; est. ${est:.3f}", flush=True)

    if args.dry_run:
        for vid, title, tc in voices[:15]:
            print(f"   {vid}  uses={tc}  {title[:40]}")
        print(f"[fish] drawn ex: {sample!r}  short ex: {build_text(1, '')[0]!r}  (--dry-run: nothing synthesized)")
        return 0

    if n_d:
        os.makedirs(args.out, exist_ok=True)
    if n_s:
        os.makedirs(out_short, exist_ok=True)
    made_d = made_s = 0
    for vi, (vid, _t, _tc) in enumerate(voices):
        if args.mode in ("drawn", "both"):
            ci = 0  # per-voice clip index: each sweep covers EVERY tag once
            for _ in range(args.per_voice):
                for tag in tags:
                    n_a = a_counts[ci % len(a_counts)]  # vary vowel length across the sweep
                    made_d += synth_one(key, args, vid, ci, n_a, tag, args.out, f"a{n_a}")
                    ci += 1
                    time.sleep(args.sleep)
        if args.mode in ("short", "both"):
            ci = 0
            for _ in range(args.short_per_voice):
                for tag in short_tags:  # neutral only — keep short «захар» genuinely short
                    made_s += synth_one(key, args, vid, ci, 1, tag, out_short, "short")
                    ci += 1
                    time.sleep(args.sleep)
        if (vi + 1) % 25 == 0:
            print(f"  ...{vi + 1}/{len(voices)} voices | drawn {made_d} short {made_s}", flush=True)

    print(f"\n[fish] DONE: drawn(positives) {made_d} -> {args.out}")
    print(f"[fish]       short(negatives) {made_s} -> {out_short}")
    print("[fish] NEXT: playback-recapture BOTH through the device, then STT-filter (large-v3) + train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
