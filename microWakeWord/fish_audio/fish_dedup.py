#!/usr/bin/env python3
"""Entity-level dedup of the fish catalog (review tool).

Exact name-dedup misses celebrity/character clones hiding under DIFFERENT titles:
«путин», «голос путина», «в.в. путин», «Владимир Путин» are ONE entity but four strings.
Conversely, generic auto-names («молодой женский голос» ×2336, «драматический ...» ×900) are
NOT clones — they are different authors' own voices (hundreds of distinct authors) = real
diversity, KEEP them.

Two modes:
  (default) RANK candidate entities: tokenize titles, drop generic voice-DESCRIPTOR
            stopwords (adjectives + role nouns), a remaining token recurring across many
            distinct voices = candidate; cluster by WORD-PREFIX (catches declensions
            путин→путина, avoids mid-word noise демо↛воландеморт). Prints #voices/#authors/
            keeper(max task_count)/sample aliases for eyeballing.
  --search T DRILL into one entity: list EVERY voice whose title contains T (substring),
            sorted by task_count desc, with task_count/author/id — so you keep the top one
            and copy the rest as an alias drop-list.

NOTE: morphology is NOT used to filter descriptors — that would also kill -ский/-ий/-ой
surnames (Зеленский, Достоевский, Толстой). Descriptors are removed by the STOP list instead.
Latin aliases (putin) surface as their own token; cross-script merge is left to the eye.

Usage:
  python fish_dedup.py microWakeWord/fish_catalog_ru.jsonl --min-voices 12 --top 60
  python fish_dedup.py microWakeWord/fish_catalog_ru.jsonl --search путин
"""
import sys, json, re, argparse
from collections import defaultdict
import fish_catalog as fc  # reuse gender()/age_bucket() (same dir; script dir is on sys.path)

# generic voice descriptors (adjectives + role nouns) — NOT entities, keep all such voices
STOP = set("""
голос голоса голосок голосовой voice вокал мужской мужская мужик женский женская male female
молодой молодая юный юная young рассказчик рассказчица рассказчики narrator диктор дикторский
девушка девочка девичий парень мужчина женщина дед бабушка тётя дядя бабка дедушка мальчик
юноша актриса актёр актер актерский актёрский блогер блогерша геймер комментатор собеседник
демо демоверсия клон клонирования клонирование тест тестовый дубляж озвучка озвучивание спикер
speaker аудио audio робот бот ассистент assistant нейросеть style стиль версия version обычный
спокойный спокойная энергичный энергичная игривый игривая выразительный выразительная
яркий яркая ясный ясная глубокий глубокая опытный опытная мудрый мудрая зрелый зрелая
драматический драматическая драматичный драматичная эмоциональный эмоциональная динамичный
разговорный разговорная уверенный уверенная неуверенный игровой игровая живой живая
экспрессивный экспрессивная властный властная грозный грозная харизматичный гневный гневная
старый старая прямой прямая четкий чёткий чёткая мелодичный мощный мощная вдумчивый деловой
деловая громкий громкая мягкий мягкая весёлый веселый весёлая веселая нежный нежная добрый
добрая строгий строгая серьезный серьёзный серьезная грубый грубая низкий высокий приятный
приятная профессиональный профессиональная тёплый теплый бодрый быстрый медленный сильный
красивый сексуальный брутальный авторитетный анимационный сказочный волшебный взрослый
русский русская russian рус rus английский нейтральный детский детская ребенок ребёнок
рассказчика рассказчику рассказчиков пожилой пожилая баритон бас загадочный загадочная
игрок ведущий ведущая энтузиаст мужчины мама папа малыш малышка певица певец дерзкий дерзкая
информатор собеседник комментатор вдумчивый вдумчивая прямой прямая загадочный мужчиной
дружелюбный дружелюбная озорной озорная яростный яростная чистый чистая меланхоличный
меланхоличная радостный радостная маленький маленькая средний средняя грустный грустная
злой злая старший старшая наставник подруга героиня диктора стример персонаж цифровой
геймера геймерша озвучка героя героев старшая взрослая старый старая шепот шёпот
""".split())

WORD_RX = re.compile(r"[a-zа-я0-9]+")


def norm_words(title):
    # lowercase + ё->е so «ёжик»/«ежик» and «чёткий»/«четкий» collapse
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", title.lower().replace("ё", "е"))
    return WORD_RX.findall(t)


def load_masks(path):
    """Confirmed entity masks (deduped): case-insensitive, ё==е, trailing * ignored."""
    masks = []
    for line in open(path, encoding="utf-8"):
        s = line.split("#")[0].strip().lower().replace("ё", "е").rstrip("*").strip()
        if s:
            masks.append(s)
    return list(dict.fromkeys(masks))


def mask_hits(words, title_norm, masks):
    """Which masks match this voice. Short masks (<=3) match a WHOLE WORD (so «жд» doesn't
    hit «каждый»); longer masks match as substring (catches declensions путин->путина)."""
    wset = set(words)
    return [m for m in masks if (m in wset if len(m) <= 3 else m in title_norm)]


def is_name_token(w):
    return len(w) >= 4 and not w.isdigit() and w not in STOP


def load(jsonl):
    items = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    # cache normalized word list per voice
    return [(it, norm_words(it.get("title") or "")) for it in items]


def rank(rows, min_voices, top):
    tok_voices = defaultdict(set)
    for it, words in rows:
        for w in {w for w in words if is_name_token(w)}:
            tok_voices[w].add(it["_id"])
    cands = sorted((w for w, s in tok_voices.items() if len(s) >= min_voices),
                   key=lambda w: -len(tok_voices[w]))
    kept = []
    for w in cands:
        if any(w in k or k in w for k in kept):  # subsume путин/путина into one
            continue
        kept.append(w)
    kept = kept[:top]

    print(f"# entity-dedup review on {len(rows)} voices (token in >= {min_voices} voices)\n")
    print(f"{'entity':16}{'voices':>7}{'auth':>6}  keeper (max task_count) | sample aliases")
    total_drop = 0
    for w in kept:
        cluster = [it for it, words in rows if any(x.startswith(w) for x in words)]
        ids = {it["_id"] for it in cluster}
        authors = {(it.get("author") or {}).get("_id") for it in cluster}
        keeper = max(cluster, key=lambda it: it.get("task_count") or 0)
        aliases = sorted({(it.get("title") or "").strip() for it in cluster} - {(keeper.get("title") or "").strip()})
        total_drop += len(ids) - 1
        print(f"{w:16}{len(ids):>7}{len(authors):>6}  {keeper.get('title')!r} | {aliases[:5]}")
    print(f"\nkeep 1 per listed entity -> drop ~{total_drop} voices ({len(kept)} entities). "
          f"Drill any with --search <token>.")


def search(rows, q):
    q = q.lower()
    hits = [it for it, words in rows if q in (it.get("title") or "").lower()]
    hits.sort(key=lambda it: -(it.get("task_count") or 0))
    authors = {(it.get("author") or {}).get("_id") for it in hits}
    print(f"# '{q}': {len(hits)} voices, {len(authors)} authors (sorted by task_count; keep top, drop rest)\n")
    print(f"{'task_count':>11}  {'author':22} {'id':32} title")
    for it in hits:
        au = ((it.get("author") or {}).get("nickname") or "")[:22]
        print(f"{it.get('task_count') or 0:>11}  {au:22} {it.get('_id'):32} {it.get('title')}")


def report(rows, min_voices, top, path):
    """Write a browsable Markdown review: a summary table of candidate entities, then each
    cluster fully expanded (KEEP = max task_count, the rest are the alias drop-list)."""
    tok_voices = defaultdict(set)
    for it, words in rows:
        for w in {w for w in words if is_name_token(w)}:
            tok_voices[w].add(it["_id"])
    cands = sorted((w for w, s in tok_voices.items() if len(s) >= min_voices), key=lambda w: -len(tok_voices[w]))
    kept = []
    for w in cands:
        if any(w in k or k in w for k in kept):
            continue
        kept.append(w)
    kept = kept[:top]

    clusters = {}
    for w in kept:
        cl = [it for it, words in rows if any(x.startswith(w) for x in words)]
        cl.sort(key=lambda it: -(it.get("task_count") or 0))
        clusters[w] = cl

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Fish catalog — entity-dedup review ({len(rows)} voices)\n\n")
        f.write("Eyeball the table, decide which rows are real ENTITIES (character/celebrity) vs\n")
        f.write("descriptors/common-names to ignore. In each cluster below, the **KEEP** line is the\n")
        f.write("highest task_count; all `drop` lines are aliases to exclude.\n\n")
        f.write("| # | entity | voices | authors | keeper (max task_count) |\n|--:|---|--:|--:|---|\n")
        for i, w in enumerate(kept, 1):
            cl = clusters[w]
            au = len({(it.get("author") or {}).get("_id") for it in cl})
            f.write(f"| {i} | **{w}** | {len(cl)} | {au} | {cl[0].get('title')} |\n")
        f.write("\n---\n\n")
        for w in kept:
            cl = clusters[w]
            au = len({(it.get("author") or {}).get("_id") for it in cl})
            f.write(f"## {w} — {len(cl)} voices, {au} authors\n\n```\n")
            for j, it in enumerate(cl):
                tag = "KEEP" if j == 0 else "drop"
                nick = ((it.get("author") or {}).get("nickname") or "")[:20]
                f.write(f"{tag} {it.get('task_count') or 0:>8}  {it.get('_id')}  {nick:20}  {it.get('title')}\n")
            f.write("```\n\n")
    print(f"[report] {len(kept)} entities -> {path}  (open it in the editor)")


def export_clean(rows, path):
    """Dump the CLEAN voice pool (rows already have mask-matched entities removed) as TSV,
    sorted by task_count desc, with gender/age so generation can sample intelligently."""
    rows2 = sorted(rows, key=lambda r: -(r[0].get("task_count") or 0))
    gcnt, n_child = defaultdict(int), 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("id\tgender\tage\ttask_count\ttitle\n")
        for it, _ in rows2:
            ge, ag = fc.gender(it.get("tags") or []), fc.age_bucket(it)
            gcnt[ge] += 1
            n_child += (ag == "child")
            title = (it.get("title") or "").replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{it['_id']}\t{ge}\t{ag}\t{it.get('task_count') or 0}\t{title}\n")
    print(f"[clean] {len(rows2)} voices -> {path}")
    print(f"[clean] gender: {dict(gcnt)} ; children: {n_child}")


def main():
    ap = argparse.ArgumentParser(description="Entity-level dedup review of the fish catalog.")
    ap.add_argument("jsonl")
    ap.add_argument("--min-voices", type=int, default=12)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--search", default=None, help="drill one entity: list all voices whose title contains this")
    ap.add_argument("--report", default=None, help="write a full Markdown review file to this path")
    ap.add_argument("--exclude", default=None, help="file of confirmed entity masks; drop matching voices from the corpus first")
    ap.add_argument("--export-clean", default=None, help="write the clean voice pool (after --exclude) as TSV to this path")
    args = ap.parse_args()
    rows = load(args.jsonl)
    if args.exclude:
        masks = load_masks(args.exclude)
        counts, keep = defaultdict(int), []
        for it, words in rows:
            hit = mask_hits(words, (it.get("title") or "").lower().replace("ё", "е"), masks)
            if hit:
                for m in hit:
                    counts[m] += 1
            else:
                keep.append((it, words))
        print(f"[exclude] {len(masks)} masks dropped {len(rows) - len(keep)} voices; {len(keep)} remain")
        big = sorted(counts.items(), key=lambda kv: -kv[1])[:20]
        print("[exclude] biggest masks (watch for over-broad): " + ", ".join(f"{m}={c}" for m, c in big))
        rows = keep
    if args.export_clean:
        export_clean(rows, args.export_clean)
    elif args.search:
        search(rows, args.search)
    elif args.report:
        report(rows, args.min_voices, args.top, args.report)
    else:
        rank(rows, args.min_voices, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
