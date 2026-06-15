#!/usr/bin/env python3
"""Pull the full fish.audio voice catalog for a language and report stats so we can pick
voices intelligently: total count, train_mode ("version"), gender + age/children (from the
voice `tags`), and NAME-DEDUP — the top voices by task_count are celebrity/character clones
(Хоумлендер / Путин / Зеленский), heavily duplicated, so the raw popularity ranking is a
poor sampler. Dedup by normalized title shows how many DISTINCT voices actually exist.

Catalog is dumped to a JSONL (one voice per line) so analysis can be re-run without
re-fetching. page_size up to 1000 works; pagination is clean (no overlap); has_more is
always null, so we page until a page is empty / short / all-duplicate / errors.

Usage:
  python fish_catalog.py                       # pull ru, dump + report
  python fish_catalog.py --language ru --out microWakeWord/fish_catalog_ru.jsonl
  python fish_catalog.py --analyze-only microWakeWord/fish_catalog_ru.jsonl   # re-report
"""
import sys, os, json, re, time, argparse, urllib.parse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import gen_fish_positives as g  # reuse api_key() + _get()

AGE_MAP = {
    "child": "child", "kid": "child", "kids": "child", "baby": "child", "infant": "child",
    "teen": "teen", "teenager": "teen", "teenage": "teen",
    "young": "young", "young-adult": "young", "youth": "young",
    "middle-aged": "middle", "adult": "middle", "mature": "middle",
    "old": "old", "older": "old", "elderly": "old", "senior": "old", "aged": "old",
}
CHILD_RX = re.compile(r"детск|ребён|ребен|малыш|child|\bkid|\bbaby", re.I)


def fetch_all(key, lang, sort, page_size, out_jsonl, sleep, max_pages):
    seen, pn = set(), 1
    with open(out_jsonl, "w", encoding="utf-8") as f:
        while pn <= max_pages:
            q = {"visibility": "public", "sort_by": sort, "page_size": page_size, "page_number": pn}
            if lang:
                q["language"] = lang
            url = f"{g.API}/model?{urllib.parse.urlencode(q)}"
            data = None
            for attempt in range(5):  # retry transient timeouts instead of aborting the whole pull
                try:
                    data = g._get(url, key, timeout=120)
                    break
                except Exception as e:
                    print(f"[catalog] page {pn} try {attempt + 1}/5 failed ({e})", flush=True)
                    time.sleep(2 * (attempt + 1))
            if data is None:
                print(f"[catalog] page {pn} ERROR (5 retries) -> stop at {len(seen)}", flush=True)
                break
            items = data.get("items", [])
            if not items:
                print(f"[catalog] page {pn} empty -> done (total {len(seen)})", flush=True)
                break
            new = 0
            for it in items:
                i = it.get("_id")
                if i and i not in seen:
                    seen.add(i)
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    new += 1
            if pn % 10 == 0:
                print(f"[catalog] page {pn}: +{new} (total {len(seen)} / {data.get('total')})", flush=True)
            if new == 0:
                print(f"[catalog] page {pn}: all duplicates -> done (total {len(seen)})", flush=True)
                break
            if len(items) < page_size:
                print(f"[catalog] page {pn}: short page ({len(items)}) -> done (total {len(seen)})", flush=True)
                break
            pn += 1
            time.sleep(sleep)
    return len(seen)


def norm_name(title):
    """Normalize a voice title for dedup: lowercase, drop [..]/(..) suffixes and punctuation."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", title.lower())
    t = re.sub(r"[^a-zа-яё0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def gender(tags):
    tl = {x.lower() for x in tags}
    m, fem = "male" in tl, "female" in tl
    return "male" if m and not fem else "female" if fem and not m else "both" if m and fem else "unknown"


def is_child(it):
    """Child detection takes PRIORITY over the young/middle age tags that fish's auto-tagger
    slaps on almost everything (those masked children: 2 by tag vs 695 by title)."""
    tags = {x.lower() for x in (it.get("tags") or [])}
    if tags & {"child", "kid", "kids", "baby", "infant"}:
        return True
    return bool(CHILD_RX.search(f"{it.get('title') or ''} {it.get('description') or ''}"))


def age_bucket(it):
    if is_child(it):
        return "child"
    for t in (x.lower() for x in (it.get("tags") or [])):
        b = AGE_MAP.get(t)
        if b and b != "child":
            return b
    return "unknown"


def analyze(jsonl):
    items = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    n = len(items)
    tm = Counter(it.get("train_mode") for it in items)
    gen = Counter(gender(it.get("tags") or []) for it in items)
    ages = Counter(age_bucket(it) for it in items)
    n_child = sum(1 for it in items if is_child(it))

    groups = defaultdict(list)
    for it in items:
        groups[norm_name(it.get("title") or "")].append(it)
    groups.pop("", None)
    uniq = len(groups)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    dup_clips = sum(len(v) - 1 for v in dups.values())
    top = sorted(dups.items(), key=lambda kv: -len(kv[1]))[:30]

    print(f"\n========== fish catalog report ({jsonl}) ==========")
    print(f"total voices fetched : {n}")
    print(f"train_mode (version) : {dict(tm)}")
    print(f"gender               : {dict(gen)}")
    print(f"age buckets          : {dict(ages)}")
    print(f"children voices      : {n_child}")
    print(f"unique names         : {uniq}   (duplicate-name clones: {dup_clips})")
    print(f"\nTOP-30 duplicated names (clone clusters):")
    for name, lst in top:
        print(f"  {len(lst):>4}x  {name[:34]:34}  e.g. {lst[0].get('title')!r}")
    # gender within children, and clean unique-voice gender mix
    child_gen = Counter(gender(it.get("tags") or []) for it in items if is_child(it))
    print(f"\nchildren by gender   : {dict(child_gen)}")


def main():
    ap = argparse.ArgumentParser(description="Pull + analyze the fish.audio voice catalog.")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--sort-by", default="task_count")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fish_catalog_ru.jsonl"))
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-pages", type=int, default=10000)
    ap.add_argument("--analyze-only", default=None, help="skip fetch, analyze this JSONL")
    args = ap.parse_args()

    if args.analyze_only:
        analyze(args.analyze_only)
        return 0
    key = g.api_key(None)
    if not key:
        print("error: no fish.audio key", file=sys.stderr)
        return 2
    print(f"[catalog] pulling language={args.language} sort={args.sort_by} page_size={args.page_size}...", flush=True)
    got = fetch_all(key, args.language, args.sort_by, args.page_size, args.out, args.sleep, args.max_pages)
    print(f"[catalog] fetched {got} voices -> {args.out}", flush=True)
    analyze(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
