"""Per-profile preferred-voice marker embedded in system-prompt text.

A profile's text may carry ONE single-line marker recording a preferred TTS
voice. It is metadata for the panel + activation flow only and is stripped in
src.prompt.build_system_prompt before the prompt reaches the model.

Syntax (single line, no spaces inside values):
    <<<<<VOICE provider=<id> field=value field=value ...>>>>>
"""
import re

# Single-line by design: the body matches everything up to the next ">>>>>" but
# NOT across a newline, so a stray ">>>>>" later in the prompt cannot swallow
# real content.
_VOICE_RE = re.compile(r"<<<<<VOICE\b(?P<body>[^\n]*?)>>>>>")

# Never apply credentials pasted into prompt text, even if a field name matches.
SECRET_FIELDS = frozenset({"api_key", "token", "password"})


def parse_voice_marker(text):
    """First voice marker as {"provider": str, "fields": {k: v}} or None.

    `fields` is every key=value pair except the special `provider` key. Tokens
    without "=" are ignored; a missing/empty provider yields None."""
    if not text:
        return None
    m = _VOICE_RE.search(text)
    if not m:
        return None
    kv = {}
    for tok in m.group("body").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip()
            if k:
                kv[k] = v.strip()
    provider = kv.pop("provider", "")
    return {"provider": provider, "fields": kv} if provider else None


def strip_voice_marker(text):
    """Remove every voice marker (plus trailing spaces and the EOL it sits on)."""
    if not text:
        return text
    return re.sub(r"[ \t]*<<<<<VOICE\b[^\n]*?>>>>>[ \t]*\n?", "", text)


def build_voice_patch(provider, fields, allowed_fields, variants, current=None):
    """Config patch making `provider` the first TTS variant and overriding the given
    voice fields.

    `variants` is the current tts.variants list ({"provider", "enabled"} dicts): the
    patch puts `provider` first, enabled, and keeps every other variant in its order
    and state (a variant already listed for `provider` is moved, not duplicated).
    `allowed_fields` is the set of the provider's ConfigModel field names; any key
    not in it OR in SECRET_FIELDS is dropped (so a typo or a pasted secret never
    reaches the config). Hidden `*_label` display-companion fields are dropped too:
    they are UI metadata (the chosen voice/model's human name), not voice identity,
    and are never applied from a marker. Instead, an applied field with a `*_label`
    companion empties it when the value differs from `current` (the provider's
    instance config now): the config update merges, so the old voice's name would
    otherwise stay beside the new id, and the panel would show it. An unchanged
    value keeps its name. Provider-only (no safe fields) is valid, e.g. piper.
    Returns {"tts": {"variants": [...], "instances": {provider: {...}}}} or None
    when provider is empty."""
    if not provider:
        return None
    safe = {k: v for k, v in fields.items()
            if k in allowed_fields and k not in SECRET_FIELDS and not k.endswith("_label")}
    cur = current or {}
    safe.update({f"{k}_label": "" for k in list(safe)
                 if f"{k}_label" in allowed_fields and str(cur.get(k, "")) != str(safe[k])})
    order = [{"provider": provider, "enabled": True}]
    order += [v for v in variants if v["provider"] != provider]
    return {"tts": {"variants": order, "instances": {provider: safe}}}
