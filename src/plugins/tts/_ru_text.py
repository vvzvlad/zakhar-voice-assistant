"""Russian text adaptation helpers for TTS backends. Each backend
opts into exactly what its engine needs; nothing here runs in the
LLM stage. All functions are pure (no state) and unit-testable."""

import re

# Russian vowels the model may stress with a leading "+" (e.g. "прив+ет").
_RU_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_ACUTE = "́"  # combining acute accent, honored by espeak-ng/Piper

_PLUS_VOWEL_RE = re.compile(rf"\+([{_RU_VOWELS}])")
_STRAY_PLUS_RE = re.compile(rf"\+(?![{_RU_VOWELS}])")
_DOUBLE_SPACE_RE = re.compile(r" {2,}")


def stress_to_acute(text: str) -> str:
    """Model "+vowel" stress notation -> vowel + combining acute
    (U+0301), which espeak-ng/Piper honor. Stray '+' (not before a
    Cyrillic vowel) is dropped so it is never spoken; double spaces
    left by removals are collapsed."""
    text = _PLUS_VOWEL_RE.sub(rf"\1{_ACUTE}", text)
    text = text.replace("+", "")  # drop any stray "+" so it isn't spoken
    return _DOUBLE_SPACE_RE.sub(" ", text)


def sanitize_plus_stress(text: str) -> str:
    """Keep "+vowel" stress pairs (Yandex SpeechKit native notation),
    drop any other stray '+' so it is never spoken; collapse double
    spaces."""
    text = _STRAY_PLUS_RE.sub("", text)
    return _DOUBLE_SPACE_RE.sub(" ", text)


def expand_units(text: str) -> str:
    """Spell out units: "°С"->"градусов", "%"->"процентов", "м/с"->"метров в секунду"."""
    text = text.replace("°С", "градусов")
    text = text.replace("%", "процентов")
    return text.replace("м/с", "метров в секунду")


def phonetic_ru(text: str) -> str:
    """Phonetic spelling hacks for engines that mispronounce these
    words ("что"->"што", "чтобы"->"штобы", "конечно"->"конешно").
    Piper-grade engines only — cloud TTS pronounces the originals fine.
    Run AFTER stress_to_acute: the combining acute trails the vowel,
    so these substring replacements still match stressed words."""
    text = text.replace("что", "што")
    text = text.replace("чтобы", "штобы")
    return text.replace("конечно", "конешно")
