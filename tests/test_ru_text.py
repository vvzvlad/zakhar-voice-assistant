"""Tests for the shared Russian TTS text helpers (src/plugins/tts/_ru_text.py)."""

from src.plugins.tts._ru_text import (
    drop_plus_stress,
    expand_units,
    phonetic_ru,
    sanitize_plus_stress,
    stress_to_acute,
    stress_to_uppercase,
)


def _piper_chain(text: str) -> str:
    """The exact adaptation chain Piper/TeraTTS backends apply."""
    return phonetic_ru(expand_units(stress_to_acute(text)))


# --- stress_to_acute ---------------------------------------------------------

def test_stress_to_acute_single_word():
    # "+" before the stressed vowel becomes a combining acute accent (U+0301) after it.
    assert stress_to_acute("прив+ет") == "приве́т"


def test_stress_to_acute_multiple_words():
    assert stress_to_acute("больш+ая к+омната") == "больша́я ко́мната"


def test_stress_to_acute_stray_plus_dropped():
    # A "+" not preceding a Cyrillic vowel is dropped so it isn't spoken,
    # and the double space it leaves is collapsed.
    assert stress_to_acute("два + два") == "два два"
    assert "+" not in stress_to_acute("два + два")


def test_stress_to_acute_passthrough():
    assert stress_to_acute("просто текст") == "просто текст"


# --- stress_to_uppercase (neural engines that strip "+"/acute) ---------------

def test_stress_to_uppercase_single_word():
    # "+" before the stressed vowel disappears and the vowel is upper-cased.
    assert stress_to_uppercase("прив+ет") == "привЕт"


def test_stress_to_uppercase_multiple_words():
    assert stress_to_uppercase("больш+ая к+омната") == "большАя кОмната"


def test_stress_to_uppercase_stray_plus_dropped():
    # A "+" not preceding a Cyrillic vowel is dropped so it isn't spoken,
    # and the double space it leaves is collapsed.
    out = stress_to_uppercase("два + два")
    assert "+" not in out
    assert out == "два два"


def test_stress_to_uppercase_yo_edge():
    # "ё" upper-cases to "Ё" (start-of-word stress).
    assert stress_to_uppercase("+ёлка") == "Ёлка"


def test_stress_to_uppercase_passthrough():
    assert stress_to_uppercase("просто текст") == "просто текст"


# --- sanitize_plus_stress (Yandex native notation) ---------------------------

def test_sanitize_plus_stress_keeps_vowel_pairs():
    assert sanitize_plus_stress("прив+ет") == "прив+ет"
    assert sanitize_plus_stress("больш+ая к+омната") == "больш+ая к+омната"


def test_sanitize_plus_stress_drops_stray_plus():
    out = sanitize_plus_stress("два + два")
    assert "+" not in out
    assert out == "два два"  # double space collapsed


def test_sanitize_plus_stress_passthrough():
    assert sanitize_plus_stress("просто текст") == "просто текст"


# --- drop_plus_stress (plain-text engines, e.g. Fish Audio) ------------------

def test_drop_plus_stress_removes_vowel_markup():
    # The "+vowel" pair loses its '+' entirely; the plain vowel remains.
    assert drop_plus_stress("прив+ет") == "привет"
    assert drop_plus_stress("больш+ая к+омната") == "большая комната"


def test_drop_plus_stress_drops_stray_plus_and_collapses_spaces():
    out = drop_plus_stress("два + два")
    assert "+" not in out
    assert out == "два два"  # double space collapsed


def test_drop_plus_stress_passthrough():
    assert drop_plus_stress("просто текст") == "просто текст"


# --- expand_units -------------------------------------------------------------

def test_expand_units_percent():
    assert expand_units("99%") == "99процентов"


def test_expand_units_wind_speed():
    assert expand_units("3 м/с") == "3 метров в секунду"


def test_expand_units_degrees():
    assert expand_units("20°С") == "20градусов"


# --- phonetic_ru ---------------------------------------------------------------

def test_phonetic_ru_replacements():
    out = phonetic_ru("что чтобы конечно")
    assert "што" in out
    assert "штобы" in out
    assert "конешно" in out
    assert "что" not in out
    assert "конечно" not in out


# --- backend chains -------------------------------------------------------------

def test_piper_chain_stress_then_phonetic():
    # Stress conversion runs first, so "что"->"што" still matches the stressed word
    # (the combining acute trails the vowel and doesn't break the substring match).
    assert _piper_chain("чт+о") == "што́"


def test_piper_chain_full_processing():
    out = _piper_chain("прив+ет, что нового? 50% и 3 м/с")
    assert "приве́т" in out
    assert "што" in out
    assert "%" not in out
    assert "процентов" in out
    assert "метров в секунду" in out
    assert "+" not in out


def test_plain_fallback_phrase_unchanged_by_both_chains():
    # Spoken fallback phrases (reply_error etc.) carry no markup/units and must
    # pass through both backend chains byte-for-byte.
    phrase = "Захворал я, барин. Голова не варит."
    assert _piper_chain(phrase) == phrase
    assert sanitize_plus_stress(expand_units(phrase)) == phrase
