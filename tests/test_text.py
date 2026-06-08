from src.text import processing_response


def test_strips_think_and_command_blocks():
    raw = "<think>secret reasoning</think>Привет<command>room_light:on</command>"
    assert processing_response(raw) == "Привет"


def test_trims_surrounding_whitespace():
    raw = "   <think>x</think>   ответ   "
    assert processing_response(raw) == "ответ"


def test_phonetic_and_symbol_replacements():
    raw = "что чтобы конечно 50% 5 м/с 20°С"
    out = processing_response(raw)
    # что → што (note: "чтобы" → "штобы" then "конечно" → "конешно")
    assert "што" in out
    assert "штобы" in out
    assert "конешно" in out
    # Symbol replacements.
    assert "%" not in out
    assert "процентов" in out
    assert "м/с" not in out
    assert "метров в секунду" in out
    assert "°С" not in out
    assert "градусов" in out
    # No leftover source tokens.
    assert "что " not in out
    assert "чтобы" not in out
    assert "конечно" not in out


def test_percent_replacement_isolated():
    assert processing_response("99%") == "99процентов"


def test_wind_replacement_isolated():
    assert processing_response("3 м/с") == "3 метров в секунду"


def test_stress_mark_moved_onto_vowel():
    # "+" before the stressed vowel becomes a combining acute accent (U+0301) after it.
    assert processing_response("прив+ет") == "приве́т"


def test_stray_plus_removed():
    # A "+" not preceding a vowel is dropped so it isn't spoken.
    assert "+" not in processing_response("два + два")


def test_stress_then_word_replacement():
    # Stress conversion runs first, so "что"->"што" still matches afterwards.
    assert processing_response("чт+о") == "што́"


def test_stress_multiple_words():
    assert processing_response("больш+ая к+омната") == "больша́я ко́мната"
