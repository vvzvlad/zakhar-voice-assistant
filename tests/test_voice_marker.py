"""Unit tests for the pure preferred-voice marker module (src.voice_marker)."""

from src.voice_marker import (
    SECRET_FIELDS,
    build_voice_patch,
    parse_voice_marker,
    strip_voice_marker,
)


# --- parse_voice_marker --------------------------------------------------------

def test_parse_single_field():
    out = parse_voice_marker("hi <<<<<VOICE provider=yandex voice=zahar>>>>> bye")
    assert out == {"provider": "yandex", "fields": {"voice": "zahar"}}


def test_parse_multi_field():
    out = parse_voice_marker(
        "<<<<<VOICE provider=fishaudio model=s2-pro reference_id=3b5c9f>>>>>"
    )
    assert out == {
        "provider": "fishaudio",
        "fields": {"model": "s2-pro", "reference_id": "3b5c9f"},
    }


def test_parse_provider_only():
    out = parse_voice_marker("<<<<<VOICE provider=teratts>>>>>")
    assert out == {"provider": "teratts", "fields": {}}


def test_parse_missing_provider_returns_none():
    # A marker with key=value pairs but no provider key is not actionable.
    assert parse_voice_marker("<<<<<VOICE voice=zahar speed=1.2>>>>>") is None


def test_parse_no_marker_returns_none():
    assert parse_voice_marker("just a normal prompt, no marker here") is None
    assert parse_voice_marker("") is None
    assert parse_voice_marker(None) is None


def test_parse_ignores_tokens_without_equals():
    out = parse_voice_marker("<<<<<VOICE provider=yandex garbage voice=zahar>>>>>")
    assert out == {"provider": "yandex", "fields": {"voice": "zahar"}}


def test_parse_does_not_cross_newline():
    # The marker is single-line: an unterminated "<<<<<VOICE" on one line must not
    # swallow a stray ">>>>>" on a later line, so this yields no match.
    text = "<<<<<VOICE provider=yandex\nsome other line >>>>>"
    assert parse_voice_marker(text) is None


def test_parse_stray_close_on_another_line_is_safe():
    # A complete marker is parsed; a stray ">>>>>" on another line is irrelevant.
    text = "<<<<<VOICE provider=yandex voice=zahar>>>>>\nfoo >>>>> bar"
    assert parse_voice_marker(text) == {"provider": "yandex", "fields": {"voice": "zahar"}}


def test_parse_first_marker_wins():
    text = "<<<<<VOICE provider=yandex voice=a>>>>> <<<<<VOICE provider=piper voice=b>>>>>"
    assert parse_voice_marker(text) == {"provider": "yandex", "fields": {"voice": "a"}}


# --- strip_voice_marker --------------------------------------------------------

def test_strip_removes_marker_and_its_eol():
    text = "line one\n<<<<<VOICE provider=yandex voice=zahar>>>>>\nline three\n"
    assert strip_voice_marker(text) == "line one\nline three\n"


def test_strip_keeps_surrounding_text_inline():
    # Marker with no trailing newline: the marker plus the spaces hugging it go,
    # leaving the surrounding words intact.
    text = "BODY <<<<<VOICE provider=yandex voice=zahar>>>>> TAIL"
    assert strip_voice_marker(text) == "BODYTAIL"


def test_strip_removes_multiple_markers():
    text = (
        "<<<<<VOICE provider=yandex voice=a>>>>>\n"
        "middle\n"
        "<<<<<VOICE provider=piper voice_path=x.onnx>>>>>\n"
    )
    assert strip_voice_marker(text) == "middle\n"


def test_strip_absent_marker_unchanged():
    text = "a normal prompt with <<<<<TDW>>>>> only"
    assert strip_voice_marker(text) == text


def test_strip_handles_none_and_empty():
    assert strip_voice_marker("") == ""
    assert strip_voice_marker(None) is None


# --- build_voice_patch ---------------------------------------------------------

def test_build_patch_drops_unknown_fields():
    patch = build_voice_patch(
        "piper", {"voice_path": "x.onnx", "bogus": "1"}, {"voice_path", "sentence_silence"}
    )
    assert patch == {"tts": {"selected": "piper", "instances": {"piper": {"voice_path": "x.onnx"}}}}


def test_build_patch_drops_secret_fields_even_if_allowed():
    # api_key is in SECRET_FIELDS, so it is dropped despite being an allowed field.
    assert "api_key" in SECRET_FIELDS
    patch = build_voice_patch(
        "yandex", {"voice": "zahar", "api_key": "leaked"}, {"voice", "api_key"}
    )
    assert patch == {"tts": {"selected": "yandex", "instances": {"yandex": {"voice": "zahar"}}}}


def test_build_patch_provider_only():
    patch = build_voice_patch("teratts", {}, {"voice", "speed"})
    assert patch == {"tts": {"selected": "teratts", "instances": {"teratts": {}}}}


def test_build_patch_empty_provider_returns_none():
    assert build_voice_patch("", {"voice": "zahar"}, {"voice"}) is None
