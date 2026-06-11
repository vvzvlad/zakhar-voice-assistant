from src.llm_text import clean_llm_output


def test_strips_think_and_command_blocks():
    raw = "<think>secret reasoning</think>Привет<command>room_light:on</command>"
    assert clean_llm_output(raw) == "Привет"


def test_trims_surrounding_whitespace():
    raw = "   <think>x</think>   ответ   "
    assert clean_llm_output(raw) == "ответ"


def test_tag_removal_double_spaces_collapsed():
    raw = "раз <command>x</command> два"
    assert clean_llm_output(raw) == "раз два"


def test_multiline_think_block_stripped():
    raw = "<think>line one\nline two</think>ответ"
    assert clean_llm_output(raw) == "ответ"


def test_plus_stress_passes_through_untouched():
    # The model's "+vowel" notation is the canonical LLM->TTS contract:
    # the LLM stage must NOT convert or drop it.
    assert clean_llm_output("прив+ет") == "прив+ет"
    assert clean_llm_output("два + два") == "два + два"


def test_units_pass_through_untouched():
    # Unit expansion is a TTS-backend concern, not an LLM-stage one.
    assert clean_llm_output("50% и 5 м/с при 20°С") == "50% и 5 м/с при 20°С"


def test_phonetic_words_pass_through_untouched():
    # "что"->"што" etc. are Piper-specific hacks; the LLM stage keeps originals.
    assert clean_llm_output("что чтобы конечно") == "что чтобы конечно"
