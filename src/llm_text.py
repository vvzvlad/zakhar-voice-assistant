"""LLM output cleaning. Stage-agnostic: stress marks ('+') and units
pass through untouched — adapting text to a speech engine is each TTS
backend's own job (see src/plugins/tts/_ru_text.py)."""

import re


def clean_llm_output(text: str) -> str:
    """Strip <think>/<command> blocks and trim. Keeps the model's
    '+stress' notation intact (the canonical LLM->TTS text contract)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<command>.*?</command>', '', text, flags=re.DOTALL)
    text = text.strip()
    # Tag removal can leave double spaces; collapse them.
    return re.sub(r" {2,}", " ", text)
