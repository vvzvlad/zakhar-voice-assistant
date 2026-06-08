"""Response post-processing."""

import re


def processing_response(response: str) -> str:
    # Remove <think>...</think> and <command>...</command> tags and their response
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<command>.*?</command>', '', response, flags=re.DOTALL)
    response = response.strip()

    # Russian stress: the model marks the stressed vowel with a "+" placed right
    # BEFORE it (e.g. "прив+ет"). espeak-ng/Piper honor the combining acute accent
    # (U+0301) placed AFTER the vowel, so move the mark onto the vowel. Doing this
    # before the word replacements below keeps those substring replacements working
    # (the combining char trails the vowel and doesn't break "что"->"што" etc.).
    response = re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1́", response)
    response = response.replace("+", "")  # drop any stray "+" so it isn't spoken
    response = re.sub(r" {2,}", " ", response)  # collapse double spaces left by removed "+"

    response = response.replace("что", "што")
    response = response.replace("чтобы", "штобы")
    response = response.replace("конечно", "конешно")
    response = response.replace("°С", "градусов")
    response = response.replace("%", "процентов")
    response = response.replace("м/с", "метров в секунду")
    return response
