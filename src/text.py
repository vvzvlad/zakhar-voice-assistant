"""Response post-processing."""

import re


def processing_response(response: str) -> str:
    # Remove <think>...</think> and <command>...</command> tags and their response
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<command>.*?</command>', '', response, flags=re.DOTALL)
    response = response.strip()

    response = response.replace("что", "што")
    response = response.replace("чтобы", "штобы")
    response = response.replace("конечно", "конешно")
    response = response.replace("°С", "градусов")
    response = response.replace("%", "процентов")
    response = response.replace("м/с", "метров в секунду")
    return response
