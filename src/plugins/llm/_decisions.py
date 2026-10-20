"""OpenRouter Decisions API pieces shared by the jev handler (src/plugins/llm/jev.py)
and the jev follow-up check (src/plugins/followup/jev.py), so neither provider
imports the other."""

# OpenRouter Decisions API: typed answers (choice / noul) with probabilities, shared by
# every decision model OpenRouter serves.
DECISIONS_API_URL = "https://openrouter.ai/api/alpha/decisions"


def noul_answer(answers: dict, key: str) -> float:
    """The probability of the noul answer `key`. Raises KeyError, TypeError or
    ValueError when the answer is missing or malformed."""
    answer = answers[key]
    if answer["type"] != "noul":
        raise ValueError(f"{key}: not a noul answer: {answer!r}")
    return float(answer["noul"])
