"""Shared OpenAI-compatible chat-completions backend (OpenRouter, Groq, etc.)."""

from src.plugins.llm.base import LlmBackend


class OpenAICompatLlmBackend(LlmBackend):
    """One round-trip against an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self, url, api_key, model, temperature, max_tokens, client,
        extra_headers=None, timeout=300,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client
        self.extra_headers = extra_headers
        self.timeout = timeout

    async def complete(self, messages, tools, *, has_next=False):
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Content-Type": "application/json",
            # Keyless local servers get no header: an empty "Bearer " value is illegal
            # and httpx refuses to send the request at all.
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            **(self.extra_headers or {}),
        }
        resp = await self.client.post(
            self.url, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()
