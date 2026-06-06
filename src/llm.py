"""Groq chat API client."""

import json

import httpx
from loguru import logger

from src.commands import process_commands_in_content
from src.prompt import build_system_prompt
from src.settings import settings
from src.text import processing_response

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def call_groq_api(
    client_ext: httpx.AsyncClient, client_local: httpx.AsyncClient, text: str
) -> str:
    """Call Groq API with the given text and return plain-text result.

    The external client (proxied) is used for the Groq request and for building the
    system prompt (weather); the local client dispatches smart-home commands.
    On success returns the assistant text.
    On error returns human-readable string starting with "Ошибка: ".
    """
    url = GROQ_API_URL
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.groq_api_key}"}

    payload = {
        "messages": [
            {"role": "system", "content": await build_system_prompt(client_ext)},
            {"role": "user", "content": text}
        ],
        "model": settings.groq_model,
        "temperature": 0.8,
        "max_completion_tokens": 4096,
        "top_p": 0.95,
        "stream": False,
        "reasoning_effort": "medium",
        "stop": None
    }

    try:
        response = await client_ext.post(url, headers=headers, json=payload, timeout=300)
        logger.info(f"Groq API response status: {response.status_code}")

        if response.status_code == 200:
            response_json = response.json()
            logger.info(f"Groq API response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")

            # Extract content from choices[0].message.content
            if 'choices' in response_json and len(response_json['choices']) > 0:
                content = response_json['choices'][0]['message']['content']
                logger.debug(f"Raw content: {content}")

                # Process <command>...</command> blocks before stripping them
                await process_commands_in_content(client_local, content)

                content = processing_response(content)

                logger.debug(f"Cleaned content: {content}")
                return content
            else:
                logger.error("No choices found in Groq API response")
                return f"Ошибка: не найден ответ от модели"
        else:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            # If rate limited, return fixed Russian message
            if response.status_code == 429:
                return (
                    "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии"
                )
            # Try to extract detailed error message
            try:
                err_json = response.json()
                reason_msg = err_json.get("error", {}).get("message")
            except (ValueError, json.JSONDecodeError):
                reason_msg = None
            return f"Ошибка: {reason_msg if reason_msg else error_msg}"

    except httpx.HTTPError as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        return f"Ошибка: {str(e)}"
