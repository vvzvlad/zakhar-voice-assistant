"""Smart-home command extraction, parsing, and dispatch."""

import json
import re

import httpx
from loguru import logger

from src.settings import settings


def extract_command_blocks(text: str) -> list[str]:
    """Return list of inner texts for all <command>...</command> blocks.
    Supports multiple blocks and multiline payloads.
    """
    try:
        return re.findall(r"<command>(.*?)</command>", text, flags=re.DOTALL | re.IGNORECASE)
    except re.error as regex_error:
        logger.error(f"Regex error in extract_command_blocks: {str(regex_error)}")
        return []


def parse_command_payload(payload_text: str) -> dict | None:
    """Parse a single command payload into a dictionary.

    Supported formats: Simple pair: device_id:value (e.g., room_light:on or room_ac:22)
    Fallback returns None if nothing matches.
    """
    text = payload_text.strip()

    # Try simple device:value format
    simple_match = re.match(r"^([A-Za-z0-9_\-\.]+)\s*:\s*([A-Za-z0-9_\-\.]+)$", text)
    if simple_match:
        device_id, value = simple_match.group(1), simple_match.group(2)
        return {"device_id": device_id, "value": value}

    return None


async def handle_command(client: httpx.AsyncClient, command_dict: dict) -> None:
    """POST the command to SMARTHOME_URL. Errors are swallowed (logged only)."""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"command": command_dict}
        await client.post(settings.smarthome_url, headers=headers, json=payload, timeout=5)
    except httpx.HTTPError as e:
        logger.error(f"Command handler request error: {str(e)}")
    except (TypeError, ValueError) as e:
        logger.error(f"Command handler error: {str(e)}")


async def process_commands_in_content(client: httpx.AsyncClient, content: str) -> list:
    """Find and process all <command> blocks in the content.

    Returns list of parsed command dicts.
    """
    blocks = extract_command_blocks(content)
    if not blocks:
        return []
    logger.info(f"Found {len(blocks)} command tag(s) in model response")
    parsed_list = []
    for idx, block in enumerate(blocks):
        parsed = parse_command_payload(block)
        logger.info(f"Parsed command #{idx + 1}: {json.dumps(parsed, ensure_ascii=False)}")
        parsed_list.append(parsed)
        await handle_command(client, parsed)
    return parsed_list
