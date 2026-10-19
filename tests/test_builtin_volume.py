"""Tests for the built-in volume tool (src/builtin_mcp/volume.py), driven through
BuiltinMcpSource.call like the LLM's tool calls: it acts on the speaker in the
current_device ContextVar and answers in Russian, never raising."""

import asyncio

from src.builtin_mcp.volume import build_volume_server
from src.run_context import current_device
from src.tool_hub import BuiltinMcpSource


class FakeManager:
    """DeviceManager double: records get/set_volume calls, answers 40 or raises `error`."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def get_volume(self, device_name):
        self.calls.append((device_name,))
        if self.error is not None:
            raise self.error
        return 40

    def set_volume(self, device_name, level=None, change=None):
        self.calls.append((device_name, level, change))
        if self.error is not None:
            raise self.error
        return 40


def call(manager, args, device, tool="set_volume"):
    token = current_device.set(device)
    try:
        async def main():
            source = BuiltinMcpSource("volume", build_volume_server(lambda: manager))
            await source.start()
            return await source.call(tool, args)

        return asyncio.run(main())
    finally:
        current_device.reset(token)


def test_set_volume_acts_on_the_speaker_the_phrase_came_from():
    manager = FakeManager()
    assert call(manager, {"change": -10}, "kitchen") == "Громкость колонки: 40%."
    assert call(manager, {"level": 40}, "hall") == "Громкость колонки: 40%."
    assert manager.calls == [("kitchen", None, -10), ("hall", 40, None)]


def test_set_volume_failure_is_a_russian_message():
    for error in (LookupError("unknown device 'kitchen'"), RuntimeError("kitchen is offline")):
        out = call(FakeManager(error), {"level": 30}, "kitchen")
        assert out == f"Не удалось изменить громкость: {error}"


def test_set_volume_without_a_speaker_changes_nothing():
    manager = FakeManager()
    out = call(manager, {"level": 30}, None)
    assert out.startswith("Не знаю, с какой колонки")
    assert manager.calls == []


def test_get_volume_reads_the_speaker_the_phrase_came_from():
    manager = FakeManager()
    assert call(manager, {}, "kitchen", "get_volume") == "Громкость колонки: 40%."
    assert manager.calls == [("kitchen",)]
    out = call(FakeManager(RuntimeError("kitchen is offline")), {}, "kitchen", "get_volume")
    assert out == "Не удалось узнать громкость: kitchen is offline"
    assert call(manager, {}, None, "get_volume").startswith("Не знаю, с какой колонки")
