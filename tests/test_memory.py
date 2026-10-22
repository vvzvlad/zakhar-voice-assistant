"""Tests for the assistant's memory: MemoryStore over data/memory.json (src/memory.py)
and its remember/forget tools (src/builtin_mcp/memory.py), driven through
BuiltinMcpSource.call like the LLM's tool calls."""

import asyncio

from src.builtin_mcp.memory import build_memory_server
from src.memory import MemoryStore
from src.tool_hub import BuiltinMcpSource


def test_a_missing_file_is_no_facts(tmp_path):
    assert MemoryStore(str(tmp_path / "memory.json")).facts() == []


def test_facts_persist_across_stores_on_the_same_file(tmp_path):
    path = str(tmp_path / "memory.json")
    store = MemoryStore(path)
    assert store.add("  Пользователь — мужчина ") is True
    assert store.add("Кота зовут Барсик") is True
    assert store.add("пользователь — мужчина") is False   # already saved, ignoring case
    assert MemoryStore(path).facts() == ["Пользователь — мужчина", "Кота зовут Барсик"]

    assert store.remove("- кота зовут барсик ") is True    # as listed in the prompt, any case
    assert store.remove("Собаку зовут Шарик") is False
    assert MemoryStore(path).facts() == ["Пользователь — мужчина"]


def call(store, tool, fact):
    async def main():
        source = BuiltinMcpSource("memory", build_memory_server(store))
        await source.start()
        return await source.call(tool, {"fact": fact})

    return asyncio.run(main())


def test_remember_then_forget_through_the_tools(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    assert call(store, "remember", "Пользователь — мужчина") == "Запомнил: Пользователь — мужчина"
    assert call(store, "remember", "Пользователь — мужчина") == "Это уже запомнено."
    assert store.facts() == ["Пользователь — мужчина"]
    assert call(store, "forget", "Пользователь — мужчина") == "Забыл: Пользователь — мужчина"
    assert call(store, "forget", "Пользователь — мужчина") == "Такого факта в памяти нет."
    assert store.facts() == []


def test_a_failing_store_is_a_russian_answer_not_an_exception(tmp_path):
    # The memory path is a directory: reading it fails, the tool answers with the error.
    store = MemoryStore(str(tmp_path))
    assert call(store, "remember", "x").startswith("Не удалось запомнить: ")
    assert call(store, "forget", "x").startswith("Не удалось забыть: ")
