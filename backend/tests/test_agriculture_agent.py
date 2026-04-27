from unittest.mock import AsyncMock, patch

import pytest

from agents.agriculture_agent import HONEST_FALLBACK, AgricultureAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> AgricultureAgent:
    return AgricultureAgent()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_answer_returns_ollama_response():
    agent = _make_agent()
    chunks = ["Gero ana shuka shi a watan Yuni.", "Dawa ana shuka ta a Yuli."]

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(return_value="Ana shuka gero a watan Yuni.")):
        result = await agent.answer("Yaushe zan shuka gero?")

    assert result == "Ana shuka gero a watan Yuni."


async def test_answer_strips_whitespace_from_ollama_response():
    agent = _make_agent()

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(return_value="  Amsa a Hausa.  \n")):
        result = await agent.answer("Tambaya")

    assert result == "Amsa a Hausa."


async def test_answer_includes_all_chunks_in_prompt():
    agent = _make_agent()
    chunks = ["Farko: gero.", "Na biyu: dawa.", "Na uku: wake."]
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_message
        return "Amsa"

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Yaushe zan shuka gero?")

    assert "Farko: gero." in captured["system"]
    assert "Na biyu: dawa." in captured["system"]
    assert "Na uku: wake." in captured["system"]
    assert "Yaushe zan shuka gero?" in captured["user"]


async def test_answer_passes_max_tokens_120_and_json_output_false():
    agent = _make_agent()
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured.update(kwargs)
        return "Amsa"

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya")

    assert captured.get("max_tokens") == 120
    assert captured.get("json_output") is False


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

async def test_answer_returns_fallback_when_chromadb_empty():
    agent = _make_agent()

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_chromadb_raises():
    agent = _make_agent()

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_ollama_raises():
    agent = _make_agent()

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("ollama down"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_ollama_returns_empty():
    agent = _make_agent()

    with patch("agents.agriculture_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.agriculture_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK
