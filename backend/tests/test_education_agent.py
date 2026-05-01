from unittest.mock import AsyncMock, patch

import pytest

from agents.education_agent import HONEST_FALLBACK, EducationAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> EducationAgent:
    return EducationAgent()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_answer_returns_ollama_response():
    agent = _make_agent()
    chunks = ["Ilimi na firamare kyauta ne a Niger.", "Ana shiga makaranta a shekara 7."]

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(return_value="Ana shiga makaranta ta firamare a shekara 7.")):
        result = await agent.answer("Yaushe ake fara makaranta?")

    assert result == "Ana shiga makaranta ta firamare a shekara 7."


async def test_answer_strips_whitespace_from_ollama_response():
    agent = _make_agent()

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(return_value="  Amsa a Hausa.  \n")):
        result = await agent.answer("Tambaya")

    assert result == "Amsa a Hausa."


async def test_answer_includes_all_chunks_in_prompt():
    agent = _make_agent()
    chunks = ["Farko: firamare.", "Na biyu: kolej.", "Na uku: lise."]
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_message
        return "Amsa"

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Menene tsarin makaranta a Niger?")

    assert "Farko: firamare." in captured["system"]
    assert "Na biyu: kolej." in captured["system"]
    assert "Na uku: lise." in captured["system"]
    assert "Menene tsarin makaranta a Niger?" in captured["user"]


async def test_answer_passes_max_tokens_120_and_json_output_false():
    agent = _make_agent()
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured.update(kwargs)
        return "Amsa"

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya")

    assert captured.get("max_tokens") == 120
    assert captured.get("json_output") is False


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

async def test_answer_returns_fallback_when_chromadb_empty():
    agent = _make_agent()

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_chromadb_raises():
    agent = _make_agent()

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_ollama_raises():
    agent = _make_agent()

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("ollama down"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_fallback_when_ollama_returns_empty():
    agent = _make_agent()

    with patch("agents.education_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.education_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


# ---------------------------------------------------------------------------
# Fallback directs to education office — not a generic "I don't know"
# ---------------------------------------------------------------------------

async def test_honest_fallback_mentions_education_office():
    """Fallback must direct the caller to their local education office."""
    assert "ofishin ilimi" in HONEST_FALLBACK, (
        "HONEST_FALLBACK must direct caller to the education office"
    )
