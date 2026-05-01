from unittest.mock import AsyncMock, patch

import pytest

from agents.health_agent import DOCTOR_SUFFIX, HONEST_FALLBACK, HealthAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> HealthAgent:
    return HealthAgent()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_answer_returns_ollama_response_with_doctor_suffix():
    agent = _make_agent()
    chunks = ["Malaria tana haifar da zazzaɓi.", "Ana ba da magani kyauta a CSCOM."]

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(return_value="Ka sha maganin ACT kuma ka huta.")):
        result = await agent.answer("Ina da zazzaɓi, me zan yi?")

    assert result == f"Ka sha maganin ACT kuma ka huta. {DOCTOR_SUFFIX}"


async def test_answer_strips_whitespace_before_appending_suffix():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(return_value="  Amsa a Hausa.  \n")):
        result = await agent.answer("Tambaya")

    assert result == f"Amsa a Hausa. {DOCTOR_SUFFIX}"


async def test_answer_includes_all_chunks_in_prompt():
    agent = _make_agent()
    chunks = ["Farko: malaria.", "Na biyu: diarrhea.", "Na uku: rashin abinci."]
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_message
        return "Amsa"

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Yaro yana da zazzaɓi?")

    assert "Farko: malaria." in captured["system"]
    assert "Na biyu: diarrhea." in captured["system"]
    assert "Na uku: rashin abinci." in captured["system"]
    assert "Yaro yana da zazzaɓi?" in captured["user"]


async def test_answer_passes_max_tokens_120_and_json_output_false():
    agent = _make_agent()
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured.update(kwargs)
        return "Amsa"

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya")

    assert captured.get("max_tokens") == 120
    assert captured.get("json_output") is False


# ---------------------------------------------------------------------------
# Non-negotiable: doctor suffix ALWAYS present (NAMU_CONTEXT.md §10 rule 4)
# ---------------------------------------------------------------------------

async def test_doctor_suffix_always_appended_to_normal_answer():
    """Core non-negotiable: every health answer ends with the doctor recommendation."""
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(return_value="Sha ruwa mai yawa.")):
        result = await agent.answer("Me zan yi kan zazzaɓi?")

    assert result.endswith(DOCTOR_SUFFIX), (
        f"Doctor suffix missing from answer: {result!r}"
    )


async def test_doctor_suffix_in_honest_fallback():
    """Fallback message must also contain the doctor recommendation."""
    assert DOCTOR_SUFFIX in HONEST_FALLBACK, (
        "HONEST_FALLBACK must always contain the doctor recommendation"
    )


async def test_fallback_returned_when_chromadb_empty_contains_doctor_advice():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])):
        result = await agent.answer("Tambaya")

    assert DOCTOR_SUFFIX in result


async def test_fallback_returned_when_chromadb_raises_contains_doctor_advice():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        result = await agent.answer("Tambaya")

    assert DOCTOR_SUFFIX in result


async def test_fallback_returned_when_ollama_raises_contains_doctor_advice():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("ollama down"))):
        result = await agent.answer("Tambaya")

    assert DOCTOR_SUFFIX in result


async def test_fallback_returned_when_ollama_empty_contains_doctor_advice():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await agent.answer("Tambaya")

    assert DOCTOR_SUFFIX in result


# ---------------------------------------------------------------------------
# Fallback behaviour — return values
# ---------------------------------------------------------------------------

async def test_answer_returns_honest_fallback_when_chromadb_empty():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_honest_fallback_when_chromadb_raises():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_honest_fallback_when_ollama_raises():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("ollama down"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_answer_returns_honest_fallback_when_ollama_returns_empty():
    agent = _make_agent()

    with patch("agents.health_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.health_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK
