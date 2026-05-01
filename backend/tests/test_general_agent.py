from unittest.mock import AsyncMock, patch

import pytest

from agents.general_agent import HONEST_FALLBACK, GeneralAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> GeneralAgent:
    return GeneralAgent()


# ---------------------------------------------------------------------------
# Stage 1: RAG path (ChromaDB has chunks)
# ---------------------------------------------------------------------------

async def test_rag_path_returns_ollama_response():
    agent = _make_agent()
    chunks = ["Niamey ita ce babban birnin Niger.", "Niger tana da yankuna takwas."]

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(return_value="Niamey ita ce babban birnin Niger.")):
        result = await agent.answer("Menene babban birnin Niger?")

    assert result == "Niamey ita ce babban birnin Niger."


async def test_rag_path_strips_whitespace():
    agent = _make_agent()

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(return_value="  Amsa a Hausa.  \n")):
        result = await agent.answer("Tambaya")

    assert result == "Amsa a Hausa."


async def test_rag_path_includes_all_chunks_in_prompt():
    agent = _make_agent()
    chunks = ["Farko: geography.", "Na biyu: gwamnati.", "Na uku: kasuwa."]
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_message
        return "Amsa"

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=chunks)), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya kan Niger?")

    assert "Farko: geography." in captured["system"]
    assert "Na biyu: gwamnati." in captured["system"]
    assert "Na uku: kasuwa." in captured["system"]
    assert "Tambaya kan Niger?" in captured["user"]


async def test_rag_path_passes_max_tokens_120_and_json_output_false():
    agent = _make_agent()
    captured: dict = {}

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured.update(kwargs)
        return "Amsa"

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya")

    assert captured.get("max_tokens") == 120
    assert captured.get("json_output") is False


# ---------------------------------------------------------------------------
# Stage 2: open prompt path (ChromaDB empty — unique to GeneralAgent)
# ---------------------------------------------------------------------------

async def test_falls_through_to_open_prompt_when_chromadb_empty():
    """When ChromaDB returns no chunks, GeneralAgent tries Gemma 4 directly."""
    agent = _make_agent()
    open_answer = "Wannan tambaya tana da amsa mai sauƙi."

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(return_value=open_answer)):
        result = await agent.answer("Tambaya gama-gari")

    assert result == open_answer


async def test_falls_through_to_open_prompt_when_chromadb_raises():
    """ChromaDB failure is logged and open prompt is tried."""
    agent = _make_agent()
    open_answer = "Amsar Gemma 4 kai tsaye."

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(side_effect=Exception("connection refused"))), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(return_value=open_answer)):
        result = await agent.answer("Tambaya")

    assert result == open_answer


async def test_falls_through_to_open_prompt_when_rag_ollama_returns_empty():
    """Empty RAG answer falls through to open prompt, not HONEST_FALLBACK."""
    agent = _make_agent()
    open_answer = "Amsar ta biyu."
    call_count = 0

    async def mock_chat(system_prompt, user_message, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call (RAG) returns empty, second call (open) returns answer
        return "" if call_count == 1 else open_answer

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        result = await agent.answer("Tambaya")

    assert result == open_answer
    assert call_count == 2


async def test_open_prompt_uses_different_system_prompt_than_rag():
    """Stage 2 uses the open prompt, not the RAG prompt with {chunks}."""
    agent = _make_agent()
    captured_prompts: list[str] = []

    async def mock_chat(system_prompt, user_message, **kwargs):
        captured_prompts.append(system_prompt)
        return "" if len(captured_prompts) == 1 else "Amsa"

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=mock_chat)):
        await agent.answer("Tambaya")

    assert len(captured_prompts) == 2
    # RAG prompt contains "Retrieved knowledge"
    assert "Retrieved knowledge" in captured_prompts[0]
    # Open prompt does NOT contain "Retrieved knowledge"
    assert "Retrieved knowledge" not in captured_prompts[1]


# ---------------------------------------------------------------------------
# Final fallback — both stages fail
# ---------------------------------------------------------------------------

async def test_returns_honest_fallback_when_both_stages_fail():
    """If ChromaDB is empty AND Gemma 4 raises, return HONEST_FALLBACK."""
    agent = _make_agent()

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("ollama down"))):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_returns_honest_fallback_when_open_ollama_returns_empty():
    """Empty open-prompt response returns HONEST_FALLBACK."""
    agent = _make_agent()

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=[])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK


async def test_returns_honest_fallback_when_rag_ollama_raises_and_open_also_raises():
    agent = _make_agent()
    call_count = 0

    async def always_raise(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise Exception("ollama down")

    with patch("agents.general_agent.chromadb_service.query",
               new=AsyncMock(return_value=["chunk"])), \
         patch("agents.general_agent.ollama_service.chat",
               new=AsyncMock(side_effect=always_raise)):
        result = await agent.answer("Tambaya")

    assert result == HONEST_FALLBACK
    assert call_count == 2  # tried both RAG and open prompt
