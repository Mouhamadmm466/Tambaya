from unittest.mock import AsyncMock, patch

import pytest

from agents.router_agent import RouteResult, RouterAgent, _parse_category, router_agent
from database.models import CallCategory


# ---------------------------------------------------------------------------
# _parse_category — pure unit tests (no mocking needed)
# ---------------------------------------------------------------------------

def test_parse_valid_json_health():
    assert _parse_category('{"category": "health"}') == CallCategory.health


def test_parse_valid_json_agriculture():
    assert _parse_category('{"category": "agriculture"}') == CallCategory.agriculture


def test_parse_valid_json_education():
    assert _parse_category('{"category": "education"}') == CallCategory.education


def test_parse_valid_json_general():
    assert _parse_category('{"category": "general"}') == CallCategory.general


def test_parse_valid_json_unclear():
    assert _parse_category('{"category": "unclear"}') == CallCategory.unclear


def test_parse_json_with_surrounding_text():
    assert _parse_category('Here is the answer: {"category": "health"} done.') == CallCategory.health


def test_parse_malformed_json_falls_back_to_substring():
    assert _parse_category('The category is health for this question') == CallCategory.health


def test_parse_unrecognized_category_defaults_to_general():
    assert _parse_category('{"category": "sports"}') == CallCategory.general


def test_parse_empty_string_defaults_to_general():
    assert _parse_category('') == CallCategory.general


def test_parse_gibberish_defaults_to_general():
    assert _parse_category('xyzxyzxyz') == CallCategory.general


# ---------------------------------------------------------------------------
# RouterAgent.classify — integration tests (OllamaService mocked)
# ---------------------------------------------------------------------------

async def test_classify_health_question():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "health"}')):
        result = await router_agent.classify("Ciwon kai yana damuna, zan sha aspirin?")
    assert result.category == CallCategory.health


async def test_classify_agriculture_question():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "agriculture"}')):
        result = await router_agent.classify("Yaushe zan shuka masara a yankin Maradi?")
    assert result.category == CallCategory.agriculture


async def test_classify_education_question():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "education"}')):
        result = await router_agent.classify("Yaya zan shigar da 'ya'yana makaranta?")
    assert result.category == CallCategory.education


async def test_classify_general_question():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "general"}')):
        result = await router_agent.classify("Menene babban birni na Niger?")
    assert result.category == CallCategory.general


async def test_classify_unclear_input():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "unclear"}')):
        result = await router_agent.classify("kdjfk lslsl mmm")
    assert result.category == CallCategory.unclear


async def test_classify_returns_raw_response():
    raw = '{"category": "health"}'
    with patch("agents.router_agent.ollama_service.chat", new=AsyncMock(return_value=raw)):
        result = await router_agent.classify("Ciwon kai")
    assert result.raw_response == raw


async def test_classify_malformed_json_falls_back_via_substring():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value="The answer is agriculture for this farming question")):
        result = await router_agent.classify("Ina masara?")
    assert result.category == CallCategory.agriculture


async def test_classify_ollama_exception_defaults_to_general():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        result = await router_agent.classify("Tambaya")
    assert result.category == CallCategory.general
    assert result.raw_response == ""


async def test_classify_unrecognized_llm_output_defaults_to_general():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value='{"category": "politics"}')):
        result = await router_agent.classify("Wani labari?")
    assert result.category == CallCategory.general


async def test_classify_empty_llm_response_defaults_to_general():
    with patch("agents.router_agent.ollama_service.chat",
               new=AsyncMock(return_value="")):
        result = await router_agent.classify("Tambaya")
    assert result.category == CallCategory.general
