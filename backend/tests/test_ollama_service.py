from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.ollama_service import OllamaService


def _mock_httpx_client(response_payload: dict, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=response_payload)
    response.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


_GOOD_RESPONSE = {
    "message": {
        "role": "assistant",
        "content": '{"category": "health"}',
    }
}


# ---------------------------------------------------------------------------
# OllamaService.chat — HTTP client tests
# ---------------------------------------------------------------------------

async def test_chat_posts_to_correct_url():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.chat("system", "user message")

    args, _ = mock_client.post.call_args
    assert args[0] == "http://localhost:11434/api/chat"


async def test_chat_sends_correct_model():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.chat("system", "user message")

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["model"] == "gemma4:4b"


async def test_chat_sends_stream_false():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.chat("system", "user message")

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["stream"] is False


async def test_chat_sends_temperature_zero():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.chat("system", "user message")

    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["options"]["temperature"] == 0


async def test_chat_sends_system_and_user_messages():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.chat("my system prompt", "my user question")

    _, kwargs = mock_client.post.call_args
    messages = kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "my system prompt"}
    assert messages[1] == {"role": "user", "content": "my user question"}


async def test_chat_returns_message_content():
    svc = OllamaService()
    mock_client = _mock_httpx_client(_GOOD_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await svc.chat("system", "user message")

    assert result == '{"category": "health"}'


async def test_chat_raises_on_http_error():
    svc = OllamaService()
    mock_client = _mock_httpx_client({}, status_code=503)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception):
            await svc.chat("system", "user message")
