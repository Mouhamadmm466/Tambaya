from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import settings as _settings
from services.elevenlabs_service import ElevenLabsService

_FAKE_MP3 = b"ID3\x00\x00\x00fake-mp3-bytes"
_TEST_KEY = "test-xi-api-key-abc123"
_TEST_VOICE = "EXAVITQu4vr4xnSDxMaL"


def _mock_httpx(content: bytes = _FAKE_MP3, status: int = 200):
    response = MagicMock()
    response.content = content
    response.status_code = status
    response.raise_for_status = MagicMock(
        side_effect=None if status < 400 else Exception(f"HTTP {status}")
    )
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _enabled_settings():
    """Temporarily set valid api key + voice id for tests that need them."""
    _settings.elevenlabs_api_key = _TEST_KEY
    _settings.elevenlabs_voice_id = _TEST_VOICE
    _settings.elevenlabs_model = "eleven_multilingual_v2"


def _reset_settings():
    _settings.elevenlabs_api_key = ""
    _settings.elevenlabs_voice_id = ""


# ---------------------------------------------------------------------------
# URL and headers
# ---------------------------------------------------------------------------

async def test_synthesize_posts_to_voice_id_url():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.synthesize("Gero ana shuka shi a watan Yuni.")

    _reset_settings()
    args, _ = mock_client.post.call_args
    assert _TEST_VOICE in args[0]
    assert "text-to-speech" in args[0]


async def test_synthesize_sends_api_key_header():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.synthesize("test")

    _reset_settings()
    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["xi-api-key"] == _TEST_KEY


async def test_synthesize_sends_correct_model():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.synthesize("test")

    _reset_settings()
    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["model_id"] == "eleven_multilingual_v2"


async def test_synthesize_sends_mp3_64k_output_format():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.synthesize("test")

    _reset_settings()
    _, kwargs = mock_client.post.call_args
    assert kwargs["json"]["output_format"] == "mp3_44100_64"


async def test_synthesize_sends_voice_settings():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.synthesize("test")

    _reset_settings()
    _, kwargs = mock_client.post.call_args
    vs = kwargs["json"]["voice_settings"]
    assert vs["stability"] == 0.55
    assert vs["similarity_boost"] == 0.75


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

async def test_synthesize_returns_bytes():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx(content=_FAKE_MP3)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await svc.synthesize("Amsa a Hausa.")

    _reset_settings()
    assert result == _FAKE_MP3
    assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_synthesize_raises_without_api_key():
    svc = ElevenLabsService()
    # conftest sets api_key = "" — no need to clear it here
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        await svc.synthesize("test")


async def test_synthesize_raises_without_voice_id():
    _settings.elevenlabs_api_key = _TEST_KEY
    _settings.elevenlabs_voice_id = ""
    svc = ElevenLabsService()

    with pytest.raises(ValueError, match="ELEVENLABS_VOICE_ID"):
        await svc.synthesize("test")

    _settings.elevenlabs_api_key = ""


async def test_synthesize_raises_on_http_error():
    _enabled_settings()
    svc = ElevenLabsService()
    mock_client = _mock_httpx(status=401)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception):
            await svc.synthesize("test")

    _reset_settings()
