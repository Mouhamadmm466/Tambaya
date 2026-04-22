import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.whisper_service import TranscriptionResult, WhisperService


def _mock_httpx_response(payload: dict, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload)
    response.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    return response


def _mock_httpx_client(response):
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


_GOOD_RESPONSE = {
    "text": "Me ne?",
    "language": "ha",
    "avg_log_prob": -0.5,
    "no_speech_prob": 0.1,
    "duration_ms": 800,
    "is_usable": True,
}


# ---------------------------------------------------------------------------
# WhisperService.transcribe — HTTP client tests
# ---------------------------------------------------------------------------

async def test_transcribe_returns_correct_result():
    svc = WhisperService()
    mock_client = _mock_httpx_client(_mock_httpx_response(_GOOD_RESPONSE))

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await svc.transcribe(b"fake-audio")

    assert result.text == "Me ne?"
    assert result.language == "ha"
    assert result.avg_log_prob == -0.5
    assert result.no_speech_prob == 0.1
    assert result.duration_ms == 800
    assert result.succeeded is True


async def test_transcribe_sends_authorization_header():
    svc = WhisperService()
    mock_client = _mock_httpx_client(_mock_httpx_response(_GOOD_RESPONSE))

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.transcribe(b"fake-audio")

    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-whisper-key"


async def test_transcribe_posts_to_correct_url():
    svc = WhisperService()
    mock_client = _mock_httpx_client(_mock_httpx_response(_GOOD_RESPONSE))

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.transcribe(b"fake-audio")

    args, _ = mock_client.post.call_args
    assert args[0] == "http://localhost:8001/transcribe"


async def test_transcribe_sends_audio_bytes_as_content():
    svc = WhisperService()
    audio = b"real-wav-bytes"
    mock_client = _mock_httpx_client(_mock_httpx_response(_GOOD_RESPONSE))

    with patch("httpx.AsyncClient", return_value=mock_client):
        await svc.transcribe(audio)

    _, kwargs = mock_client.post.call_args
    assert kwargs["content"] == audio


async def test_transcribe_raises_on_http_error():
    svc = WhisperService()
    error_response = _mock_httpx_response({}, status_code=401)
    mock_client = _mock_httpx_client(error_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception):
            await svc.transcribe(b"fake-audio")


# ---------------------------------------------------------------------------
# TranscriptionResult — dataclass tests
# ---------------------------------------------------------------------------

def test_transcription_result_fields():
    r = TranscriptionResult(
        text="Me ne?",
        language="ha",
        avg_log_prob=-0.5,
        no_speech_prob=0.1,
        duration_ms=800,
        succeeded=True,
    )
    assert r.text == "Me ne?"
    assert r.succeeded is True


def test_transcription_result_not_succeeded_when_failed():
    r = TranscriptionResult(
        text="",
        language="ha",
        avg_log_prob=-2.0,
        no_speech_prob=0.9,
        duration_ms=400,
        succeeded=False,
    )
    assert r.succeeded is False
