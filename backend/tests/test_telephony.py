import hashlib
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.telephony_service import (
    build_fallback_xml,
    build_poor_quality_xml,
    build_transcription_response_xml,
    build_voice_response_xml,
    download_recording,
    hash_phone_number,
)
from services.whisper_service import TranscriptionResult

VALID_TOKEN = "test-webhook-token"
AT_INITIATED_FORM = {
    "sessionId": "ATVId_test123",
    "callerNumber": "+22790000000",
    "callSessionState": "Initiated",
    "isActive": "1",
}


# ---------------------------------------------------------------------------
# hash_phone_number — pure unit tests
# ---------------------------------------------------------------------------

def test_hash_is_64_char_hex():
    result = hash_phone_number("+22790000000")
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_hash_is_deterministic():
    assert hash_phone_number("+22790000000") == hash_phone_number("+22790000000")


def test_different_numbers_produce_different_hashes():
    assert hash_phone_number("+22790000001") != hash_phone_number("+22790000002")


def test_hash_matches_sha256_directly():
    number = "+22790000000"
    expected = hashlib.sha256(number.encode()).hexdigest()
    assert hash_phone_number(number) == expected


# ---------------------------------------------------------------------------
# build_voice_response_xml — pure unit tests
# ---------------------------------------------------------------------------

def test_voice_xml_contains_say():
    xml = build_voice_response_xml(VALID_TOKEN)
    assert "<Say" in xml
    assert "Welcome to Namu Tambaya" in xml


def test_voice_xml_contains_record():
    xml = build_voice_response_xml(VALID_TOKEN)
    assert "<Record" in xml


def test_voice_xml_record_has_token_in_callback_url():
    xml = build_voice_response_xml(VALID_TOKEN)
    assert "/api/telephony/recording" in xml
    assert VALID_TOKEN in xml


def test_voice_xml_record_parameters():
    xml = build_voice_response_xml(VALID_TOKEN)
    assert 'maxLength="60"' in xml
    assert 'trimSilence="true"' in xml
    assert 'playBeep="true"' in xml


def test_voice_xml_is_parseable():
    xml = build_voice_response_xml(VALID_TOKEN)
    body = xml.split("?>", 1)[-1]  # strip XML declaration
    ET.fromstring(body)             # raises on invalid XML


# ---------------------------------------------------------------------------
# build_fallback_xml — pure unit tests
# ---------------------------------------------------------------------------

def test_fallback_xml_contains_say():
    xml = build_fallback_xml()
    assert "<Say" in xml


def test_fallback_xml_mentions_unavailable():
    xml = build_fallback_xml()
    assert "unavailable" in xml.lower()


def test_fallback_xml_is_parseable():
    xml = build_fallback_xml()
    body = xml.split("?>", 1)[-1]
    ET.fromstring(body)


# ---------------------------------------------------------------------------
# build_transcription_response_xml — pure unit tests
# ---------------------------------------------------------------------------

def test_transcription_xml_contains_transcript():
    xml = build_transcription_response_xml("Me ne?")
    assert "Me ne?" in xml
    assert "<Say" in xml


def test_transcription_xml_is_parseable():
    xml = build_transcription_response_xml("Me ne?")
    body = xml.split("?>", 1)[-1]
    ET.fromstring(body)


# ---------------------------------------------------------------------------
# build_poor_quality_xml — pure unit tests
# ---------------------------------------------------------------------------

def test_poor_quality_xml_contains_say():
    xml = build_poor_quality_xml()
    assert "<Say" in xml


def test_poor_quality_xml_is_parseable():
    xml = build_poor_quality_xml()
    body = xml.split("?>", 1)[-1]
    ET.fromstring(body)


# ---------------------------------------------------------------------------
# voice_webhook router — integration tests (DB mocked)
# ---------------------------------------------------------------------------

async def test_voice_webhook_no_token_returns_403(client):
    resp = await client.post("/api/telephony/voice", data=AT_INITIATED_FORM)
    assert resp.status_code == 403


async def test_voice_webhook_wrong_token_returns_403(client):
    resp = await client.post("/api/telephony/voice?token=wrong-token", data=AT_INITIATED_FORM)
    assert resp.status_code == 403


def _mock_db_session():
    """Returns a properly typed mock DB session.
    db.add() is synchronous in SQLAlchemy — must be MagicMock, not AsyncMock.
    db.commit() is async — must be AsyncMock.
    db.execute() is async — AsyncMock by default.
    """
    session = AsyncMock()
    session.add = MagicMock()
    return session


async def test_voice_webhook_returns_200_xml(client):
    with patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_sl.return_value.__aenter__ = AsyncMock(return_value=_mock_db_session())
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = await client.post(
            f"/api/telephony/voice?token={VALID_TOKEN}",
            data=AT_INITIATED_FORM,
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]


async def test_voice_webhook_response_contains_say_and_record(client):
    with patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_sl.return_value.__aenter__ = AsyncMock(return_value=_mock_db_session())
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = await client.post(
            f"/api/telephony/voice?token={VALID_TOKEN}",
            data=AT_INITIATED_FORM,
        )

    assert "<Say" in resp.text
    assert "<Record" in resp.text


async def test_voice_webhook_logs_call_to_db(client):
    with patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_session = _mock_db_session()
        mock_sl.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        await client.post(
            f"/api/telephony/voice?token={VALID_TOKEN}",
            data=AT_INITIATED_FORM,
        )

    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


async def test_voice_webhook_db_error_caller_still_gets_greeting(client):
    """DB failure is caught and logged — the caller still receives the greeting XML.
    The call must never be dropped due to a logging failure.
    """
    with patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_sl.side_effect = Exception("DB connection refused")

        resp = await client.post(
            f"/api/telephony/voice?token={VALID_TOKEN}",
            data=AT_INITIATED_FORM,
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]
    assert "<Say" in resp.text
    assert "<Record" in resp.text


async def test_voice_webhook_non_initiated_state_returns_xml(client):
    """AT sends other states (Active, Completed); we respond with XML for all of them."""
    with patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_sl.return_value.__aenter__ = AsyncMock(return_value=_mock_db_session())
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = await client.post(
            f"/api/telephony/voice?token={VALID_TOKEN}",
            data={**AT_INITIATED_FORM, "callSessionState": "Completed"},
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# recording_callback router — integration tests (httpx + whisper mocked)
# ---------------------------------------------------------------------------

def _good_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="Me ne?",
        language="ha",
        avg_log_prob=-0.5,
        no_speech_prob=0.1,
        duration_ms=800,
        succeeded=True,
    )


def _poor_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="",
        language="ha",
        avg_log_prob=-2.0,
        no_speech_prob=0.9,
        duration_ms=400,
        succeeded=False,
    )


def _mock_httpx_download(content: bytes = b"fake-wav-audio-data"):
    mock_response = MagicMock()
    mock_response.content = content
    mock_response.raise_for_status = MagicMock()
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    return mock_http


async def test_recording_callback_no_token_returns_403(client):
    resp = await client.post(
        "/api/telephony/recording",
        data={"sessionId": "ATVId_test123", "recordingUrl": "https://voice.at.com/test.wav"},
    )
    assert resp.status_code == 403


async def test_recording_callback_wrong_token_returns_403(client):
    resp = await client.post(
        "/api/telephony/recording?token=wrong",
        data={"sessionId": "ATVId_test123", "recordingUrl": "https://voice.at.com/test.wav"},
    )
    assert resp.status_code == 403


async def test_recording_callback_downloads_audio_and_transcribes(client):
    mock_http = _mock_httpx_download()

    with patch("httpx.AsyncClient", return_value=mock_http), \
         patch("routers.telephony.whisper_service.transcribe",
               new=AsyncMock(return_value=_good_transcription_result())):
        resp = await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "callerNumber": "+22790000000",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
                "durationInSeconds": "8",
            },
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]
    mock_http.get.assert_awaited_once_with(
        "https://voice.africastalking.com/recordings/test.wav"
    )


async def test_recording_callback_good_transcription_returns_transcript_xml(client):
    mock_http = _mock_httpx_download()

    with patch("httpx.AsyncClient", return_value=mock_http), \
         patch("routers.telephony.whisper_service.transcribe",
               new=AsyncMock(return_value=_good_transcription_result())):
        resp = await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
            },
        )

    assert "Me ne?" in resp.text


async def test_recording_callback_poor_quality_returns_poor_quality_xml(client):
    mock_http = _mock_httpx_download()

    with patch("httpx.AsyncClient", return_value=mock_http), \
         patch("routers.telephony.whisper_service.transcribe",
               new=AsyncMock(return_value=_poor_transcription_result())):
        resp = await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
            },
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]
    assert "clearly" in resp.text.lower()


async def test_recording_callback_transcription_exception_returns_fallback_xml(client):
    """Whisper service error is caught — caller still gets a response."""
    mock_http = _mock_httpx_download()

    with patch("httpx.AsyncClient", return_value=mock_http), \
         patch("routers.telephony.whisper_service.transcribe",
               new=AsyncMock(side_effect=Exception("whisper timeout"))):
        resp = await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
            },
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]
    assert "unavailable" in resp.text.lower()


async def test_recording_callback_updates_db_with_transcription_metadata(client):
    mock_http = _mock_httpx_download()
    mock_session = _mock_db_session()

    with patch("httpx.AsyncClient", return_value=mock_http), \
         patch("routers.telephony.whisper_service.transcribe",
               new=AsyncMock(return_value=_good_transcription_result())), \
         patch("routers.telephony.AsyncSessionLocal") as mock_sl:
        mock_sl.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
            },
        )

    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


async def test_recording_callback_no_recording_url_returns_200(client):
    """AT can call the recording endpoint without a recordingUrl in edge cases."""
    resp = await client.post(
        f"/api/telephony/recording?token={VALID_TOKEN}",
        data={"sessionId": "ATVId_test123"},
    )
    assert resp.status_code == 200


async def test_recording_callback_download_failure_returns_fallback_xml(client):
    """Network failure on recording download must not crash the endpoint."""
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=Exception("network error"))
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_http):
        resp = await client.post(
            f"/api/telephony/recording?token={VALID_TOKEN}",
            data={
                "sessionId": "ATVId_test123",
                "recordingUrl": "https://voice.africastalking.com/recordings/test.wav",
            },
        )

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]
    assert "unavailable" in resp.text.lower()
