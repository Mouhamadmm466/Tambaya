import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Form, Query, Response
from sqlalchemy import update

from config import settings
from database.connection import AsyncSessionLocal
from database.models import CallLog, CallOutcome
from services.telephony_service import (
    build_fallback_xml,
    build_poor_quality_xml,
    build_transcription_response_xml,
    build_voice_response_xml,
    download_recording,
    hash_phone_number,
)
from services.whisper_service import whisper_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telephony", tags=["telephony"])

_XML = "application/xml"


def _token_valid(token: str) -> bool:
    """Constant-time comparison prevents timing attacks on the webhook token."""
    if not settings.webhook_secret:
        logger.warning("WEBHOOK_SECRET is not set — all webhook requests are accepted")
        return True
    return hmac.compare_digest(token, settings.webhook_secret)


@router.post("/voice")
async def voice_webhook(
    token: str = Query(default=""),
    session_id: Optional[str] = Form(default=None, alias="sessionId"),
    caller_number: Optional[str] = Form(default=None, alias="callerNumber"),
    call_session_state: Optional[str] = Form(default=None, alias="callSessionState"),
) -> Response:
    """Africa's Talking Voice Callback URL.

    AT calls this endpoint when a user dials the Namu Tambaya number.
    Must return valid AT XML within ~10 seconds or AT drops the call.
    All exceptions are caught and answered with fallback XML — never a 500.
    """
    if not _token_valid(token):
        return Response(status_code=403)

    try:
        if call_session_state == "Initiated" and caller_number and session_id:
            caller_hash = hash_phone_number(caller_number)
            try:
                async with AsyncSessionLocal() as db:
                    db.add(CallLog(
                        call_id=session_id,
                        caller_hash=caller_hash,
                        outcome=CallOutcome.success,
                    ))
                    await db.commit()
            except Exception:
                # Non-critical — the caller still gets the greeting even if logging fails
                logger.exception("DB write failed for session %s", session_id)

        xml = build_voice_response_xml(token)
        return Response(content=xml, media_type=_XML)

    except Exception:
        logger.exception("Unhandled error in voice webhook for session %s", session_id)
        return Response(content=build_fallback_xml(), media_type=_XML)


@router.post("/recording")
async def recording_callback(
    token: str = Query(default=""),
    session_id: Optional[str] = Form(default=None, alias="sessionId"),
    caller_number: Optional[str] = Form(default=None, alias="callerNumber"),
    recording_url: Optional[str] = Form(default=None, alias="recordingUrl"),
    duration_seconds: Optional[str] = Form(default=None, alias="durationInSeconds"),
) -> Response:
    """Africa's Talking Recording Callback URL.

    AT POSTs here when the caller's recording is ready, then waits for XML
    to know what to do next. We transcribe the audio and echo back the transcript.
    All exceptions return fallback XML — the call must never be silently dropped.
    """
    if not _token_valid(token):
        return Response(status_code=403)

    if not recording_url:
        return Response(status_code=200)

    try:
        audio_bytes = await download_recording(recording_url)
        logger.info(
            "Recording received: session=%s size=%d bytes duration=%ss",
            session_id,
            len(audio_bytes),
            duration_seconds,
        )

        result = await whisper_service.transcribe(audio_bytes)
        logger.info(
            "Transcription done: session=%s succeeded=%s duration_ms=%d text=%r",
            session_id,
            result.succeeded,
            result.duration_ms,
            result.text[:60] if result.text else "",
        )

        if session_id:
            try:
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(CallLog)
                        .where(CallLog.call_id == session_id)
                        .values(
                            transcription_succeeded=result.succeeded,
                            transcription_time_ms=result.duration_ms,
                            no_speech_prob=result.no_speech_prob,
                            avg_log_prob=result.avg_log_prob,
                            outcome=CallOutcome.success if result.succeeded else CallOutcome.fallback,
                        )
                    )
                    await db.commit()
            except Exception:
                logger.exception("DB update failed for transcription session %s", session_id)

        if result.succeeded and result.text:
            xml = build_transcription_response_xml(result.text)
        else:
            xml = build_poor_quality_xml()

        return Response(content=xml, media_type=_XML)

    except Exception:
        logger.exception("Failed to process recording for session %s", session_id)
        return Response(content=build_fallback_xml(), media_type=_XML)
