import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Form, Query, Response
from sqlalchemy import update

from agents.agriculture_agent import agriculture_agent
from agents.education_agent import education_agent
from agents.general_agent import general_agent
from agents.health_agent import health_agent
from agents.router_agent import router_agent
from config import settings
from database.connection import AsyncSessionLocal
from database.models import CallCategory, CallLog, CallOutcome
from services.telephony_service import (
    build_agent_response_xml,
    build_fallback_xml,
    build_poor_quality_xml,
    build_voice_response_xml,
    download_recording,
    hash_phone_number,
)
from services.whisper_service import whisper_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telephony", tags=["telephony"])

_XML = "application/xml"

# Route each classified category to its agent.
# unclear maps to general — NAMU_CONTEXT.md: "Router confidence low → General Knowledge agent"
_AGENT_MAP = {
    CallCategory.health: health_agent,
    CallCategory.agriculture: agriculture_agent,
    CallCategory.education: education_agent,
    CallCategory.general: general_agent,
    CallCategory.unclear: general_agent,
}


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

    Pipeline: download → transcribe → route → agent answer → XML response.
    AT waits for our XML to control the next call step.
    All exceptions return fallback XML — the call must never be silently dropped.
    """
    if not _token_valid(token):
        return Response(status_code=403)

    if not recording_url:
        return Response(status_code=200)

    try:
        # Step 1: download audio
        audio_bytes = await download_recording(recording_url)
        logger.info(
            "Recording received: session=%s size=%d bytes duration=%ss",
            session_id, len(audio_bytes), duration_seconds,
        )

        # Step 2: transcribe
        transcription = await whisper_service.transcribe(audio_bytes)
        logger.info(
            "Transcription: session=%s succeeded=%s duration_ms=%d text=%r",
            session_id, transcription.succeeded, transcription.duration_ms,
            transcription.text[:60] if transcription.text else "",
        )

        # Step 3: route (only when transcription produced usable text)
        category: Optional[CallCategory] = None
        answer_text: Optional[str] = None

        if transcription.succeeded and transcription.text:
            try:
                route = await router_agent.classify(transcription.text)
                category = route.category
                logger.info("Routing: session=%s category=%s", session_id, category.value)
            except Exception:
                logger.exception("Router failed for session %s — using general", session_id)
                category = CallCategory.general

            # Step 4: get agent answer
            try:
                agent = _AGENT_MAP[category]
                answer_text = await agent.answer(transcription.text)
            except Exception:
                logger.exception("Agent failed for session %s — using fallback", session_id)

        # Step 5: persist transcription + category to DB
        if session_id:
            try:
                update_values: dict = {
                    "transcription_succeeded": transcription.succeeded,
                    "transcription_time_ms": transcription.duration_ms,
                    "no_speech_prob": transcription.no_speech_prob,
                    "avg_log_prob": transcription.avg_log_prob,
                    "outcome": CallOutcome.success if transcription.succeeded else CallOutcome.fallback,
                }
                if category is not None:
                    update_values["category"] = category
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(CallLog)
                        .where(CallLog.call_id == session_id)
                        .values(**update_values)
                    )
                    await db.commit()
            except Exception:
                logger.exception("DB update failed for session %s", session_id)

        # Step 6: return AT XML
        if answer_text:
            return Response(content=build_agent_response_xml(answer_text), media_type=_XML)
        elif not transcription.succeeded:
            return Response(content=build_poor_quality_xml(), media_type=_XML)
        else:
            return Response(content=build_fallback_xml(), media_type=_XML)

    except Exception:
        logger.exception("Failed to process recording for session %s", session_id)
        return Response(content=build_fallback_xml(), media_type=_XML)
