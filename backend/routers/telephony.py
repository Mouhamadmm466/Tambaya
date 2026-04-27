import asyncio
import hmac
import logging
import os
import uuid
from pathlib import Path
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
from services.elevenlabs_service import elevenlabs_service
from services.telephony_service import (
    build_agent_response_xml,
    build_fallback_xml,
    build_play_response_xml,
    build_poor_quality_xml,
    build_voice_response_xml,
    download_recording,
    hash_phone_number,
)
from services.whisper_service import whisper_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telephony", tags=["telephony"])

_XML = "application/xml"

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


async def _to_voice_xml(text: str, session_id: Optional[str]) -> str:
    """Convert answer text to AT XML.

    Tries ElevenLabs TTS → <Play url="..."/> if API key and voice ID are set.
    Falls back to <Say> when ElevenLabs is unavailable — the caller always hears
    something, even if TTS quality degrades.
    """
    if not (settings.elevenlabs_api_key and settings.elevenlabs_voice_id):
        return build_agent_response_xml(text)

    try:
        audio_bytes = await elevenlabs_service.synthesize(text)
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = Path(settings.audio_temp_dir) / filename
        filepath.write_bytes(audio_bytes)
        audio_url = (
            f"{settings.at_callback_base_url.rstrip('/')}/audio/{filename}"
        )
        # Schedule file deletion after 120s — AT downloads within seconds of the XML response.
        # call_later schedules a sync callback without creating a pending coroutine.
        asyncio.get_event_loop().call_later(
            120,
            lambda p=str(filepath): os.unlink(p) if os.path.exists(p) else None,
        )
        logger.info(
            "TTS audio saved: session=%s file=%s url=%s",
            session_id, filename, audio_url,
        )
        return build_play_response_xml(audio_url)
    except Exception:
        logger.exception(
            "ElevenLabs synthesis failed for session %s — using <Say> fallback",
            session_id,
        )
        return build_agent_response_xml(text)


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

    Full pipeline: download → transcribe → route → RAG answer → ElevenLabs TTS
    → <Play> XML. AT waits for our XML to control the next call step.
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

        category: Optional[CallCategory] = None
        answer_text: Optional[str] = None

        if transcription.succeeded and transcription.text:
            # Step 3: route
            try:
                route = await router_agent.classify(transcription.text)
                category = route.category
                logger.info(
                    "Routing: session=%s category=%s", session_id, category.value
                )
            except Exception:
                logger.exception(
                    "Router failed for session %s — using general", session_id
                )
                category = CallCategory.general

            # Step 4: agent answer (RAG for agriculture, stubs for others)
            try:
                agent = _AGENT_MAP[category]
                answer_text = await agent.answer(transcription.text)
                logger.info(
                    "Agent answer: session=%s category=%s text=%r",
                    session_id, category.value, answer_text[:60],
                )
            except Exception:
                logger.exception(
                    "Agent failed for session %s — answer_text will be None",
                    session_id,
                )

        # Step 5: persist to DB
        if session_id:
            try:
                update_values: dict = {
                    "transcription_succeeded": transcription.succeeded,
                    "transcription_time_ms": transcription.duration_ms,
                    "no_speech_prob": transcription.no_speech_prob,
                    "avg_log_prob": transcription.avg_log_prob,
                    "outcome": (
                        CallOutcome.success if transcription.succeeded
                        else CallOutcome.fallback
                    ),
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

        # Step 6: TTS → XML
        if answer_text:
            xml = await _to_voice_xml(answer_text, session_id)
        elif not transcription.succeeded:
            xml = build_poor_quality_xml()
        else:
            xml = build_fallback_xml()

        return Response(content=xml, media_type=_XML)

    except Exception:
        logger.exception("Failed to process recording for session %s", session_id)
        return Response(content=build_fallback_xml(), media_type=_XML)
