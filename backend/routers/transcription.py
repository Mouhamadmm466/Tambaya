from fastapi import APIRouter, HTTPException, Request

from config import settings
from services.whisper_service import whisper_service

router = APIRouter(prefix="/api/transcription", tags=["transcription"])


@router.post("/test")
async def test_transcription(request: Request):
    """Dev endpoint: POST raw WAV bytes, get a transcription back.

    Requires WHISPER_SERVICE_URL to be set in .env.
    Not exposed in production (remove from main.py include before go-live).
    """
    if not settings.whisper_service_url:
        raise HTTPException(status_code=503, detail="WHISPER_SERVICE_URL is not set")

    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data in request body")

    result = await whisper_service.transcribe(audio_bytes)
    return {
        "text": result.text,
        "language": result.language,
        "avg_log_prob": result.avg_log_prob,
        "no_speech_prob": result.no_speech_prob,
        "duration_ms": result.duration_ms,
        "succeeded": result.succeeded,
    }
