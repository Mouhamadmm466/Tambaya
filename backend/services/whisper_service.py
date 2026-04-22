import logging
from dataclasses import dataclass

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    avg_log_prob: float
    no_speech_prob: float
    duration_ms: int
    succeeded: bool


class WhisperService:
    """Calls the remote faster-whisper microservice over HTTP.

    Settings are read at call time (not __init__) so test patching takes effect
    without restarting the process.
    """

    async def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        url = settings.whisper_service_url.rstrip("/") + "/transcribe"
        headers: dict[str, str] = {"Content-Type": "audio/wav"}
        if settings.whisper_api_key:
            headers["Authorization"] = f"Bearer {settings.whisper_api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, content=audio_bytes, headers=headers)
            response.raise_for_status()

        data = response.json()
        return TranscriptionResult(
            text=data["text"],
            language=data["language"],
            avg_log_prob=data["avg_log_prob"],
            no_speech_prob=data["no_speech_prob"],
            duration_ms=data["duration_ms"],
            succeeded=data["is_usable"],
        )


whisper_service = WhisperService()
