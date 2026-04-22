import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from transcribe import WhisperTranscriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_transcriber: WhisperTranscriber | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber
    _transcriber = WhisperTranscriber()
    yield
    _transcriber = None


app = FastAPI(title="Namu Whisper Service", lifespan=lifespan)


def _check_auth(authorization: str) -> None:
    api_key = os.environ.get("WHISPER_API_KEY", "")
    if api_key and authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _transcriber is not None}


@app.post("/transcribe")
async def transcribe_audio(
    request: Request,
    authorization: str = Header(default=""),
):
    _check_auth(authorization)

    if _transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")

    result = await asyncio.to_thread(_transcriber.transcribe, audio_bytes)

    return {
        "text": result.text,
        "language": result.language,
        "avg_log_prob": result.avg_log_prob,
        "no_speech_prob": result.no_speech_prob,
        "duration_ms": result.duration_ms,
        "is_usable": result.is_usable,
    }
