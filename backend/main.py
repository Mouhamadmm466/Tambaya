import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from config import settings
from database.connection import engine
from database.models import Base
from routers import agents as agents_router
from routers import health_check, telephony, transcription

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables and required directories before accepting requests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    os.makedirs(settings.audio_temp_dir, exist_ok=True)

    if not settings.at_callback_base_url:
        logger.warning(
            "AT_CALLBACK_BASE_URL is not set — recording callbacks and audio URLs "
            "will not work. Set it in .env before accepting real calls."
        )
    if not settings.whisper_service_url:
        logger.warning(
            "WHISPER_SERVICE_URL is not set — transcription will fail."
        )
    if not settings.ollama_base_url:
        logger.warning(
            "OLLAMA_BASE_URL is not set — routing will default to 'general'."
        )
    if not settings.elevenlabs_api_key:
        logger.warning(
            "ELEVENLABS_API_KEY is not set — TTS will fall back to AT's <Say>."
        )
    yield
    await engine.dispose()


app = FastAPI(
    title="Namu Tambaya API",
    description="Hausa-language AI voice agent for Niger",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Serve generated TTS audio files so AT can fetch them via <Play url="..."/>
app.mount(
    "/audio",
    StaticFiles(directory=settings.audio_temp_dir),
    name="audio",
)

app.include_router(health_check.router)
app.include_router(telephony.router)
app.include_router(transcription.router)
app.include_router(agents_router.router)
# Phase 6: app.include_router(health_knowledge.router)
