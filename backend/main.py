import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from config import settings
from database.connection import engine
from database.models import Base
from routers import health_check, telephony, transcription
from routers import agents as agents_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    if not settings.at_callback_base_url:
        logger.warning(
            "AT_CALLBACK_BASE_URL is not set — recording callbacks will not reach this server. "
            "Set it in .env to your public URL before accepting real calls."
        )
    if not settings.whisper_service_url:
        logger.warning(
            "WHISPER_SERVICE_URL is not set — transcription will fail. "
            "Set it in .env to the RunPod whisper service URL."
        )
    if not settings.ollama_base_url:
        logger.warning(
            "OLLAMA_BASE_URL is not set — routing will fall back to 'general' for every call."
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

app.include_router(health_check.router)
app.include_router(telephony.router)
app.include_router(transcription.router)
app.include_router(agents_router.router)
# Phase 5: app.include_router(tts.router)
