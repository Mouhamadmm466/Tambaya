import os

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# Force env vars before any app import — covers contexts where Settings() is re-instantiated
os.environ["WEBHOOK_SECRET"] = "test-webhook-token"
os.environ["AT_CALLBACK_BASE_URL"] = "https://test.namu.example.com"
os.environ["WHISPER_SERVICE_URL"] = "http://localhost:8001"
os.environ["WHISPER_API_KEY"] = "test-whisper-key"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ.setdefault("GEMMA_MODEL_NAME", "gemma4:4b")
os.environ.setdefault("SECRET_KEY", "test-secret-key-do-not-use-in-prod")
os.environ.setdefault("DATABASE_URL", "postgresql://namu_user:namu_dev_2026@db:5432/namu_tambaya")

# The global settings singleton is created at container startup from the real .env values.
# os.environ changes above don't retroactively affect it, so we mutate the object directly.
# Python passes objects by reference, so routers.telephony.settings IS config.settings —
# patching the attribute here is visible in every module that imported it.
from config import settings as _settings
_settings.webhook_secret = "test-webhook-token"
_settings.at_callback_base_url = "https://test.namu.example.com"
_settings.whisper_service_url = "http://localhost:8001"
_settings.whisper_api_key = "test-whisper-key"
_settings.ollama_base_url = "http://localhost:11434"
_settings.gemma_model_name = "gemma4:4b"


@pytest.fixture
async def client():
    """Minimal test app — telephony router only, no DB lifespan.
    Individual tests patch AsyncSessionLocal where DB interaction is under test.
    """
    from routers.telephony import router as telephony_router

    app = FastAPI()
    app.include_router(telephony_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
