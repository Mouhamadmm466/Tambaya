import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Thin HTTP client for Ollama's /api/chat endpoint.

    Settings are read at call time so test patching takes effect without restart.
    """

    async def chat(self, system_prompt: str, user_message: str) -> str:
        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.gemma_model_name,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        return response.json()["message"]["content"]


ollama_service = OllamaService()
