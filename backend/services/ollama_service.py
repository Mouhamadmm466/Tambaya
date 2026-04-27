import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Thin HTTP client for Ollama's /api/generate endpoint.

    Uses /api/generate (not /api/chat) — the Ollama version on this host only
    exposes /api/generate.

    json_output=True forces valid JSON output (used by the router).
    json_output=False allows free-text Hausa (used by specialized agents).
    max_tokens=0 means no limit; set to 120 for agent responses to keep them
    phone-appropriate in length and fast to generate on CPU.

    Settings are read at call time so test patching takes effect without restart.
    """

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 0,
        json_output: bool = True,
    ) -> str:
        url = settings.ollama_base_url.rstrip("/") + "/api/generate"
        options: dict = {"temperature": 0}
        if max_tokens > 0:
            options["num_predict"] = max_tokens

        payload: dict = {
            "model": settings.gemma_model_name,
            "system": system_prompt,
            "prompt": user_message,
            "stream": False,
            "options": options,
        }
        if json_output:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        return response.json()["response"]


ollama_service = OllamaService()
