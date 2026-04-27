import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)

_API_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsService:
    """Converts Hausa answer text to speech using the custom Niger female voice.

    Uses eleven_multilingual_v2 — the only ElevenLabs model with Hausa support.
    Output: mp3_44100_64 (64 kbps MP3). Phone calls are 8 kHz so 64 kbps has
    zero perceptible quality loss over AT's voice channel, at half the file size.

    Privacy: only the generated answer text is sent — never the caller's transcript.
    """

    async def synthesize(self, text: str) -> bytes:
        if not settings.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set")
        if not settings.elevenlabs_voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is not set")

        url = f"{_API_BASE}/text-to-speech/{settings.elevenlabs_voice_id}"
        payload = {
            "text": text,
            "model_id": settings.elevenlabs_model,
            "output_format": "mp3_44100_64",
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
            },
        }
        headers = {
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.content


elevenlabs_service = ElevenLabsService()
