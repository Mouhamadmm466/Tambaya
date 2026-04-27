import hashlib
import logging
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx

from config import settings

logger = logging.getLogger(__name__)


def hash_phone_number(number: str) -> str:
    """SHA-256 of the caller's phone number.
    Enables repeat-caller analytics without storing the number itself (privacy requirement).
    """
    return hashlib.sha256(number.encode()).hexdigest()


def build_voice_response_xml(token: str) -> str:
    """AT XML returned on call initiation: greet the caller then record their question.

    Phase 5: replace <Say> with <Play url="...ElevenLabs audio..."/>.
    Phase 7: replace English greeting text with natural Niger Hausa.
    """
    base_url = settings.at_callback_base_url.rstrip("/")
    recording_callback_url = f"{base_url}/api/telephony/recording?token={token}"

    root = Element("Response")
    say = SubElement(root, "Say", voice="woman")
    say.text = "Welcome to Namu Tambaya. Please speak your question after the beep."
    SubElement(
        root,
        "Record",
        maxLength="60",
        trimSilence="true",
        playBeep="true",
        callbackUrl=recording_callback_url,
    )
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="unicode")


def build_fallback_xml() -> str:
    """Safe XML returned when any unhandled error occurs in the voice webhook.

    AT silently drops calls on non-2xx responses — always respond 200 + XML.
    Phase 7: replace English text with natural Niger Hausa.
    """
    root = Element("Response")
    say = SubElement(root, "Say", voice="woman")
    say.text = (
        "We are sorry, Namu Tambaya is temporarily unavailable. "
        "Please call back in a few minutes."
    )
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="unicode")


def build_agent_response_xml(response_text: str) -> str:
    """AT XML wrapping an agent's Hausa answer.

    Phase 5: replace <Say> with <Play url="...ElevenLabs audio..."/>.
    """
    root = Element("Response")
    say = SubElement(root, "Say", voice="woman")
    say.text = response_text
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="unicode")


def build_poor_quality_xml() -> str:
    """AT XML returned when transcription quality falls below usable thresholds.

    Phase 7: replace English text with natural Niger Hausa.
    """
    root = Element("Response")
    say = SubElement(root, "Say", voice="woman")
    say.text = (
        "We could not understand your question clearly. "
        "Please call back and speak clearly after the beep."
    )
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="unicode")


async def download_recording(recording_url: str) -> bytes:
    """Fetch caller audio from AT's servers to ours immediately on receipt.

    Non-negotiable: user audio must land on our infrastructure, never persist on AT's servers.
    Returns raw bytes. The caller decides what to do with them:
      Phase 2 — discard after confirming download succeeded.
      Phase 3 — pass to WhisperService for transcription.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(recording_url)
        response.raise_for_status()
        return response.content
