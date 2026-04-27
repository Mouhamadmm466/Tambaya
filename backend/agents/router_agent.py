import json
import logging
from dataclasses import dataclass

from database.models import CallCategory
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

# Valid category strings — derived from the enum so they stay in sync automatically.
_VALID = {cat.value for cat in CallCategory}

_SYSTEM_PROMPT = """\
You are a routing agent for Namu Tambaya, a Hausa-language voice assistant serving
rural communities in Niger, West Africa.

Your only job is to classify the user's Hausa question into exactly one of these categories:
- health: illness, symptoms, medicine, pregnancy, nutrition, first aid
- agriculture: farming, crops, planting seasons, animals, market prices, soil, weather for farming
- education: school enrollment, literacy, learning, scholarships, teachers, exams
- general: history, geography, religion, daily life, technology, news, or anything that does not fit the above
- unclear: gibberish, noise, random characters, or text that cannot be understood at all

Rules:
1. Return ONLY valid JSON: {"category": "<one of the five categories>"}
2. Never refuse. Never explain. Never add other keys.
3. When uncertain between two categories, choose the more specific one.
4. Only use "unclear" if the text is truly unintelligible — not just a short or simple question.
5. "general" is the safe default. Prefer it over "unclear".

Examples:
Input: "Ciwon kai yana damuna, zan sha aspirin?"
Output: {"category": "health"}

Input: "Yaron yana da zazzabi, me zan yi?"
Output: {"category": "health"}

Input: "Yaushe zan shuka masara a yankin Maradi?"
Output: {"category": "agriculture"}

Input: "Menene takin da ake amfani da shi a noman gero?"
Output: {"category": "agriculture"}

Input: "Yaya zan shigar da 'ya'yana makaranta a Niamey?"
Output: {"category": "education"}

Input: "Ina ana koyon Faransanci kyauta a Niger?"
Output: {"category": "education"}

Input: "Menene babban birni na Niger?"
Output: {"category": "general"}

Input: "Yaushe ne Sallah a bana?"
Output: {"category": "general"}

Input: "kdjfk lslsl mmm 999"
Output: {"category": "unclear"}\
"""


@dataclass
class RouteResult:
    category: CallCategory
    raw_response: str


def _parse_category(text: str) -> CallCategory:
    """Try JSON first, then substring search, then default to general."""
    # Try valid JSON
    try:
        data = json.loads(text.strip())
        cat = data.get("category", "").lower().strip()
        if cat in _VALID:
            return CallCategory(cat)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Try substring match (model might wrap the JSON in prose)
    text_lower = text.lower()
    for cat in CallCategory:
        if cat.value in text_lower:
            return cat

    # Safe default — NAMU_CONTEXT.md non-negotiable: always route somewhere
    logger.warning("Could not parse category from: %r — defaulting to general", text[:100])
    return CallCategory.general


class RouterAgent:
    async def classify(self, transcript: str) -> RouteResult:
        try:
            raw = await ollama_service.chat(_SYSTEM_PROMPT, transcript)
            category = _parse_category(raw)
            return RouteResult(category=category, raw_response=raw)
        except Exception:
            logger.exception("Router LLM call failed — defaulting to general")
            return RouteResult(category=CallCategory.general, raw_response="")


router_agent = RouterAgent()
