import logging

from services.chromadb_service import chromadb_service
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

_COLLECTION = "health_niger"

# Appended to EVERY health answer — non-negotiable (NAMU_CONTEXT.md §10 rule 4).
# Enforced in Python code, not just in the prompt, so Gemma 4 cannot skip it.
DOCTOR_SUFFIX = (
    "Koyaushe ku tuntubi likita ko ma'aikacin lafiya kusa da ku."
)

# Returned when ChromaDB has no relevant chunks or any service fails.
# Built using DOCTOR_SUFFIX so there is one source of truth for the recommendation text.
HONEST_FALLBACK = (
    "Bana da cikakken bayani kan wannan a yanzu. "
    + DOCTOR_SUFFIX
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are Namu, a trusted health information assistant for rural communities in Niger, West Africa.
Answer the health question below in 2-3 short, clear sentences in Hausa.
Use ONLY the facts provided in the retrieved knowledge below.
Do NOT add a closing recommendation to see a doctor — that will be added separately.
If the knowledge does not contain the answer, say exactly this Hausa phrase and nothing else:
"Bana da cikakken bayani kan wannan a yanzu."
Do not add greetings, do not repeat the question, do not explain your reasoning.
Keep your answer short — the caller is listening on a phone.

Retrieved knowledge:
---
{chunks}
---

Health question:\
"""


class HealthAgent:
    """Answers health questions for Niger using RAG over a Hausa health knowledge base.

    Pipeline: embed question → query ChromaDB (health_niger) → build prompt →
    Ollama generate → append DOCTOR_SUFFIX → return Hausa answer.

    The DOCTOR_SUFFIX is appended unconditionally in Python — this is the hard
    enforcement of the non-negotiable: health agent always recommends professional
    help (NAMU_CONTEXT.md §10 rule 4).

    Phase 6: full RAG implementation.
    """

    async def answer(self, question: str) -> str:
        try:
            chunks = await chromadb_service.query(question, _COLLECTION, n_results=3)
        except Exception:
            logger.exception("ChromaDB query failed — returning honest fallback")
            return HONEST_FALLBACK

        if not chunks:
            logger.warning(
                "ChromaDB returned no chunks for health question: %r", question[:60]
            )
            return HONEST_FALLBACK

        chunks_text = "\n---\n".join(chunks)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.replace("{chunks}", chunks_text)

        try:
            raw = await ollama_service.chat(
                system_prompt=system_prompt,
                user_message=question,
                max_tokens=120,
                json_output=False,
            )
            answer_text = raw.strip()
        except Exception:
            logger.exception("Ollama generation failed — returning honest fallback")
            return HONEST_FALLBACK

        if not answer_text:
            return HONEST_FALLBACK

        # Non-negotiable: always end with doctor recommendation.
        return f"{answer_text} {DOCTOR_SUFFIX}"


health_agent = HealthAgent()
