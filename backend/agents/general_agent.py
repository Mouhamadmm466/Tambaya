import logging

from services.chromadb_service import chromadb_service
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

_COLLECTION = "general_niger"

# Returned when both ChromaDB and Gemma 4's own knowledge fail.
HONEST_FALLBACK = (
    "Bana da cikakken bayani kan wannan a yanzu. "
    "Ku tuntubi shugabannin al'ummarku ko hukumomin yankinku."
)

# Used when ChromaDB has relevant chunks — grounded in local knowledge base.
_RAG_PROMPT_TEMPLATE = """\
You are Namu, a trusted general knowledge assistant for communities in Niger, West Africa.
Answer the question below in 2-3 short, clear sentences in Hausa.
Use ONLY the facts provided in the retrieved knowledge below.
If the knowledge does not contain the answer, say exactly this Hausa phrase and nothing else:
"Bana da cikakken bayani kan wannan a yanzu. Ku tuntubi shugabannin al'ummarku ko hukumomin yankinku."
Do not add greetings, do not repeat the question, do not explain your reasoning.
Keep your answer short — the caller is listening on a phone.

Retrieved knowledge:
---
{chunks}
---

Question:\
"""

# Used when ChromaDB has no relevant chunks — lets Gemma 4 answer from its own
# training knowledge, but with a strict instruction to admit ignorance honestly.
# This is the "catch-all" path for questions outside the local knowledge base.
_OPEN_PROMPT = """\
You are Namu, a trusted general knowledge assistant for Hausa-speaking communities in Niger, West Africa.
Answer the question below in 2-3 short, clear sentences in Hausa (Niger dialect).
Answer ONLY if you are confident the information is accurate and relevant to Niger or West Africa.
If you are not sure or the question is outside your knowledge, say exactly this Hausa phrase and nothing else:
"Bana da cikakken bayani kan wannan a yanzu. Ku tuntubi shugabannin al'ummarku ko hukumomin yankinku."
Do not add greetings, do not repeat the question, do not explain your reasoning.
Keep your answer short — the caller is listening on a phone.

Question:\
"""


class GeneralAgent:
    """Answers general knowledge questions for Niger using a two-stage pipeline.

    Stage 1 — RAG: embed question → query ChromaDB (general_niger) → if chunks
    found, build grounded prompt → Ollama generate → return answer.

    Stage 2 — Open: if ChromaDB returns no chunks, ask Gemma 4 directly with
    a careful prompt that requires honesty when uncertain. This is the catch-all
    path for questions outside the local knowledge base, including all questions
    routed here from the Router Agent with low confidence (unclear category).

    Both stages fall back to HONEST_FALLBACK on any service error.

    Phase 6: full RAG + open-question implementation.
    """

    async def answer(self, question: str) -> str:
        # Stage 1: try ChromaDB RAG
        try:
            chunks = await chromadb_service.query(question, _COLLECTION, n_results=3)
        except Exception:
            logger.exception("ChromaDB query failed — falling through to open prompt")
            chunks = []

        if chunks:
            chunks_text = "\n---\n".join(chunks)
            system_prompt = _RAG_PROMPT_TEMPLATE.replace("{chunks}", chunks_text)
            try:
                answer = await ollama_service.chat(
                    system_prompt=system_prompt,
                    user_message=question,
                    max_tokens=120,
                    json_output=False,
                )
                result = answer.strip()
                if result:
                    logger.info(
                        "GeneralAgent answered via RAG: %r", result[:60]
                    )
                    return result
            except Exception:
                logger.exception(
                    "Ollama RAG generation failed — falling through to open prompt"
                )

        # Stage 2: no chunks or RAG failed — try Gemma 4's own knowledge
        logger.info("GeneralAgent falling through to open prompt for: %r", question[:60])
        try:
            answer = await ollama_service.chat(
                system_prompt=_OPEN_PROMPT,
                user_message=question,
                max_tokens=120,
                json_output=False,
            )
            result = answer.strip()
            return result if result else HONEST_FALLBACK
        except Exception:
            logger.exception("Ollama open-question generation failed — returning fallback")
            return HONEST_FALLBACK


general_agent = GeneralAgent()
