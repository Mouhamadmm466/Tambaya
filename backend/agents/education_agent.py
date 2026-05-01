import logging

from services.chromadb_service import chromadb_service
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

_COLLECTION = "education_niger"

# Shown to the caller when the knowledge base has no relevant information.
# Directs them to the local education office — the right referral when the AI
# genuinely does not know.
HONEST_FALLBACK = (
    "Bana da cikakken bayani kan wannan a yanzu. "
    "Ku tuntubi ofishin ilimi na gundumarku ko malamin makaranta kusa da ku."
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are Namu, a trusted education advisor for communities in Niger, West Africa.
Answer the education question below in 2-3 short, clear sentences in Hausa.
Use ONLY the facts provided in the retrieved knowledge below.
If the knowledge does not contain the answer, say exactly this Hausa phrase and nothing else:
"Bana da cikakken bayani kan wannan a yanzu. Ku tuntubi ofishin ilimi na gundumarku ko malamin makaranta kusa da ku."
Do not add greetings, do not repeat the question, do not explain your reasoning.
Keep your answer short — the caller is listening on a phone.

Retrieved knowledge:
---
{chunks}
---

Education question:\
"""


class EducationAgent:
    """Answers education and literacy questions for Niger using RAG.

    Pipeline: embed question → query ChromaDB (education_niger) → build prompt →
    Ollama generate → return Hausa answer.

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
                "ChromaDB returned no chunks for education question: %r", question[:60]
            )
            return HONEST_FALLBACK

        chunks_text = "\n---\n".join(chunks)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.replace("{chunks}", chunks_text)

        try:
            answer = await ollama_service.chat(
                system_prompt=system_prompt,
                user_message=question,
                max_tokens=120,
                json_output=False,
            )
            return answer.strip() or HONEST_FALLBACK
        except Exception:
            logger.exception("Ollama generation failed — returning honest fallback")
            return HONEST_FALLBACK


education_agent = EducationAgent()
