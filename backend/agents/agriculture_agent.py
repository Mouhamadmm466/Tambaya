import logging

from services.chromadb_service import chromadb_service
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

_COLLECTION = "agriculture_niger"

# Shown to the caller when the knowledge base has no relevant information.
# Directs them to a local farming official — the correct recommendation when
# the AI genuinely does not know.
HONEST_FALLBACK = (
    "Bana da cikakken bayani kan wannan a yanzu. "
    "Za ku iya tambayar mai noma kusa da ku "
    "ko hukumar noma ta gunduma."
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are Namu, a trusted farming advisor for rural communities in Niger, West Africa.
Answer the farmer's question below in 2-3 short, clear sentences in Hausa.
Use ONLY the facts provided in the retrieved knowledge below.
If the knowledge does not contain the answer, say exactly this Hausa phrase and nothing else:
"Bana da cikakken bayani kan wannan a yanzu. Za ku iya tambayar mai noma kusa da ku ko hukumar noma ta gunduma."
Do not add greetings, do not repeat the question, do not explain your reasoning.
Keep your answer short — the caller is listening on a phone.

Retrieved knowledge:
---
{chunks}
---

Farmer's question:\
"""


class AgricultureAgent:
    """Answers farming, crop, and market price questions for Niger using RAG.

    Pipeline: embed question → query ChromaDB → build prompt → Ollama generate
    → return Hausa answer.

    Phase 5: full RAG implementation.
    Phase 8: move Ollama generation to RunPod GPU for <5s latency.
    """

    async def answer(self, question: str) -> str:
        try:
            chunks = await chromadb_service.query(question, _COLLECTION, n_results=3)
        except Exception:
            logger.exception("ChromaDB query failed — returning honest fallback")
            return HONEST_FALLBACK

        if not chunks:
            logger.warning("ChromaDB returned no chunks for question: %r", question[:60])
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


agriculture_agent = AgricultureAgent()
