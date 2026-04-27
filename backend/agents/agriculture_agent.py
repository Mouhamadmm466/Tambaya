import logging

logger = logging.getLogger(__name__)


class AgricultureAgent:
    """Answers farming, crop, and market price questions for Niger.

    Phase 4: returns a Hausa acknowledgment placeholder.
    Phase 5: replaces answer() with ChromaDB RAG + Ollama generation.
    """

    async def answer(self, question: str) -> str:
        # Phase 4 stub — replaced in Phase 5
        return (
            "An karbi tambayarku kan noma. "
            "Za mu ba ku amsar da ta dace da yankin Niger nan take."
        )


agriculture_agent = AgricultureAgent()
