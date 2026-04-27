import logging

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Handles questions that do not fit health, agriculture, or education.

    Also serves as the fallback when router confidence is low (unclear category).
    Phase 4: returns a Hausa acknowledgment placeholder.
    Phase 6: replaces answer() with ChromaDB RAG + Ollama generation.
    """

    async def answer(self, question: str) -> str:
        # Phase 4 stub — replaced in Phase 6
        return (
            "An karbi tambayarku. "
            "Za mu ba ku amsa nan take."
        )


general_agent = GeneralAgent()
