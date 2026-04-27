import logging

logger = logging.getLogger(__name__)


class EducationAgent:
    """Answers education and literacy questions relevant to Niger.

    Phase 4: returns a Hausa acknowledgment placeholder.
    Phase 6: replaces answer() with ChromaDB RAG + Ollama generation.
    """

    async def answer(self, question: str) -> str:
        # Phase 4 stub — replaced in Phase 6
        return (
            "An karbi tambayarku kan ilimi. "
            "Za mu ba ku amsar da ta dace da makarantun Niger nan take."
        )


education_agent = EducationAgent()
