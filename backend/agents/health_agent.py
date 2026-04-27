import logging

logger = logging.getLogger(__name__)


class HealthAgent:
    """Answers health questions relevant to Niger and the Sahel region.

    Phase 4: returns a Hausa acknowledgment placeholder.
    Phase 6: replaces answer() with ChromaDB RAG + Ollama generation.
    Non-negotiable: always recommends professional help for serious symptoms.
    """

    async def answer(self, question: str) -> str:
        # Phase 4 stub — replaced in Phase 6
        return (
            "An karbi tambayarku kan lafiya. "
            "Koyaushe ku tuntubi likita ko ma'aikacin lafiya "
            "don matsalolin lafiya masu tsanani."
        )


health_agent = HealthAgent()
