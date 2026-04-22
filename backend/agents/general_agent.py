class GeneralAgent:
    """Phase 6: handles questions that do not fit health, agriculture, or education."""

    async def answer(self, question: str) -> str:
        raise NotImplementedError
