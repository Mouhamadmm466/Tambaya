class ChromaDBService:
    """Phase 4: retrieves relevant knowledge-base chunks for RAG."""

    async def query(self, text: str, collection: str, n_results: int = 5) -> list[str]:
        raise NotImplementedError
