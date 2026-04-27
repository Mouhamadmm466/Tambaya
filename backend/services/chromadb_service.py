import asyncio
import logging

import chromadb
import httpx

from config import settings

logger = logging.getLogger(__name__)

_EMBED_MODEL = "nomic-embed-text"


class _OllamaEmbedFn:
    """ChromaDB-compatible embedding function backed by Ollama's /api/embed endpoint.

    Uses nomic-embed-text v1.5 — the only self-hosted option with meaningful
    multilingual (including Hausa) semantic understanding.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        url = settings.ollama_base_url.rstrip("/") + "/api/embed"
        response = httpx.post(
            url,
            json={"model": _EMBED_MODEL, "input": input},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]


class ChromaDBService:
    """Queries ChromaDB collections for RAG retrieval.

    Uses asyncio.to_thread to run the synchronous chromadb HTTP client
    without blocking the event loop.
    """

    def __init__(self):
        self._client: chromadb.HttpClient | None = None
        self._embed_fn = _OllamaEmbedFn()

    @property
    def _chroma(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        return self._client

    def _sync_query(
        self, text: str, collection_name: str, n_results: int
    ) -> list[str]:
        col = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )
        results = col.query(query_texts=[text], n_results=n_results)
        return results.get("documents", [[]])[0]

    async def query(
        self, text: str, collection_name: str, n_results: int = 3
    ) -> list[str]:
        return await asyncio.to_thread(
            self._sync_query, text, collection_name, n_results
        )


chromadb_service = ChromaDBService()
