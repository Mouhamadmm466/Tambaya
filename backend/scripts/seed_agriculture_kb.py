#!/usr/bin/env python3
"""Seed the agriculture knowledge base into ChromaDB.

Run once after deployment, and again whenever knowledge base documents change.
Idempotent: deletes and recreates the collection on every run.

Usage (from inside the backend container):
    docker exec namu_backend python scripts/seed_agriculture_kb.py

Prerequisites:
    1. namu_chromadb container must be running
    2. nomic-embed-text must be pulled: docker exec namu_ollama ollama pull nomic-embed-text
"""
import os
import sys
from pathlib import Path

import chromadb
import httpx

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "agriculture_niger"

KB_DIR = Path(__file__).parent.parent / "knowledge_base" / "agriculture"
CHUNK_WORDS = 250
OVERLAP_WORDS = 40
BATCH_SIZE = 8


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        start += CHUNK_WORDS - OVERLAP_WORDS
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    # cosine distance gives better semantic similarity for text embeddings
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection '{COLLECTION}'")

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        print(f"ERROR: No .md files found in {KB_DIR}", file=sys.stderr)
        sys.exit(1)

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{md_file.stem}_{i}")
            all_meta.append({"source": md_file.name, "chunk_index": i})
        print(f"  {md_file.name}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks via {EMBED_MODEL}...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_docs = all_chunks[i : i + BATCH_SIZE]
        embeddings = embed_batch(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=all_ids[i : i + BATCH_SIZE],
            metadatas=all_meta[i : i + BATCH_SIZE],
        )
        print(f"  Stored chunks {i + 1}–{i + len(batch_docs)}")

    print(f"\nDone. {len(all_chunks)} chunks in '{COLLECTION}'.")
    print("Run `ollama pull nomic-embed-text` on the Ollama container if not done yet.")


if __name__ == "__main__":
    main()
