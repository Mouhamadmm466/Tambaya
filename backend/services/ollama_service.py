class OllamaService:
    """Phase 4: sends prompts to Gemma 4 via the self-hosted Ollama instance."""

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
