from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    secret_key: str

    # Database
    database_url: str

    # Africa's Talking (Phase 2)
    at_api_key: str = ""
    at_username: str = ""
    at_phone_number: str = ""

    # ElevenLabs (Phase 5)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"

    # ChromaDB (Phase 5)
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # Audio temp files — served via StaticFiles at /audio (Phase 5)
    audio_temp_dir: str = "/app/audio_temp"

    # Ollama / Gemma (Phase 4)
    ollama_base_url: str = "http://ollama:11434"
    gemma_model_name: str = "gemma4"

    # Whisper (Phase 3)
    whisper_model_size: str = "large-v3"
    whisper_language: str = "ha"

    # PostgreSQL component values (used by docker-compose; mirrored here for reference)
    postgres_db: str = "namu_tambaya"
    postgres_user: str = "namu_user"
    postgres_password: str = ""

    # Grafana
    grafana_admin_password: str = "admin"

    # Telephony / Webhooks (Phase 2)
    webhook_secret: str = ""
    # TODO: set to your public server URL (e.g. https://your-domain.com) before the first real test call
    at_callback_base_url: str = ""

    # Whisper microservice (Phase 3)
    whisper_service_url: str = ""
    whisper_api_key: str = ""
    whisper_compute_type: str = "float16"


settings = Settings()
