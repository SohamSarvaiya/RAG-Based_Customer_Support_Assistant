from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_keys: str = "dev-key"
    anthropic_api_key: str = ""
    llm_provider: str = "auto"  # auto | anthropic | openai | gemini | ollama | none
    llm_model: str = "claude-sonnet-5-5"  # Anthropic model name
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "./data/chroma"
    collection_name: str = "support_docs"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    min_score: float = 0.30
    max_upload_mb: int = 20

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
