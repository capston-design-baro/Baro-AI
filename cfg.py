import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_COMPOSE_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    RAG_DB_URL: str = os.getenv("RAG_DB_URL", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    RAG_CASE_CACHE_TTL_SECONDS: int = int(os.getenv("RAG_CASE_CACHE_TTL_SECONDS", "604800"))

settings = Settings()
