import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_COMPOSE_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")


settings = Settings()