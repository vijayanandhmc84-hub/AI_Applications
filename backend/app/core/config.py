"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Multimodal Document Chat"
    
    # Database
    DATABASE_URL: str = "postgresql://docuser:docpass@localhost:5432/docdb"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # # OpenAI
    # OPENAI_API_KEY: Optional[str] = None
    # OPENAI_MODEL: str = "gpt-4o-mini"
    # OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    
    # Upload Settings
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    
    # Vector Store Settings
    EMBEDDING_DIMENSION: int = 384  # sentence-transformers all-MiniLM-L6-v2
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
