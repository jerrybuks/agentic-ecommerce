"""Configuration settings for the application."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    database_url: str = Field(..., alias="DATABASE_URL")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_MODEL")
    openai_api_base: str = Field(default="", alias="OPENAI_API_BASE")
    # Chat model for agents (can be different from embedding model)
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    # Default similarity threshold for retrieval (0.0-1.0)
    default_similarity_threshold: float = Field(default=0.7, alias="DEFAULT_SIMILARITY_THRESHOLD")
    # LLM request timeout in seconds
    llm_timeout: float = Field(default=20.0, alias="LLM_TIMEOUT")
    project_name: str = "Agentic Ecommerce"
    api_version: str = "v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        populate_by_name = True


settings = Settings()

