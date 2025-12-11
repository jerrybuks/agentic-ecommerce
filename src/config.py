"""Configuration settings for the application."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    database_url: str = Field(..., alias="DATABASE_URL")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_MODEL")
    openai_api_base: str = Field(default="", alias="OPENAI_API_BASE")
    project_name: str = "Agentic Ecommerce"
    api_version: str = "v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        populate_by_name = True


settings = Settings()

