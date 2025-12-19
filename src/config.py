"""Configuration settings for the application."""
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Ensure .env values are loaded into os.environ so downstream clients (e.g., Langfuse)
# can read them at import-time.
load_dotenv()


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
    # LLM generation parameters
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=1.0, alias="LLM_TOP_P")
    # LLM max tokens limits
    llm_max_tokens_orchestrator: int = Field(default=150, alias="LLM_MAX_TOKENS_ORCHESTRATOR")
    llm_max_tokens_agent: int = Field(default=500, alias="LLM_MAX_TOKENS_AGENT")
    project_name: str = "Agentic Ecommerce"
    api_version: str = "v1"
    # Langfuse observability
    langfuse_base_url: str = Field(default="", alias="LANGFUSE_BASE_URL")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        populate_by_name = True


settings = Settings()

