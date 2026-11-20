"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General App Settings
    app_name: str = Field(
        default="LPU CSE-AIML RAG Chatbot Backend",
        description="Application name",
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    app_env: str = Field(
        default="development",
        description="Environment: development, test, production",
    )
    debug: bool = Field(default=True, description="Debug mode")
    api_host: str = Field(default="0.0.0.0", description="Host to bind server")
    api_port: int = Field(default=8000, description="Port to bind server")
    allowed_origins: str = Field(
        default=(
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:5174,http://127.0.0.1:5174"
        ),
        description="Comma-separated list of explicit allowed CORS origins (wildcards disallowed with credentials)",
    )

    # Data Directories
    raw_data_dir: str = Field(
        default="data/raw",
        description="Relative or absolute path to raw data directory",
    )
    processed_data_dir: str = Field(
        default="data/processed",
        description="Relative or absolute path to processed data directory",
    )

    # Local LLM (Ollama) Settings
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for local Ollama HTTP API",
    )
    llm_primary_model: str = Field(
        default="qwen3:4b-instruct-2507-q4_K_M",
        description="Active primary local LLM model tag",
    )
    llm_fallback_model: str = Field(
        default="qwen2.5:3b-instruct-q4_K_M",
        description="Documented fallback model tag (not active for automatic fallback)",
    )
    llm_num_ctx: int = Field(
        default=4096,
        description="Context window length in tokens",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for deterministic academic generation",
    )
    llm_keep_alive: str = Field(
        default="30m",
        description="Model keep-alive duration in VRAM (e.g. '30m')",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout in seconds for Ollama HTTP API generation requests",
    )
    llm_max_concurrency: int = Field(
        default=1,
        description="Maximum concurrent LLM generation requests allowed",
    )

    # LLM Provider Selection ("ollama" or "gemini")
    llm_provider: str = Field(
        default="ollama",
        description="Active LLM provider: 'ollama' (default local runtime) or 'gemini' (Google AI Studio API)",
    )

    # Google Gemini API Settings (Phase 11 - Free Tier)
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google AI Studio Gemini API key (optional, free tier only)",
    )
    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Default Gemini model tag for grounded generation",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Base URL for Google Generative Language REST API",
    )
    gemini_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds for Gemini HTTP API requests",
    )
    gemini_max_concurrency: int = Field(
        default=2,
        description="Maximum concurrent Gemini requests allowed (strictly bounds RPM on free tier)",
    )
    gemini_fallback_to_local: bool = Field(
        default=True,
        description="Whether to automatically fall back to local Ollama on Gemini rate limits or failures",
    )

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        ]
        return list(dict.fromkeys(origins))

    def resolve_path(self, relative_path_str: str) -> Path:
        """Resolves a path relative to current working directory or repository root."""
        p = Path(relative_path_str)
        if p.is_absolute() and p.exists():
            return p
        # Check from current working directory
        cwd = Path.cwd()
        if (cwd / p).exists():
            return (cwd / p).resolve()
        # Check from backend directory parent (repo root)
        backend_dir = Path(__file__).resolve().parent.parent.parent
        repo_root = backend_dir.parent if backend_dir.name == "backend" else backend_dir
        if (repo_root / p).exists():
            return (repo_root / p).resolve()
        # Fallback to direct resolution
        return (cwd / p).resolve()

    @property
    def resolved_raw_data_dir(self) -> Path:
        return self.resolve_path(self.raw_data_dir)

    @property
    def resolved_processed_data_dir(self) -> Path:
        return self.resolve_path(self.processed_data_dir)


# Global singleton settings instance
settings = Settings()
