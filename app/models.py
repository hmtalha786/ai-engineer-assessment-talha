"""Configuration, API models, shared data types, and expected errors."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve paths from the repository root, regardless of where the server is started.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Every retrieval result and API source must use one of these names.
SourceType = Literal["text_rag", "superhero_api", "web"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or the local .env file."""

    # Secrets are optional at startup so missing-key errors can be returned clearly.
    gemini_api_key: SecretStr | None = None
    superhero_api_token: SecretStr | None = None

    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    docs_dir: Path = PROJECT_ROOT / "docs"
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150
    rag_result_count: int = 4
    rag_score_threshold: float = 0.52
    superhero_api_base_url: str = "https://superheroapi.com"
    external_request_timeout_seconds: float = 15.0

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them for the lifetime of the process."""
    return Settings()


# Expected errors carry an HTTP status code that the API layer can safely expose.
class AppError(Exception):
    """Expected application failure that can safely be returned to a caller."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConfigurationError(AppError):
    """A required API key or runtime dependency is missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503)


class RetrievalError(AppError):
    """An information source failed or returned no useful result."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code)


class GenerationError(AppError):
    """Gemini failed while routing or generating an answer."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class AskRequest(BaseModel):
    """Validated request body accepted by POST /ask."""

    question: str = Field(min_length=2, max_length=2_000)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        # Length validation alone would allow a string containing only spaces.
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("question must contain at least two non-whitespace characters")
        return cleaned


class Source(BaseModel):
    """Public source metadata included in the API response."""

    type: SourceType
    name: str
    reference: str | None = None


class AskResponse(BaseModel):
    """The answer, selected route, and sources that actually succeeded."""

    answer: str
    sources: list[Source] = Field(min_length=1)
    route: list[SourceType]


class ContextItem(BaseModel):
    """Normalized result shared by every retrieval source."""

    content: str = Field(min_length=1)
    source_type: SourceType
    source_name: str
    source_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_source(self) -> Source:
        """Remove internal content and metadata before building the response."""
        return Source(
            type=self.source_type,
            name=self.source_name,
            reference=self.source_reference,
        )


class RouteDecision(BaseModel):
    """Structured output expected from the Gemini routing call."""

    sources: list[SourceType] = Field(min_length=1, max_length=2)
    superhero_names: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_route(self) -> "RouteDecision":
        # Preserve order while removing accidental duplicates from model output.
        self.sources = list(dict.fromkeys(self.sources))

        # Only combinations implemented by the orchestrator are accepted.
        allowed = {
            ("text_rag",),
            ("superhero_api",),
            ("web",),
            ("text_rag", "superhero_api"),
            ("superhero_api", "text_rag"),
        }
        if tuple(self.sources) not in allowed:
            raise ValueError("unsupported source combination")
        if "superhero_api" in self.sources and not self.superhero_names:
            raise ValueError("superhero_names is required for superhero_api")
        if "superhero_api" not in self.sources:
            self.superhero_names = []
        return self
