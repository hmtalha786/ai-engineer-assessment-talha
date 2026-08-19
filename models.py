"""Settings, API models, shared data types, and application errors.

Only data declarations and small helper functions live here. Pydantic models and
the error type are classes because that is what pydantic/FastAPI and Python's
`raise` require -- there are no service or orchestration classes anywhere.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths from the project root, regardless of where the server is started.
PROJECT_ROOT = Path(__file__).resolve().parent

# Every retrieval result and API source must use one of these names.
SourceType = Literal["text_rag", "superhero_api", "web"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or the local .env file."""

    # Secrets are optional at startup so missing-key errors can be returned clearly.
    # SecretStr keeps the values out of logs and reprs; unwrap with secret().
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


# Built once at import; every module imports this same object.
settings = Settings()


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class AppError(Exception):
    """Expected application failure that can safely be returned to a caller."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Factory functions replace error subclasses: one class, clear call sites.
def bad_request(message: str) -> AppError:
    """The caller sent something invalid."""
    return AppError(message, status_code=400)


def configuration_error(message: str) -> AppError:
    """A required API key or runtime dependency is missing."""
    return AppError(message, status_code=503)


def retrieval_error(message: str, status_code: int = 502) -> AppError:
    """An information source failed or returned no useful result."""
    return AppError(message, status_code=status_code)


def generation_error(message: str) -> AppError:
    """Gemini failed while routing or generating an answer."""
    return AppError(message, status_code=502)


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Validated request body accepted by POST /ask."""

    question: str = Field(min_length=2, max_length=2_000)


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
    """Normalized result shared by every retrieval tool."""

    content: str = Field(min_length=1)
    source_type: SourceType
    source_name: str
    source_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    """Structured output expected from the Gemini routing call."""

    sources: list[SourceType] = Field(min_length=1, max_length=2)
    superhero_names: list[str] = Field(default_factory=list, max_length=5)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def secret(value: SecretStr | None) -> str | None:
    """Unwrap an optional secret only where an SDK actually needs the raw string."""
    return value.get_secret_value() if value else None


def clean_question(question: str) -> str:
    """Trim the question and reject input that is only whitespace."""
    # Length validation alone would allow a string containing only spaces.
    cleaned = question.strip()
    if len(cleaned) < 2:
        raise bad_request("question must contain at least two non-whitespace characters")
    return cleaned


def to_source(item: ContextItem) -> Source:
    """Remove internal content and metadata before building the response."""
    return Source(
        type=item.source_type,
        name=item.source_name,
        reference=item.source_reference,
    )


# --------------------------------------------------------------------------
# Shared Gemini clients (built once, reused by router.py and generate.py)
# --------------------------------------------------------------------------
_clients: dict[str, Any] = {}


def get_chat_model() -> Any:
    """Return the shared deterministic Gemini chat model, or None without a key."""
    if "chat" not in _clients:
        if not settings.gemini_api_key:
            _clients["chat"] = None
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            _clients["chat"] = ChatGoogleGenerativeAI(
                model=settings.gemini_chat_model,
                api_key=secret(settings.gemini_api_key),
                temperature=0,
                retries=2,
                request_timeout=settings.external_request_timeout_seconds,
            )
    return _clients["chat"]


def get_embeddings() -> Any:
    """Return the shared Gemini embedding model, or None without a key."""
    if "embeddings" not in _clients:
        if not settings.gemini_api_key:
            _clients["embeddings"] = None
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            _clients["embeddings"] = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                api_key=secret(settings.gemini_api_key),
            )
    return _clients["embeddings"]
