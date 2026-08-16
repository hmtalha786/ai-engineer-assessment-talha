"""Focused unit tests that avoid real Gemini and Superhero API calls."""

from pathlib import Path

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from app.generate import ContextBuilder
from app.models import AskRequest, ContextItem, RetrievalError, RouteDecision
from app.orchestrator import ChatOrchestrator
from app.rag import RAGService
from app.sources import SuperheroService


# --- Request and route validation ----------------------------------------
def test_blank_question_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_superhero_route_requires_a_name() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(sources=["superhero_api"])


@pytest.mark.asyncio
# --- Superhero adapter ----------------------------------------------------
async def test_superhero_not_found_is_handled() -> None:
    service = SuperheroService("secret", "https://example.test", 1)

    # MockTransport exercises HTTP handling without making a network request.
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "error", "error": "not found"})

    await service.client.aclose()
    service.client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(RetrievalError, match="No superhero named") as error:
            await service.retrieve("Unknown Hero")
        assert error.value.status_code == 404
    finally:
        await service.close()


# --- RAG lifecycle and failure handling ----------------------------------
class DeterministicEmbeddings(Embeddings):
    """Tiny predictable vectors keep RAG tests fast and independent of Gemini."""

    calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        # Related words deliberately share a vector for predictable matching.
        lowered = text.lower()
        if "deployment" in lowered or "rollback" in lowered:
            return [1.0, 0.0, 0.0]
        if "batman" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_index_is_built_once_and_reused_for_requests(tmp_path: Path) -> None:
    (tmp_path / "guide.txt").write_text(
        "A production deployment requires tests and a rollback plan.",
        encoding="utf-8",
    )
    embeddings = DeterministicEmbeddings()
    rag = RAGService(tmp_path, embeddings, 200, 20, 2, 0.0)

    await rag.initialize()
    first = await rag.retrieve("What does a deployment require?")
    second = await rag.retrieve("Tell me the rollback requirement.")

    assert rag.initialization_count == 1
    assert embeddings.calls == 1
    assert first[0].source_name == "guide.txt"
    assert second[0].source_name == "guide.txt"


@pytest.mark.asyncio
async def test_empty_docs_starts_but_retrieval_is_unavailable(tmp_path: Path) -> None:
    rag = RAGService(tmp_path, DeterministicEmbeddings(), 200, 20, 2, 0.0)

    await rag.initialize()

    assert rag.initialization_count == 1
    assert rag.retriever is None
    with pytest.raises(RetrievalError, match="No readable") as error:
        await rag.retrieve("What does the document say?")
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_corrupt_pdf_is_skipped_without_stopping_valid_txt(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a real PDF")
    (tmp_path / "valid.txt").write_text("Deployment guidance.", encoding="utf-8")
    rag = RAGService(tmp_path, DeterministicEmbeddings(), 200, 20, 2, 0.0)

    await rag.initialize()

    assert rag.retriever is not None
    assert rag.loaded_files == ["valid.txt"]
    assert rag.load_errors["broken.pdf"]


# --- End-to-end orchestration with dependency-injected fakes -------------
# Each fake implements only the method the orchestrator depends on.
class FakeRouter:
    async def route(self, _question: str) -> RouteDecision:
        return RouteDecision(
            sources=["text_rag", "superhero_api"],
            superhero_names=["Batman"],
        )


class FakeRAG:
    async def retrieve(self, _question: str) -> list[ContextItem]:
        return [
            ContextItem(
                content="The handbook requires a rollback plan.",
                source_type="text_rag",
                source_name="engineering_handbook.txt",
            )
        ]


class FakeSuperhero:
    async def retrieve(self, name: str) -> ContextItem:
        return ContextItem(
            content='{"name": "Batman", "intelligence": "100"}',
            source_type="superhero_api",
            source_name=name,
        )


class FakeWeb:
    async def retrieve(self, _question: str) -> list[ContextItem]:
        raise AssertionError("web should not be called")


class FakeAnswerGenerator:
    async def generate(self, _question: str, context: str) -> str:
        assert "rollback plan" in context
        assert "Batman" in context
        return "A concise answer grounded in both selected sources."


@pytest.mark.asyncio
# This verifies the complete flow without calling external services.
async def test_combined_route_returns_actual_sources() -> None:
    orchestrator = ChatOrchestrator(
        router=FakeRouter(),
        rag=FakeRAG(),
        superhero=FakeSuperhero(),
        web_search=FakeWeb(),
        context_builder=ContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    response = await orchestrator.ask("Compare the handbook and Batman.")

    assert response.route == ["text_rag", "superhero_api"]
    assert [(source.type, source.name) for source in response.sources] == [
        ("text_rag", "engineering_handbook.txt"),
        ("superhero_api", "Batman"),
    ]
