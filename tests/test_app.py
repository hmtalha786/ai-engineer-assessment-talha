"""Focused unit tests that avoid real Gemini and Superhero API calls."""

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

import api
import models
import tools.hero_api_call as hero_api_call
from generate import build_context
from models import AppError, ContextItem, RouteDecision, clean_question, settings
from router import normalize_route
from tools.hero_api_call import fetch_hero, fetch_hero_record, search_hero_id
from tools.rag_generate import build_index, get_index
from tools.rag_retrieve import retrieve_documents


# --- Request and route validation ----------------------------------------
def test_blank_question_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        clean_question("   ")
    assert error.value.status_code == 400


def test_superhero_route_requires_a_name() -> None:
    with pytest.raises(AppError):
        normalize_route(RouteDecision(sources=["superhero_api"]))


def test_unsupported_route_combination_is_rejected() -> None:
    with pytest.raises(AppError):
        normalize_route(RouteDecision(sources=["text_rag", "web"]))


def test_names_are_dropped_when_superhero_is_not_selected() -> None:
    decision = normalize_route(
        RouteDecision(sources=["web"], superhero_names=["Batman"])
    )
    assert decision.superhero_names == []


# --- Superhero tool: the two-call name -> id -> full record lookup --------
def superhero_client(handler) -> httpx.AsyncClient:
    """Exercise HTTP handling without making a network request."""
    return httpx.AsyncClient(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    )


async def test_superhero_not_found_is_handled() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "error", "error": "not found"})

    async with superhero_client(handler) as client:
        with pytest.raises(AppError, match="No superhero named") as error:
            await search_hero_id(client, "Unknown Hero")
        assert error.value.status_code == 404


async def test_search_prefers_an_exact_name_match() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": "success",
                "results": [
                    {"id": "69", "name": "Batman II"},
                    {"id": "70", "name": "Batman"},
                ],
            },
        )

    async with superhero_client(handler) as client:
        hero_id, name = await search_hero_id(client, "batman")
    assert (hero_id, name) == ("70", "Batman")


async def test_fetch_hero_record_returns_every_section_from_one_call() -> None:
    """One GET on /{id} yields all sections; no per-aspect requests are made."""
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "response": "success",
                "id": "70",
                "name": "Batman",
                "powerstats": {"combat": "100"},
                "biography": {"full-name": "Bruce Wayne"},
                "appearance": {"race": "Human"},
                "work": {"occupation": "Businessman"},
                "connections": {"relatives": "Damian Wayne (son)"},
                "image": {"url": "https://example.test/batman.jpg"},
            },
        )

    async with superhero_client(handler) as client:
        details = await fetch_hero_record(client, "70")

    assert len(calls) == 1
    assert calls[0].endswith("/70")
    # Every useful section arrives together...
    assert set(details) == {
        "name",
        "powerstats",
        "biography",
        "appearance",
        "work",
        "connections",
    }
    # ...and the portrait URL is dropped as prompt noise.
    assert "image" not in details


async def test_fetch_hero_makes_exactly_two_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole flow: search by name for the id, then one call for everything."""
    called_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        called_paths.append(request.url.path)
        if "/search/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "response": "success",
                    "results": [{"id": "70", "name": "Batman"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "response": "success",
                "id": "70",
                "name": "Batman",
                "powerstats": {"intelligence": "100"},
                "biography": {"publisher": "DC Comics"},
            },
        )

    real_client = httpx.AsyncClient

    def client_with_mock_transport(**kwargs) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(hero_api_call.httpx, "AsyncClient", client_with_mock_transport)
    monkeypatch.setattr(settings, "superhero_api_token", SecretStr("test-token"))

    item = await fetch_hero("Batman")

    assert called_paths == ["/api/test-token/search/Batman", "/api/test-token/70"]
    assert item.source_name == "Batman"
    assert item.metadata == {"id": "70"}
    assert "intelligence" in item.content
    assert "DC Comics" in item.content


async def test_fetch_hero_without_a_token_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "superhero_api_token", None)

    with pytest.raises(AppError, match="SUPERHERO_API_TOKEN") as error:
        await fetch_hero("Batman")
    assert error.value.status_code == 503


# --- RAG lifecycle and failure handling ----------------------------------
class DeterministicEmbeddings(Embeddings):
    """Tiny predictable vectors keep RAG tests fast and independent of Gemini."""

    def __init__(self) -> None:
        self.calls = 0

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


@pytest.fixture(autouse=True)
def permissive_rag_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the similarity threshold out of the way of the fake embeddings."""
    monkeypatch.setattr(settings, "rag_score_threshold", 0.0)
    monkeypatch.setattr(settings, "rag_result_count", 2)


async def test_index_is_built_once_and_reused_for_requests(tmp_path: Path) -> None:
    (tmp_path / "guide.txt").write_text(
        "A production deployment requires tests and a rollback plan.",
        encoding="utf-8",
    )
    embeddings = DeterministicEmbeddings()

    build_index(tmp_path, embeddings)
    first = await retrieve_documents("What does a deployment require?")
    second = await retrieve_documents("Tell me the rollback requirement.")

    # One build, two retrievals: the index is not rebuilt per request.
    assert embeddings.calls == 1
    assert first[0].source_name == "guide.txt"
    assert second[0].source_name == "guide.txt"


async def test_empty_docs_starts_but_retrieval_is_unavailable(tmp_path: Path) -> None:
    build_index(tmp_path, DeterministicEmbeddings())

    assert get_index()["retriever"] is None
    with pytest.raises(AppError, match="No readable") as error:
        await retrieve_documents("What does the document say?")
    assert error.value.status_code == 503


async def test_corrupt_pdf_is_skipped_without_stopping_valid_txt(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a real PDF")
    (tmp_path / "valid.txt").write_text("Deployment guidance.", encoding="utf-8")

    index = build_index(tmp_path, DeterministicEmbeddings())

    assert index["retriever"] is not None
    assert index["files"] == ["valid.txt"]
    assert index["errors"]["broken.pdf"]


# --- End-to-end pipeline with the tool calls replaced --------------------
async def test_combined_route_returns_actual_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route(_question: str) -> RouteDecision:
        return RouteDecision(
            sources=["text_rag", "superhero_api"],
            superhero_names=["Batman"],
        )

    async def fake_documents(_question: str) -> list[ContextItem]:
        return [
            ContextItem(
                content="The handbook requires a rollback plan.",
                source_type="text_rag",
                source_name="engineering_handbook.txt",
            )
        ]

    async def fake_hero(name: str) -> ContextItem:
        return ContextItem(
            content=json.dumps({"name": name, "powerstats": {"intelligence": "100"}}),
            source_type="superhero_api",
            source_name=name,
        )

    async def fake_web(_question: str) -> list[ContextItem]:
        raise AssertionError("web should not be called")

    async def fake_generate(_question: str, context: str) -> str:
        assert "rollback plan" in context
        assert "Batman" in context
        return "A concise answer grounded in both selected sources."

    # api.py holds these as module-level names, so replacing them replaces the step.
    monkeypatch.setattr(api, "decide_route", fake_route)
    monkeypatch.setattr(api, "retrieve_documents", fake_documents)
    monkeypatch.setattr(api, "fetch_hero", fake_hero)
    monkeypatch.setattr(api, "search_web", fake_web)
    monkeypatch.setattr(api, "generate_answer", fake_generate)

    response = await api.ask(models.AskRequest(question="Compare the handbook and Batman."))

    assert response.route == ["text_rag", "superhero_api"]
    assert [(source.type, source.name) for source in response.sources] == [
        ("text_rag", "engineering_handbook.txt"),
        ("superhero_api", "Batman"),
    ]


async def test_all_sources_failing_raises_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route(_question: str) -> RouteDecision:
        return RouteDecision(sources=["web"])

    async def failing_web(_question: str) -> list[ContextItem]:
        raise AppError("Gemini web retrieval is unavailable.", status_code=502)

    monkeypatch.setattr(api, "decide_route", fake_route)
    monkeypatch.setattr(api, "search_web", failing_web)

    with pytest.raises(AppError, match="web retrieval") as error:
        await api.ask(models.AskRequest(question="What happened today?"))
    assert error.value.status_code == 502


# --- Context building -----------------------------------------------------
def test_duplicate_content_is_only_included_once() -> None:
    item = ContextItem(content="same text", source_type="web", source_name="A")
    duplicate = ContextItem(content="same text", source_type="web", source_name="B")

    context = build_context([item, duplicate])

    assert context.count("same text") == 1
