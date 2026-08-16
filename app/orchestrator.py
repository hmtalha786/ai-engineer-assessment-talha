"""Coordinate the request pipeline and wire its dependencies together."""

import asyncio
import logging
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.generate import AnswerGenerator, ContextBuilder
from app.models import AppError, AskResponse, ContextItem, RetrievalError, Settings, Source
from app.rag import RAGService
from app.route import RouterService
from app.sources import SuperheroService, WebSearchService


logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Run the route -> retrieve -> normalize -> answer pipeline."""

    def __init__(
        self,
        router: RouterService,
        rag: RAGService,
        superhero: SuperheroService,
        web_search: WebSearchService,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.router = router
        self.rag = rag
        self.superhero = superhero
        self.web_search = web_search
        self.context_builder = context_builder
        self.answer_generator = answer_generator

    async def ask(self, question: str) -> AskResponse:
        """Execute route -> retrieve -> normalize -> generate -> respond."""
        # 1. Decide which external knowledge sources this question needs.
        decision = await self.router.route(question)
        # 2. Build retrieval coroutines only for selected sources.
        operations = []
        if "text_rag" in decision.sources:
            operations.append(self.rag.retrieve(question))
        if "superhero_api" in decision.sources:
            operations.extend(
                self.superhero.retrieve(name) for name in decision.superhero_names
            )
        if "web" in decision.sources:
            operations.append(self.web_search.retrieve(question))

        # Run independent retrievals concurrently and preserve partial successes.
        results = await asyncio.gather(*operations, return_exceptions=True)
        # 3. Flatten successful adapter results into ContextItem objects.
        items: list[ContextItem] = []
        failures: list[AppError] = []
        for result in results:
            if isinstance(result, AppError):
                failures.append(result)
                logger.warning("A selected retrieval source failed: %s", result.message)
            elif isinstance(result, Exception):
                failures.append(RetrievalError("An information source failed unexpectedly."))
                logger.exception("Unexpected retrieval failure", exc_info=result)
            elif isinstance(result, list):
                items.extend(result)
            else:
                items.append(result)

        # Fail only when every selected retrieval failed or returned nothing.
        if not items:
            if failures:
                raise failures[0]
            raise RetrievalError(
                "No information was retrieved for this question.", status_code=404
            )

        # 4. Build grounded context, then ask Gemini for the final answer.
        context = self.context_builder.build(items)
        answer = await self.answer_generator.generate(question, context)
        # 5. Deduplicate source metadata without changing retrieval order.
        sources: list[Source] = []
        seen: set[tuple[str, str, str | None]] = set()
        for item in items:
            source = item.as_source()
            key = (source.type, source.name, source.reference)
            if key not in seen:
                seen.add(key)
                sources.append(source)

        return AskResponse(answer=answer, sources=sources, route=decision.sources)


# The container is the single place where concrete service dependencies are chosen.
@dataclass
class ServiceContainer:
    """Create and own the service graph used by FastAPI."""

    rag: RAGService
    superhero: SuperheroService
    web_search: WebSearchService
    orchestrator: ChatOrchestrator

    @classmethod
    def create(cls, settings: Settings) -> "ServiceContainer":
        """Construct the complete dependency graph from application settings."""
        # SecretStr prevents accidental logging; unwrap only for SDK construction.
        api_key = (
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None
        )
        superhero_token = (
            settings.superhero_api_token.get_secret_value()
            if settings.superhero_api_token
            else None
        )
        # Router and answer generator share one deterministic Gemini client.
        llm = (
            ChatGoogleGenerativeAI(
                model=settings.gemini_chat_model,
                api_key=api_key,
                temperature=0,
                retries=2,
                request_timeout=settings.external_request_timeout_seconds,
            )
            if api_key
            else None
        )
        embeddings = (
            GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                api_key=api_key,
            )
            if api_key
            else None
        )

        # Adapters own external access; the orchestrator only coordinates them.
        rag = RAGService(
            docs_dir=settings.docs_dir,
            embeddings=embeddings,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            result_count=settings.rag_result_count,
            score_threshold=settings.rag_score_threshold,
        )
        superhero = SuperheroService(
            token=superhero_token,
            base_url=settings.superhero_api_base_url,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
        web_search = WebSearchService(api_key=api_key, model=settings.gemini_chat_model)
        # Constructor injection keeps orchestration easy to test with fakes.
        orchestrator = ChatOrchestrator(
            router=RouterService(llm),
            rag=rag,
            superhero=superhero,
            web_search=web_search,
            context_builder=ContextBuilder(),
            answer_generator=AnswerGenerator(llm),
        )
        return cls(
            rag=rag,
            superhero=superhero,
            web_search=web_search,
            orchestrator=orchestrator,
        )

    async def close(self) -> None:
        """Release clients owned by this object during app shutdown."""
        await self.superhero.close()
        await self.web_search.close()
