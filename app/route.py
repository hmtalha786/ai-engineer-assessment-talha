"""Choose which information sources a question needs."""

from langchain_core.language_models.chat_models import BaseChatModel

from app.models import ConfigurationError, GenerationError, RouteDecision

# The first model call chooses data sources; it never writes the final answer.
ROUTER_PROMPT = """You route questions to information sources.

Available routes:
- text_rag: questions about documents supplied to this application.
- superhero_api: factual questions about one or more named superheroes.
- web: general or current questions not about the supplied documents or superheroes.
- text_rag + superhero_api: questions explicitly requiring both document content and superhero data.

Use only one of those exact route choices. Extract canonical superhero names when the
superhero API is selected. Do not answer the question and do not call tools.

Question: {question}
"""

# The router is intentionally isolated because it is the first pipeline stage.
class RouterService:
    """Ask Gemini for a validated RouteDecision instead of free-form text."""

    def __init__(self, llm: BaseChatModel | None) -> None:
        # LangChain parses model output directly into the Pydantic route model.
        self._router = llm.with_structured_output(RouteDecision) if llm else None

    async def route(self, question: str) -> RouteDecision:
        if self._router is None:
            raise ConfigurationError("GEMINI_API_KEY is required to route questions.")
        try:
            # Validate again so fake/test models and real output follow one path.
            decision = await self._router.ainvoke(ROUTER_PROMPT.format(question=question))
            return RouteDecision.model_validate(decision)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise GenerationError(
                "Gemini could not determine an information route."
            ) from exc


