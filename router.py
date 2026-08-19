"""Step 2 of the request pipeline: decide which sources a question needs."""


from models import (
    RouteDecision,
    configuration_error,
    generation_error,
    get_chat_model,
)

# The first model call chooses data sources; it never writes the final answer.
ROUTER_PROMPT = """You route questions to information sources.

Available routes:
- text_rag: questions about documents supplied to this application.
- superhero_api: factual questions about one or more named superheroes.
- web: general or current questions not about the supplied documents or superheroes.
- text_rag + superhero_api: questions explicitly requiring both document content and superhero data.

Use only one of those exact route choices. Do not answer the question and do not call tools.

When superhero_api is selected, put the canonical superhero names in
superhero_names. The tool always fetches the complete record for each name
(powerstats, biography, appearance, work and connections), so do not try to
narrow it down to one detail.

Question: {question}
"""

# Only combinations the retrieval step implements are accepted.
ALLOWED_ROUTES = {
    ("text_rag",),
    ("superhero_api",),
    ("web",),
    ("text_rag", "superhero_api"),
    ("superhero_api", "text_rag"),
}


def normalize_route(decision: RouteDecision) -> RouteDecision:
    """Validate and tidy the model's decision before the pipeline acts on it."""
    # Preserve order while removing accidental duplicates from model output.
    decision.sources = list(dict.fromkeys(decision.sources))

    if tuple(decision.sources) not in ALLOWED_ROUTES:
        raise generation_error("Gemini selected an unsupported source combination.")

    if "superhero_api" in decision.sources:
        if not decision.superhero_names:
            raise generation_error("Gemini selected the superhero route without a name.")
    else:
        # Names are meaningless without the superhero route.
        decision.superhero_names = []

    return decision


async def decide_route(question: str) -> RouteDecision:
    """Ask Gemini for a validated RouteDecision instead of free-form text."""
    llm = get_chat_model()
    if llm is None:
        raise configuration_error("GEMINI_API_KEY is required to route questions.")

    try:
        # LangChain parses model output directly into the Pydantic route model.
        raw = await llm.with_structured_output(RouteDecision).ainvoke(
            ROUTER_PROMPT.format(question=question)
        )
        # Validate again so fake/test models and real output follow one path.
        decision = RouteDecision.model_validate(raw)
    except Exception as exc:
        raise generation_error("Gemini could not determine an information route.") from exc

    return normalize_route(decision)
