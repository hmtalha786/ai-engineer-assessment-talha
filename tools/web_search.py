"""Grounded web search through Gemini's Google Search tool."""


from models import ContextItem, configuration_error, retrieval_error, secret, settings

SEARCH_PROMPT = (
    "Search the web for reliable information needed to answer this question. "
    "Return a concise factual research summary.\n\n"
    "Question: {question}"
)


def extract_references(response: object) -> list[tuple[str, str | None]]:
    """Read the titles and URLs Gemini used, tolerating a missing grounding block."""
    references: list[tuple[str, str | None]] = []
    try:
        metadata = response.candidates[0].grounding_metadata
        for chunk in metadata.grounding_chunks or []:
            if chunk.web and chunk.web.uri:
                references.append((chunk.web.title or "Web result", chunk.web.uri))
    except (AttributeError, IndexError, TypeError):
        references = []

    # Retain honest source metadata even when Gemini omits URL details.
    return references or [("Google Search via Gemini", None)]


async def search_web(question: str) -> list[ContextItem]:
    """Return a grounded web summary plus any citations supplied by Gemini."""
    if not settings.gemini_api_key:
        raise configuration_error("GEMINI_API_KEY is required for web retrieval.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=secret(settings.gemini_api_key))
    try:
        # Supplying Google Search turns this into a grounded search call.
        response = await client.aio.models.generate_content(
            model=settings.gemini_chat_model,
            contents=SEARCH_PROMPT.format(question=question),
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as exc:
        raise retrieval_error("Gemini web retrieval is unavailable.") from exc

    text = (response.text or "").strip()
    if not text:
        raise retrieval_error("Gemini web retrieval returned no usable content.")

    return [
        ContextItem(
            content=text,
            source_type="web",
            source_name=name,
            source_reference=reference,
        )
        for name, reference in dict.fromkeys(extract_references(response))
    ]
