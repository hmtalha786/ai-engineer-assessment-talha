"""External retrieval adapters for superhero data and grounded web search."""

import json
from urllib.parse import quote

import httpx
from google import genai
from google.genai import types

from app.models import ConfigurationError, ContextItem, RetrievalError

class SuperheroService:
    """Retrieve structured character data from the Superhero API."""

    def __init__(self, token: str | None, base_url: str, timeout_seconds: float) -> None:
        self.token = token
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Release clients owned by this object during app shutdown."""
        await self.client.aclose()

    async def retrieve(self, name: str) -> ContextItem:
        """Find one hero and normalize the useful response fields."""
        if not self.token:
            raise ConfigurationError(
                "SUPERHERO_API_TOKEN is required for superhero questions."
            )
        try:
            # URL-encode user/model supplied names before adding them to the path.
            response = await self.client.get(
                f"/api/{self.token}/search/{quote(name, safe='')}"
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RetrievalError("The Superhero API timed out.", status_code=504) from exc
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"The Superhero API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise RetrievalError("The Superhero API could not be reached.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RetrievalError("The Superhero API returned malformed JSON.") from exc

        if payload.get("response") != "success" or not payload.get("results"):
            raise RetrievalError(f"No superhero named '{name}' was found.", status_code=404)

        # Prefer an exact name match; otherwise use the API's best-ranked result.
        results = payload["results"]
        hero = next(
            (
                candidate
                for candidate in results
                if str(candidate.get("name", "")).casefold() == name.casefold()
            ),
            results[0],
        )
        if not isinstance(hero, dict) or not hero.get("name"):
            raise RetrievalError("The Superhero API returned an unexpected response.")

        # Drop image URLs and other fields that do not help answer questions.
        useful_fields = {
            key: hero.get(key)
            for key in (
                "name",
                "powerstats",
                "biography",
                "appearance",
                "work",
                "connections",
            )
            if hero.get(key) is not None
        }
        return ContextItem(
            content=json.dumps(useful_fields, ensure_ascii=True),
            source_type="superhero_api",
            source_name=str(hero["name"]),
            metadata={"id": hero.get("id")},
        )


class WebSearchService:
    """Use Gemini's Google Search grounding for general/current questions."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model = model

    async def close(self) -> None:
        """Release clients owned by this object during app shutdown."""
        if self.client is not None:
            await self.client.aio.aclose()

    async def retrieve(self, question: str) -> list[ContextItem]:
        """Return a grounded web summary and any citations supplied by Gemini."""
        if self.client is None:
            raise ConfigurationError("GEMINI_API_KEY is required for web retrieval.")
        try:
            # Supplying Google Search turns this into a grounded search call.
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=(
                    "Search the web for reliable information needed to answer this question. "
                    "Return a concise factual research summary.\n\n"
                    f"Question: {question}"
                ),
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        except Exception as exc:
            raise RetrievalError("Gemini web retrieval is unavailable.") from exc

        text = (response.text or "").strip()
        if not text:
            raise RetrievalError("Gemini web retrieval returned no usable content.")

        # Grounding chunks contain the titles and URLs used by the model.
        references: list[tuple[str, str | None]] = []
        try:
            metadata = response.candidates[0].grounding_metadata
            for chunk in metadata.grounding_chunks or []:
                if chunk.web and chunk.web.uri:
                    references.append((chunk.web.title or "Web result", chunk.web.uri))
        except (AttributeError, IndexError, TypeError):
            references = []

        # Retain honest source metadata even when Gemini omits URL details.
        if not references:
            references = [("Google Search via Gemini", None)]
        return [
            ContextItem(
                content=text,
                source_type="web",
                source_name=name,
                source_reference=reference,
            )
            for name, reference in dict.fromkeys(references)
        ]


