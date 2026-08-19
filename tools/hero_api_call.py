"""Superhero API tool -- https://superheroapi.com/

Users ask by name but the API is keyed by numeric character id, so a lookup is
two calls:

    1. GET /api/{token}/search/{name}   -> find the character, read its "id"
    2. GET /api/{token}/{id}            -> fetch the whole record in one request

The second call returns powerstats, biography, appearance, work and connections
together, so there is never a reason to request those sections one at a time.
"""

import json
from typing import Any
from urllib.parse import quote

import httpx

from models import (
    ContextItem,
    configuration_error,
    retrieval_error,
    secret,
    settings,
)

# Everything worth sending to the LLM. "image" is dropped -- a portrait URL
# cannot help answer a question, and it only adds noise to the prompt.
RECORD_FIELDS = ("name", "powerstats", "biography", "appearance", "work", "connections")


async def _get_json(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    """Perform one GET against the Superhero API and map failures to AppError."""
    try:
        response = await client.get(path)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise retrieval_error("The Superhero API timed out.", status_code=504) from exc
    except httpx.HTTPStatusError as exc:
        raise retrieval_error(
            f"The Superhero API returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise retrieval_error("The Superhero API could not be reached.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise retrieval_error("The Superhero API returned malformed JSON.") from exc

    if not isinstance(payload, dict):
        raise retrieval_error("The Superhero API returned an unexpected response.")
    return payload


async def search_hero_id(client: httpx.AsyncClient, name: str) -> tuple[str, str]:
    """Call #1 -- resolve a superhero name to its (id, canonical name)."""
    # URL-encode user/model supplied names before adding them to the path.
    token = secret(settings.superhero_api_token)
    payload = await _get_json(client, f"/api/{token}/search/{quote(name, safe='')}")

    results = payload.get("results")
    if payload.get("response") != "success" or not results:
        raise retrieval_error(f"No superhero named '{name}' was found.", status_code=404)

    # Prefer an exact name match; otherwise use the API's best-ranked result.
    hero = next(
        (
            candidate
            for candidate in results
            if isinstance(candidate, dict)
            and str(candidate.get("name", "")).casefold() == name.casefold()
        ),
        results[0],
    )
    if not isinstance(hero, dict) or not hero.get("id") or not hero.get("name"):
        raise retrieval_error("The Superhero API returned an unexpected response.")

    return str(hero["id"]), str(hero["name"])


async def fetch_hero_record(client: httpx.AsyncClient, hero_id: str) -> dict[str, Any]:
    """Call #2 -- fetch every useful section for a known hero id, in one request."""
    token = secret(settings.superhero_api_token)
    payload = await _get_json(client, f"/api/{token}/{quote(hero_id, safe='')}")

    if payload.get("response") != "success":
        raise retrieval_error(
            "The Superhero API could not return this character.", status_code=404
        )

    # The full-record endpoint nests each section under its own key.
    return {
        key: payload.get(key)
        for key in RECORD_FIELDS
        if payload.get(key) is not None
    }


async def fetch_hero(name: str) -> ContextItem:
    """Run both calls and normalize the result into a ContextItem."""
    if not settings.superhero_api_token:
        raise configuration_error(
            "SUPERHERO_API_TOKEN is required for superhero questions."
        )

    # One client serves both calls, then closes -- no global connection state.
    async with httpx.AsyncClient(
        base_url=settings.superhero_api_base_url.rstrip("/"),
        timeout=httpx.Timeout(settings.external_request_timeout_seconds),
        follow_redirects=True,
    ) as client:
        hero_id, hero_name = await search_hero_id(client, name)
        details = await fetch_hero_record(client, hero_id)

    if not details:
        raise retrieval_error(
            f"The Superhero API returned no data for {hero_name}.", status_code=404
        )

    return ContextItem(
        content=json.dumps({"name": hero_name, **details}, ensure_ascii=True),
        source_type="superhero_api",
        source_name=hero_name,
        metadata={"id": hero_id},
    )
