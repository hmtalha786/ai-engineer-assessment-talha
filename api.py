"""The POST /ask endpoint.

This file is the map of a request. `ask()` below runs the pipeline one numbered
step at a time, and each step is a single call into the file that owns that
logic. Read `ask()` top to bottom to understand the whole flow, then open the
named file for the details of any one step.
"""

import asyncio

from fastapi import APIRouter

from generate import build_context, generate_answer
from models import (
    AppError,
    AskRequest,
    AskResponse,
    ContextItem,
    RouteDecision,
    Source,
    clean_question,
    retrieval_error,
    to_source,
)
from router import decide_route
from tools.hero_api_call import fetch_hero
from tools.rag_retrieve import retrieve_documents
from tools.web_search import search_web

api_router = APIRouter()


async def ask(payload: AskRequest) -> AskResponse:
    """Answer one question: validate -> route -> retrieve -> generate -> respond."""

    # ---- STEP 1: validate and clean the incoming question ------------------
    # -> models.clean_question
    question = clean_question(payload.question)

    # ---- STEP 2: work out the intent and pick the information sources ------
    # -> router.decide_route  (one Gemini call returning a structured decision)
    decision = await decide_route(question)

    # ---- STEP 3: run every selected tool, concurrently ---------------------
    # -> tools/rag_retrieve.py, tools/hero_api_call.py, tools/web_search.py
    items, failures = await run_selected_tools(question, decision)

    # ---- STEP 4: stop only if every selected source failed -----------------
    if not items:
        if failures:
            # Surface the first real failure rather than a generic message.
            raise failures[0]
        raise retrieval_error(
            "No information was retrieved for this question.", status_code=404
        )

    # ---- STEP 5: turn the results into one grounded prompt context ---------
    # -> generate.build_context
    context = build_context(items)

    # ---- STEP 6: generate the final answer from that context only ----------
    # -> generate.generate_answer  (the second and last Gemini call)
    answer = await generate_answer(question, context)

    # ---- STEP 7: list the sources that actually contributed ----------------
    sources = collect_sources(items)

    # ---- STEP 8: respond ---------------------------------------------------
    return AskResponse(answer=answer, sources=sources, route=decision.sources)


async def run_selected_tools(
    question: str, decision: RouteDecision
) -> tuple[list[ContextItem], list[AppError]]:
    """Step 3 helper: fan out to the chosen tools and keep partial successes."""
    # Build a coroutine only for the sources the router actually selected.
    tasks = []
    if "text_rag" in decision.sources:
        tasks.append(retrieve_documents(question))
    if "superhero_api" in decision.sources:
        # One two-call lookup per named hero, all in parallel.
        tasks.extend(fetch_hero(name) for name in decision.superhero_names)
    if "web" in decision.sources:
        tasks.append(search_web(question))

    # Independent retrievals run together; one failure must not cancel the rest.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    items: list[ContextItem] = []
    failures: list[AppError] = []
    for result in results:
        # An expected AppError is kept so step 4 can report the real reason.
        if isinstance(result, AppError):
            failures.append(result)
        elif isinstance(result, Exception):
            failures.append(retrieval_error(
                "An information source failed unexpectedly."))
        elif isinstance(result, list):
            items.extend(result)
        else:
            items.append(result)
    return items, failures


def collect_sources(items: list[ContextItem]) -> list[Source]:
    """Step 7 helper: deduplicate source metadata without reordering results."""
    sources: list[Source] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in items:
        source = to_source(item)
        key = (source.type, source.name, source.reference)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


# Registered without a decorator so the route is visible as an ordinary call.
api_router.add_api_route(
    "/ask", ask, methods=["POST"], response_model=AskResponse)
