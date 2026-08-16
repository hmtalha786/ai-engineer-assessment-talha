"""FastAPI entry point: application lifecycle and error mapping."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import router
from app.models import AppError, Settings, get_settings
from app.orchestrator import ServiceContainer


# Keep application logs useful while hiding noisy low-level HTTP client logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app; optional settings make isolated tests easy."""
    # Production uses cached environment settings; tests may inject their own.
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Own resources that should exist once per server process."""
        # Wire every dependency once and expose the graph through FastAPI state.
        services = ServiceContainer.create(resolved_settings)
        app.state.services = services
        # Build the document index once at startup, not once per question.
        await services.rag.initialize()
        try:
            # FastAPI serves requests while execution is paused at this yield.
            yield
        finally:
            # Always close external HTTP clients during graceful shutdown.
            await services.close()

    application = FastAPI(
        title="Sourced AI Chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)

    # Convert only expected, safe application errors into JSON responses.
    @application.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    return application


# Uvicorn imports this object when started with: uvicorn app.main:app
app = create_app()
