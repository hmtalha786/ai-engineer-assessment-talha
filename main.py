from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api import api_router
from models import AppError, settings
from tools.rag_generate import build_index, find_documents


# --- error handling ---------------------------------------------------------
async def handle_app_error(_request: Request, exc: Exception) -> JSONResponse:
    """Convert expected application errors into a clean JSON response."""
    error = exc if isinstance(exc, AppError) else AppError(
        "Internal server error.")
    return JSONResponse(status_code=error.status_code, content={"detail": error.message})


# --- FastAPI app creation ---------------------------------------------------
def create_app() -> FastAPI:
    """Step 1: build the FastAPI server and register the endpoint and errors."""
    application = FastAPI(title="Sourced AI Chatbot", version="2.0.0")
    application.include_router(api_router)
    application.add_exception_handler(AppError, handle_app_error)
    return application


# --- document loading -------------------------------------------------------
def load_documents_into_index() -> None:
    """Steps 2 and 3: scan ./docs and build the vector index once, at startup.

    This is the only place the app writes to the console, so a reader can see
    the whole startup result in one function.
    """
    documents = find_documents(settings.docs_dir)
    if not documents:
        # Not fatal: the superhero and web routes still work without documents.
        print(f"[startup] No .txt or .pdf files in {settings.docs_dir}")
        print("[startup] The document route will be unavailable.")
        return

    names = ", ".join(path.name for path in documents)
    print(
        f"[startup] Found {len(documents)} document(s) in {settings.docs_dir}: {names}")

    # Parse -> chunk -> embed -> store in FAISS, with the file name as metadata.
    index = build_index(settings.docs_dir)

    # build_index never raises; it reports what happened through the index state.
    for name, error in index["errors"].items():
        print(f"[startup] Skipped unreadable file {name} ({error})")
    if index["retriever"] is None:
        print(
            f"[startup] The document route will be unavailable: {index['reason']}")
    else:
        print(
            f"[startup] Indexed {index['chunks']} chunk(s) from {len(index['files'])} file(s)")


# --- what actually runs when uvicorn imports this module -------------------
app = create_app()
load_documents_into_index()
