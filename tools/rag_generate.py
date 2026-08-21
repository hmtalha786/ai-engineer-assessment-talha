from pathlib import Path
from typing import Any

from models import settings

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Module-level index state. rag_retrieve.py reads this through get_index().
_index: dict[str, Any] = {
    "retriever": None,
    "reason": "The document index has not been initialized.",
    "files": [],
    "errors": {},
    "chunks": 0,
}

# Supported document file extensions.
SUPPORTED_SUFFIXES = {".txt", ".pdf"}

# Reason for no documents being available.
NO_DOCUMENTS_REASON = (
    "No readable .txt or .pdf documents are available in the docs directory."
)


# --- index state access -------------------------------------------------------
def get_index() -> dict[str, Any]:
    """Expose the shared index state to the retrieval tool."""
    return _index


# --- document discovery and loading ------------------------------------------
def find_documents(docs_dir: Path) -> list[Path]:
    """Return the supported files in the docs folder, in a stable order."""
    if not docs_dir.is_dir():
        return []
    # Stable ordering makes output and tests deterministic.
    return sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


# --- document parsing, chunking, and indexing --------------------------------
def load_documents(paths: list[Path]) -> list[Any]:
    """Read supported files, recording rather than raising on a bad file."""

    documents: list[Any] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".pdf":
                # Loader selection is based only on the supported file extension.
                loaded = PyPDFLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            # The file name is the metadata used to attribute answers.
            for document in loaded:
                document.metadata["source"] = path.name
            documents.extend(loaded)
            _index["files"].append(path.name)
        except Exception as exc:
            # One corrupt document must not make every document route
            # unavailable; the file is noted and the rest still index.
            _index["errors"][path.name] = type(exc).__name__
    return documents


# --- document chunking -------------------------------------------------------
def split_documents(documents: list[Any]) -> list[Any]:
    """Split documents into overlapping chunks that retain boundary context."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    return splitter.split_documents(documents)


# --- document embedding and FAISS index construction -------------------------
def build_index(docs_dir: Path | None = None, embeddings: Any = None) -> dict[str, Any]:
    """Parse -> chunk -> embed -> store in FAISS. Runs exactly once at startup.

    Never raises: the server must still start (and the superhero/web routes must
    still work) when there are no documents or no API key.
    """
    from models import get_embeddings

    resolved_dir = docs_dir if docs_dir is not None else settings.docs_dir
    resolved_embeddings = embeddings if embeddings is not None else get_embeddings()

    # Reset state so a rebuild never reports stale files or errors.
    _index.update({"retriever": None, "files": [], "errors": {}, "chunks": 0})
    resolved_dir.mkdir(parents=True, exist_ok=True)

    paths = find_documents(resolved_dir)
    if not paths:
        _index["reason"] = NO_DOCUMENTS_REASON
        return _index

    documents = load_documents(paths)
    if not documents:
        _index["reason"] = NO_DOCUMENTS_REASON
        return _index

    if resolved_embeddings is None:
        _index["reason"] = "GEMINI_API_KEY is required to build the document index."
        return _index

    chunks = split_documents(documents)
    if not chunks:
        _index["reason"] = "The supported documents did not contain extractable text."
        return _index

    try:
        from langchain_community.vectorstores import FAISS

        # Embedding and index construction are expensive and intentionally run once.
        vector_store = FAISS.from_documents(chunks, resolved_embeddings)
        _index["retriever"] = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.rag_result_count,
                "score_threshold": settings.rag_score_threshold,
            },
        )
        _index["chunks"] = len(chunks)
        _index["reason"] = ""
    except Exception as exc:
        _index["reason"] = "The document index could not be built."
        _index["retriever"] = None
        _index["errors"]["index"] = type(exc).__name__

    return _index
