"""Build and query the document RAG index."""

import asyncio
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import ContextItem, RetrievalError


logger = logging.getLogger(__name__)

class RAGService:
    """Load documents once at startup and reuse an in-memory FAISS index."""

    def __init__(
        self,
        docs_dir: Path,
        embeddings: Embeddings | None,
        chunk_size: int,
        chunk_overlap: int,
        result_count: int,
        score_threshold: float = 0.52,
    ) -> None:
        self.docs_dir = docs_dir
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.result_count = result_count
        self.score_threshold = score_threshold
        self.retriever = None
        self.initialization_count = 0
        self.loaded_files: list[str] = []
        self.load_errors: dict[str, str] = {}
        self.unavailable_reason = "The document index has not been initialized."

    async def initialize(self) -> None:
        """Load, chunk, embed, and index documents once during app startup."""
        self.initialization_count += 1
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        # PDF/text loading is blocking, so move it off the async event loop.
        documents = await asyncio.to_thread(self._load_documents)
        # Startup still succeeds without document search; other routes remain usable.
        if not documents:
            self.unavailable_reason = (
                "No readable .txt or .pdf documents are available in the docs directory."
            )
            logger.warning(self.unavailable_reason)
            return
        if self.embeddings is None:
            self.unavailable_reason = "GEMINI_API_KEY is required to build the document index."
            logger.warning(self.unavailable_reason)
            return

        # Overlapping chunks retain context near chunk boundaries.
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            self.unavailable_reason = "The supported documents did not contain extractable text."
            logger.warning(self.unavailable_reason)
            return

        try:
            # Embedding/index construction is expensive and intentionally runs once.
            vector_store = await asyncio.to_thread(
                FAISS.from_documents,
                chunks,
                self.embeddings,
            )
            self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.result_count,
                    "score_threshold": self.score_threshold,
                },
            )
            self.unavailable_reason = ""
            logger.info(
                "RAG index initialized once with %s chunks from %s files.",
                len(chunks),
                len(self.loaded_files),
            )
        except Exception as exc:
            self.unavailable_reason = "The document index could not be built."
            logger.exception(self.unavailable_reason)
            self.retriever = None
            self.load_errors["index"] = type(exc).__name__

    def _load_documents(self) -> list[Document]:
        """Read supported files while recording, rather than raising, bad files."""
        documents: list[Document] = []
        # Stable ordering makes logs and tests deterministic.
        supported_files = sorted(
            path
            for path in self.docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
        )
        for path in supported_files:
            try:
                if path.suffix.lower() == ".pdf":
                    # Loader selection is based only on the supported file extension.
                    loaded = PyPDFLoader(str(path)).load()
                else:
                    loaded = TextLoader(str(path), encoding="utf-8").load()
                for document in loaded:
                    document.metadata["source"] = path.name
                documents.extend(loaded)
                self.loaded_files.append(path.name)
                # One corrupt document must not make every document route unavailable.
            except Exception as exc:
                logger.warning("Skipping unreadable document %s: %s", path.name, exc)
                self.load_errors[path.name] = type(exc).__name__
        return documents

    async def retrieve(self, question: str) -> list[ContextItem]:
        """Return relevant document chunks in the shared ContextItem format."""
        if self.retriever is None:
            raise RetrievalError(self.unavailable_reason, status_code=503)
        try:
            documents = await self.retriever.ainvoke(question)
        except Exception as exc:
            raise RetrievalError("Document retrieval failed.") from exc
        # An empty similarity search is a valid not-found result, not a server crash.
        if not documents:
            raise RetrievalError("No relevant document passages were found.", status_code=404)

        return [
            ContextItem(
                content=document.page_content,
                source_type="text_rag",
                source_name=Path(document.metadata.get("source", "document")).name,
                metadata={
                    key: value
                    for key, value in document.metadata.items()
                    if key != "source"
                },
            )
            for document in documents
            if document.page_content.strip()
        ]


