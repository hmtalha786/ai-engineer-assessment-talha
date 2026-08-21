from pathlib import Path

from models import ContextItem, retrieval_error
from tools.rag_generate import get_index


# --- document retrieval ------------------------------------------------------
async def retrieve_documents(question: str) -> list[ContextItem]:
    """Return the relevant document chunks in the shared ContextItem format."""
    index = get_index()
    retriever = index["retriever"]

    # No index means the route is unavailable, not that the request was bad.
    if retriever is None:
        raise retrieval_error(index["reason"], status_code=503)

    try:
        documents = await retriever.ainvoke(question)
    except Exception as exc:
        raise retrieval_error("Document retrieval failed.") from exc

    # An empty similarity search is a valid not-found result, not a server crash.
    if not documents:
        raise retrieval_error("No relevant document passages were found.", status_code=404)

    return [
        ContextItem(
            content=document.page_content,
            source_type="text_rag",
            # The file name stored at index time is what the answer cites.
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
