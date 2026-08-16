# AI Chatbot

A FastAPI chatbot that answers questions using uploaded documents, superhero data, or grounded web search. Gemini selects the best source, generates a context-based answer, and the API returns only the sources that successfully contributed.

## Tech stack

| Area              | Technology                                               |
| ----------------- | -------------------------------------------------------- |
| API               | FastAPI, Uvicorn                                         |
| AI                | Google Gemini chat, embeddings, and grounded search      |
| RAG               | LangChain loaders/splitters and an in-memory FAISS index |
| Validation/config | Pydantic and pydantic-settings                           |
| External API      | Superhero API through async HTTPX                        |
| Documents         | UTF-8 TXT and PDF through PyPDF                          |
| Testing           | Pytest and pytest-asyncio                                |

Python 3.11 or newer is required.

## Architecture and files

| File                  | Responsibility                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `app/main.py`         | Creates FastAPI, initializes services, builds RAG at startup, and closes clients at shutdown     |
| `app/api.py`          | Defines the `POST /ask` endpoint                                                                 |
| `app/models.py`       | Settings, request/response validation, shared data models, and application errors                |
| `app/route.py`        | Uses structured Gemini output to select the required information source                          |
| `app/rag.py`          | Loads, chunks, embeds, indexes, and searches local documents                                     |
| `app/sources.py`      | Retrieves Superhero API data and Gemini-grounded web results                                     |
| `app/generate.py`     | Deduplicates context and generates the grounded final answer                                     |
| `app/orchestrator.py` | Coordinates routing, concurrent retrieval, generation, source attribution, and dependency wiring |
| `tests/test_app.py`   | Focused tests using deterministic embeddings, mocked HTTP, and fake services                     |

The architecture is a small service pipeline with dependency injection. Every external source returns the same `ContextItem` model, so the orchestrator can process document, superhero, and web results uniformly.

## Working flow

1. At startup, TXT/PDF files in `docs/` are loaded, split, embedded, and indexed once in FAISS.
2. `POST /ask` validates and trims the incoming question.
3. Gemini routes it to `text_rag`, `superhero_api`, `web`, or the supported document-plus-superhero route.
4. The orchestrator runs all selected retrievals concurrently.
5. Successful results are normalized into `ContextItem` objects.
6. The context builder removes duplicate content and attaches source labels.
7. Gemini generates an answer using only the retrieved context.
8. The API returns the answer, selected route, and sources that actually succeeded.

In short: **validate -> route -> retrieve -> normalize -> generate -> respond**.

## Important logic

- **One-time RAG initialization:** the FAISS index is built during startup and reused for every request.
- **Structured routing:** Pydantic validates Gemini's decision and rejects unsupported source combinations.
- **Partial failure handling:** one failed source does not discard successful results from another source.
- **Grounded generation:** the answer prompt permits only facts present in retrieved context.
- **Safe document loading:** corrupt documents are skipped without stopping the server.
- **Source accuracy:** response sources come from successful retrievals, not from model-generated claims.
- **Testable design:** dependency injection allows external services to be replaced with predictable fakes.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add credentials to `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
SUPERHERO_API_TOKEN=your-superhero-api-token
```

Place readable `.txt` or `.pdf` files in `docs/`, then run:

```powershell
uvicorn app.main:app --reload
```

Restart after changing documents so the in-memory index is rebuilt. Never commit the real `.env` file.

## API example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Batman\u0027s powers?"}'
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "type": "superhero_api",
      "name": "Batman",
      "reference": null
    }
  ],
  "route": ["superhero_api"]
}
```

Interactive documentation: `http://127.0.0.1:8000/docs`.

## Tests

```powershell
pytest
```

Tests cover validation, API failure handling, one-time RAG initialization, empty/corrupt documents, and combined-source orchestration.

## Future possibilities

- Persistent vector storage, hybrid search, and reranking
- Conversation memory and streaming responses
- Authentication, rate limiting, and caching
- Structured logging, tracing, metrics, and health endpoints
- Docker, CI/CD, cloud deployment, and automated evaluations
- Additional retrieval tools and advanced workflow orchestration
