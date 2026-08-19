# AI Chatbot

A FastAPI chatbot with one endpoint, `POST /ask`. It reads a natural-language question, works out what the question is about, pulls facts from the right source, and answers using only what it retrieved. Every response lists the sources that actually contributed.

Three sources are available, and Gemini picks between them (or combines documents + superheroes):

| Source | Used for |
| --- | --- |
| `text_rag` | The local text dataset in `docs/` (chunked, embedded, searched in FAISS) |
| `superhero_api` | Named superheroes, via [superheroapi.com](https://superheroapi.com/) |
| `web` | General or current questions that fit neither of the above |

The brief asked for two sources. The third one is deliberate: with only documents and
superheroes, the router has no correct answer for "what is the capital of Peru" and has to
misfile it into one of the two. A fallback route turns routing into a real decision — including
the decision that neither dataset applies — and keeps the other two sources honest, because the
model is never forced to reach for a source that cannot answer.

The LLM is **Google Gemini** (hosted, via AI Studio) — used twice per request: once to route, once to answer.

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key          # aistudio.google.com
SUPERHERO_API_TOKEN=your-superhero-api-token # superheroapi.com (sign in with GitHub)
```

Put any `.txt` or `.pdf` files you want searchable in `docs/`. The repo ships with three sample
documents — an engineering handbook, a security policy, and an onboarding guide — which index to
about a dozen chunks. Three overlapping-but-distinct documents mean retrieval has to pick the
right file, not just the right paragraph.

## Run

```powershell
uvicorn main:app --reload
```

The index is built once at startup, so restart after changing `docs/`. Interactive docs at `http://127.0.0.1:8000/docs`.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Batman'\''s powers?"}'
```

```json
{
  "answer": "...",
  "sources": [{ "type": "superhero_api", "name": "Batman", "reference": null }],
  "route": ["superhero_api"]
}
```

## Tests

```powershell
pytest
```

15 tests over what I considered the core logic: question and route validation, the superhero lookup (including the not-found and no-token paths), one-time index build, empty and corrupt documents, and the end-to-end pipeline with sources and partial-failure handling.

## Decisions and trade-offs

- **Two Gemini calls, not one.** Routing is a separate structured-output call so the model returns a validated `RouteDecision` rather than free text. Costs latency; buys a deterministic, testable routing step.
- **Superhero lookups take two HTTP calls.** `/search/{name}` resolves the name to an id, then `/{id}` fetches the whole record at once. Fetching per-section (`/{id}/powerstats` etc.) would mean several calls for no gain.
- **Sources come from retrievals, not the model.** The `sources` list is assembled from results that actually succeeded, so the model can't invent a citation.
- **Partial failure is allowed.** If one selected source fails, the answer is still generated from the other. Only when everything fails does the request error.
- **The index is in-memory and built at startup.** Simple and fast to read; means a restart is needed after changing `docs/`, and it doesn't survive scaling to multiple workers.
- **Corners cut:** there's no auth or rate limiting on `/ask`, no conversation memory, and when two heroes share a name (the API has two entries called "Batman" — Bruce Wayne and Terry McGinnis) the first exact match wins rather than asking the user which one they meant.
