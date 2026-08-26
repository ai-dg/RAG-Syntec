# ragfoundry

A RAG pipeline that knows when to refuse to answer — and can prove it.

Most retrieval-augmented generation demos always answer, even when nothing
relevant was retrieved. This one has a relevance guardrail that is
**measured, not guessed**: the threshold is calibrated against the actual
distance distribution of in-topic vs. off-topic questions on the indexed
corpus, and that calibration is verifiable with a script, not just claimed
in prose. See [`design/guardrail.md`](design/guardrail.md).

> **Origin note.** The application skeleton started from a take-home RAG
> exercise. The architecture, the bug fixes, the guardrail calibration, the
> tests, the Docker/Compose setup, and everything documented in `design/`
> are original work built on top of that starting point.

## What's here

| Path | Role |
|---|---|
| `app/main.py` | FastAPI entry point; builds the index once at startup via `lifespan` |
| `app/config.py` | Settings (Pydantic), `openai`/`ollama` provider modes |
| `app/api/routes.py` | `POST /query`, `GET /health` |
| `app/services/ingestion.py` | Multi-format loading, validation, chunking, idempotent indexing |
| `app/services/retrieval.py` | Vector search + relevance guardrail |
| `app/services/generation.py` | Context-constrained prompt, LLM call |
| `scripts/inspect_metric.py` | Proves what the guardrail's score actually measures |
| `design/guardrail.md` | Why the threshold is what it is, and what it does not catch |
| `tests/` | pytest suite, no network calls required |

## Run it

### With Docker Compose (recommended — Ollama + app, one command)

```bash
cp .env.example .env
docker compose up --build
```

### Locally, with uv

```bash
uv sync
cp .env.example .env   # fill in the values for your chosen provider
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this guardrail actually measure?"}'
```

### Tests

```bash
uv run pytest
```

No API key, no running Ollama, and no `.env` file are required to run the
test suite — see `tests/conftest.py`.

## Status

This is under active development. Current state: a corrected, tested
baseline (see `design/` for what was fixed and why). Planned next: hybrid
retrieval (BM25 + dense) with reranking, layered guardrails (input
validation, output groundedness checking, citation enforcement), and a
quantitative evaluation harness (recall, false-refusal rate, false-acceptance
rate) instead of eyeballed distance thresholds.

## License

MIT — see [LICENSE](LICENSE).
