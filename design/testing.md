# Tests: what they cover and what they do not

Measured on 2026-10-06 with `uv run pytest --cov=app --cov=eval --cov=scripts`.

- **Default run** (`uv run pytest`): 203 tests, about 10 s, no network, no API
  key, no `.env` and no model needed (`tests/conftest.py` sets every setting to an
  inert value and stubs the vector store and the language model). This is what CI
  runs, with `black --check`.
- **Evaluation guard** (`uv run pytest -m eval`): one test, excluded by default
  (`addopts = "-m 'not eval'"`), that runs the retrieval guardrail on the 55
  visible golden questions against the real index and a running Ollama, and fails
  if off-topic questions get through or more than 20% of in-topic questions are
  refused. It is excluded from CI because it needs the 8 447-chunk index and the
  embedding model; it skips itself when either is missing.

**Coverage.** 88% of `app/` (567 statements, 67 missed); 60% overall including
the evaluation and corpus scripts.

| part | coverage | what is not covered |
|---|---|---|
| `app/services/pipeline.py`, `guardrail.py`, `abstention.py`, `features.py` | 100% | |
| `app/api/routes.py`, `retrieval.py` | 98% | |
| `app/services/ingestion.py` | 72% | loading real files, PDF pages, building the embedding client and the Chroma collection |
| `app/services/generation.py` | 82% | the OpenAI provider branch |
| `eval/run.py` | 0% | the runner is only exercised by real evaluation runs |
| `scripts/fetch_corpus.py`, `fetch_kali_id.py` | 0% | network acquisition from Légifrance |

**What worries me most:** `eval/run.py`. Every number in `design/` comes from it,
and a bug there would bias every comparison at once. Two bugs of that kind were
found by reading results rather than by tests (chunk texts read with the wrong
chunking in `compare_runs.py`; golden labels left stale in saved runs), each now
covered by a test. The runner's metric functions are partly covered through
`eval/metrics.py` (92%), but not its orchestration.

**Markers.** The roadmap asked for `unit`, `integration` and `eval` markers. Only
`eval` exists: every other test is hermetic, including the API tests that go
through FastAPI's test client, so a separate `integration` marker would not
change what runs where.
