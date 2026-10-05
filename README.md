# RAG-Syntec

<img src="assets/overview.png" alt="RAG-Syntec — overview" width="760">

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

> **Corpus note.** The indexed corpus is the French Syntec collective
> bargaining agreement (IDCC 1486), sourced from Légifrance/DILA under the
> Licence Ouverte / Open Licence 2.0 (Etalab) — free reuse, including
> commercial, with attribution. Contains no personal data. See
> [`design/corpus.md`](design/corpus.md) for the full scope and licensing
> position.

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
| `eval/` | Golden question set, metrics, evaluation runner, failure taxonomy, run comparison |
| `design/guardrail.md` | Why the threshold is what it is, and what it does not catch |
| `design/evaluation.md` | The baseline measured, and why each failure happened |
| `design/retrieval_diagnosis.md` | Where the relevant chunk sits in the ranking, and the predictions made before changing anything |
| `design/reranking.md` | Why a reranking stage, how it will be judged, and what it costs |
| `tests/` | pytest suite, no network calls required |

## Run it

### With Docker Compose (recommended — Ollama + app, one command)

```bash
cp .env.example .env
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3-embedding:8b
docker compose exec ollama ollama pull gemma4:latest
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
  -d '{"question": "What is the trial period for an engineer or manager (cadre) under the Syntec agreement?"}'
```

### Tests

```bash
uv run pytest
```

No API key, no running Ollama, and no `.env` file are required to run the
test suite — see `tests/conftest.py`.

## Evaluation

Every claim in this repository is meant to be backed by a measurement that can
be re-run. The evaluation tooling lives in `eval/`.

**How it is measured**
- **Golden set:** 75 labelled questions in four classes (answerable,
  in-topic but unanswerable, off-topic, adversarial). 55 are used to tune the
  system; 20 are held out and opened once, at the end, so the final number is not
  inflated by tuning.
- **Runner:** `uv run python eval/run.py --label <name>` runs every question
  through the real pipeline and writes a result file that records the git SHA,
  the corpus hash, the models and the settings, so a number can always be traced
  to the code and documents that produced it.
- **Metrics:** recall@k and MRR for retrieval, false-refusal and
  false-acceptance rates for the guardrail (always read together: a system that
  refuses everything has a perfect false-acceptance rate), a lexical
  faithfulness proxy for answers, and latency percentiles.
- **Failure taxonomy:** `uv run python eval/failures.py` puts each failed
  question in one category (guardrail, retrieval, generation) so that each
  failure points to a different fix.
- **Comparing runs:** `uv run python eval/compare_runs.py <run A> <run B>`
  compares two runs on the same footing; one variable is changed per run.

### Baseline

The reference point is deliberately plain: Markdown loader, fixed 500-character
chunks, `qwen3-embedding:8b` embeddings in a default Chroma index (squared-L2
distance), top 3 chunks, a relevance threshold of 0.74, and `gemma4` for
generation. These are the usual defaults, used as a starting point to measure
against, not as a claim that they are best. Visible split, 55 questions:

| measure | baseline | `top_k = 10` control | reranking (20 candidates, 3 kept) |
|---|---|---|---|
| recall@3, answerable questions (n = 22) | 0.50 | 0.50 | 0.64 |
| MRR (cut at 3) | 0.38 | 0.38 | 0.42 |
| false-refusal rate | 0.12 | 0.12 | 0.12 |
| false-acceptance rate | 0.32 | 0.32 | 0.32 |
| answerable questions failing the automatic answer check | 9 of 22 | 4 of 22 | 6 of 22 |
| retrieval time, median / p95 | 2.6 / 5.7 s | 2.5 / 5.5 s | 5.5 / 10.1 s |
| generation time, median / p95 | 22.9 / 34.7 s | 25.5 / 43.8 s | 28.6 / 37.1 s |

Details, caveats and the raw files: `design/evaluation.md` and
`eval/results/`. The false-acceptance rate counts adversarial questions the
distance guardrail lets through; the language model then refused most of them
(see the taxonomy).

### What the baseline shows

- **The failures are mostly ranking failures, not missing information.** For
  all 22 answerable questions the relevant chunk is within the 10 nearest
  chunks; it is first for 6 questions and within the top 3 for 12 (55%). The five
  retrieval failures have it at ranks 5 to 10, only 0.02 to 0.13 farther than
  the best chunk returned (`design/retrieval_diagnosis.md`).
- **About 20% of the index is repeated text** (the agreement's base text and its
  amendments), which can crowd the top results.
- **Simply returning 10 chunks instead of 3 helps** (answerable failures 9 to
  4) but lengthens generation by 11% (median) to 26% (p95) and leaves the
  ranking itself unchanged.

### Reranking: tried, and rejected by its own rules

A cross-encoder reads the question and a candidate chunk together and scores the
pair; the dense retriever compares two vectors computed separately. That joint
reading can separate near-identical articles, which is what the baseline shows
failing: the right chunk is retrieved but ranked a few places too low. The
tested design fetches 20 candidates, reranks them, and keeps 3 for the prompt.

The hypothesis was written down with numbers, and with the rule that would reject
it, before it was run (`design/retrieval_diagnosis.md`): kept only if recall@3
rises by at least 0.15, it fails on no more answerable questions than
`top_k = 10`, and it adds at most 3 seconds of latency at the 95th percentile.
A published result could not settle it for this corpus: one study found that
reranking by a language model improved precision and a commercial reranker did
not (see sources).

**Result: rejected in this configuration** (`design/reranking.md`). recall@3
rose by 0.136 (0.64 against the 0.65 needed, one question short), 6 answerable
questions still fail against 4 with `top_k = 10`, and retrieval p95 rose by 4.4
s. It did fix three of the five retrieval failures that were predicted to be
fixable. It also lost two questions that the dense ranking had right. In one,
the reranker promoted an older text of the agreement ("2 years of seniority" for
the severance indemnity, where the current text says 8 months) and the system
gave that outdated answer: a cross-encoder scores how well a passage matches the
question, and cannot tell which version of a provision is in force. The code
stays as an optional feature, off by default. The finding points to handling
text versions (see `TODO.md`, T13.2) as a prerequisite, not a later
improvement. Cost measured along the way: a 1.2 GB model, 0.33 s for 20 pairs
on the GPU, which has to be shared with the Ollama models (the first attempt ran
out of GPU memory until the scoring batch was made small).

### Sources

- R. Nogueira, K. Cho, *Passage Re-ranking with BERT*, 2019,
  [arXiv:1901.04085](https://arxiv.org/abs/1901.04085): reranking passages with
  a BERT cross-encoder, reported as improving MRR@10 on MS MARCO by 27% (relative)
  over the previous state of the art. Basis for the two-stage design (retrieve,
  then rerank).
- Sentence-Transformers documentation, *Retrieve & Re-Rank*,
  [sbert.net](https://www.sbert.net/examples/applications/retrieve_rerank/README.html):
  why a fast bi-encoder is used to pick candidates and a more accurate but slower
  cross-encoder to rank them.
- M. Eibich, S. Nagpal, A. Fred-Ojala, *ARAGOG: Advanced RAG Output Grading*,
  2024, [arXiv:2404.01037](https://arxiv.org/abs/2404.01037): found that LLM
  reranking improved retrieval precision, while Cohere rerank and MMR showed no
  notable advantage over a naive RAG baseline. The counter-evidence that makes
  measuring on this corpus necessary.
- Y. Zhang et al., *Qwen3 Embedding: Advancing Text Embedding and Reranking
  Through Foundation Models*, 2025,
  [arXiv:2506.05176](https://arxiv.org/abs/2506.05176), and the
  [Qwen3-Reranker-0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
  (Apache 2.0 licence, loaded through `sentence-transformers`): the reranker
  used.
- S. Es et al., *Ragas: Automated Evaluation of Retrieval Augmented
  Generation*, 2023, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217): the
  vocabulary used for the generation metrics (faithfulness, context relevance).
  The metrics here are implemented from scratch, not taken from the library.

## Status

This is under active development. Done: a corrected, tested baseline, the
labelled golden set, the evaluation runner and failure analysis, a diagnosis of
where retrieval fails, and a reranking stage that was built, measured and
rejected against rules fixed in advance (see `design/`). Next, depending on what
the measurements show: handling of text versions (the agreement's current and
superseded provisions), structure-aware chunking of the legal text, and layered
guardrails (an abstention classifier, output groundedness checking, citation
enforcement). Hybrid dense and sparse retrieval, a different embedding model and
a Qdrant migration are kept as conditional steps, to be run only if the
diagnosis calls for them.

The full roadmap — milestones, what each one must measure before it counts
as done, and what is deliberately out of scope — lives in
[`TODO.md`](TODO.md).

## License

MIT — see [LICENSE](LICENSE).
