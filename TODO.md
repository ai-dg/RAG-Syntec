# ragfoundry — implementation roadmap and learning path

**Target:** a RAG system over the French *Syntec* collective agreement that
knows when to refuse to answer, and proves it with measured numbers.
Audiences: recruiters (M2 ML/AI Engineer internship) and freelance clients,
weighted equally. Local-first; public deployment optional and last.

**How to use this file.** Each task is sized to be roughly one coherent Git
commit and one study session. Work top to bottom — the order encodes real
dependencies, not preference. Work through the *Concepts* list **before**
writing code for that task: understand it first, then build it.

**Task template.** Every task carries: Goal · Concepts · Files · Subtasks ·
Tests · Metrics · Acceptance · Interview questions · Commit.

**Non-negotiable rule:** a task is done
when a command proves it, not when the code looks right. "Should work" is
not an acceptance criterion.

**Note on ordering.** The requested sequence had the evaluation set (#3)
before source acquisition (#4). Those are swapped below: you cannot write
questions about documents you do not have yet. Everything else follows the
requested order.

**Validated stack decisions (2026-09-02):** corpus = Syntec collective
agreement · Docling for parsing · BGE-M3 for embeddings · Qdrant for storage ·
hybrid dense+sparse retrieval · LangChain kept for ingestion only · Qwen3
reranker · full evaluation harness · Langfuse last and only if time allows.

---

## Conventions used throughout

- **Metrics log.** Every measurement is appended to `eval/results/` as
  `<date>_<phase>.json`, and summarised in `eval/RESULTS.md` as a table row.
  A number that exists only in a terminokal you closed did not happen.
- **Design docs.** Each phase that makes a real decision ends with a
  `design/<topic>.md` in the Problem → Decision → Why → Result (measured) →
  Cost format, as `design/guardrail.md` already does.
- **Commits.** Conventional Commits per `COMMITS.md`.
- **Never break the baseline.** `uv run pytest` must stay green at every
  commit. Currently: `7 passed in 1.00s`.

---

# Phase 0 — Audit of the existing system

Goal of the phase: know exactly what you have before changing any of it, and
make the repository's existing claims true.

### [ ] T0.1 — Reproduce and record the baseline

- **Goal.** Establish a factual starting point that later work can be
  compared against.
- **Concepts.** Reproducibility; why a baseline recorded after changes began
  is worthless.
- **Files.** `design/audit.md` (new).
- **Subtasks.**
  1. Run `uv run pytest -q`; record the exact output.
  2. Record hardware: `nvidia-smi`, `free -g`, `nproc`.
  3. Record available models: `curl -s localhost:11434/api/tags`.
  4. Count current chunks by loading + splitting without embedding.
  5. Write it all into `design/audit.md` with the command beside each number.
- **Tests.** None (measurement task).
- **Metrics.** test count and duration; chunk count; VRAM; models available.
- **Acceptance.** A reader can re-run every command in `design/audit.md` and
  get the same numbers.
- **Interview questions.**
  - Why record a baseline before touching anything?
  - Your GPU has 8 GB. Which of your configured models fit, and which does not?
- **Commit.** `docs: record measured baseline audit of the existing system`

### [ ] T0.2 — Declare the undeclared dependencies

- **Goal.** Make the dependency manifest honest.
- **Concepts.** Direct vs. transitive dependencies; why relying on a
  transitive dependency is a latent break.
- **Files.** `pyproject.toml`.
- **Subtasks.**
  1. Confirm the gap: `grep -c 'pydantic-settings\|numpy' pyproject.toml` → `0`.
  2. Add `pydantic`, `pydantic-settings`, `numpy` to `dependencies`.
  3. `uv sync` and re-run the suite.
- **Tests.** `uv run pytest -q` still green.
- **Acceptance.** Every module imported by `app/` and `scripts/` appears in
  `pyproject.toml`.
- **Interview questions.**
  - `app/config.py` imports `pydantic_settings` and it works today. Why is
    that still a bug?
- **Commit.** `fix: declare pydantic, pydantic-settings and numpy as direct dependencies`

### [ ] T0.3 — Make `sources` deterministic

- **Goal.** Remove non-determinism that would corrupt every later measurement.
- **Concepts.** Python string hash randomisation (`PYTHONHASHSEED`); why set
  iteration order varies across processes; determinism as a testability
  precondition.
- **Files.** `app/services/generation.py:48`, `tests/test_generation.py` (new).
- **Subtasks.**
  1. Reproduce first: run `python3 -c "print(list({'a.md','b.md','c.md'}))"`
     five times in separate processes and observe different orderings.
  2. Write the failing regression test.
  3. Replace `list({...})` with a deduplicated, sorted list.
- **Tests.** New test asserts a stable, sorted `sources` list.
- **Acceptance.** The test fails on the old code and passes on the new.
- **Interview questions.**
  - Why does `list(set_of_strings)` change order between runs but not within
    one run?
  - Why does this matter more once an evaluation harness exists?
- **Commit.** `fix: return sources in a deterministic sorted order`

### [ ] T0.4 — Cache `Settings()`

- **Goal.** Stop re-reading and re-validating `.env` on every request.
- **Concepts.** `functools.lru_cache` on a settings factory; FastAPI
  dependency injection; why module-level instantiation would break the tests.
- **Files.** `app/config.py`, `app/services/retrieval.py:49`,
  `app/services/generation.py:43`.
- **Subtasks.**
  1. Add a `get_settings()` factory decorated with `lru_cache`.
  2. Replace the two per-request `Settings()` calls.
  3. Ensure the test fixture can still override the environment (clear the
     cache between tests).
- **Tests.** Existing suite green; add a test that the same object is
  returned twice.
- **Acceptance.** No `Settings()` construction remains in a request path.
- **Interview questions.**
  - Why not just create a module-level `settings = Settings()`?
  - How do your tests override a cached settings object?
- **Commit.** `refactor: cache Settings behind an lru_cache factory`

### [ ] T0.5 — Cover the API surface with tests

- **Goal.** Test the request → guardrail → refusal path end to end, which is
  currently untested.
- **Concepts.** FastAPI `TestClient`; dependency overrides; why injecting a
  fake store beats patching by import path.
- **Files.** `tests/test_api.py` (new).
- **Subtasks.**
  1. Add `httpx` to dev dependencies if needed by `TestClient`.
  2. Test the refusal path: fake store returning a far result → HTTP 200,
     `context_found: false`, the canned refusal string, empty sources.
  3. Test the accept path with a mocked generation call.
  4. Test input validation: empty question and a 2001-character question.
- **Tests.** The three above.
- **Acceptance.** Test count rises; `app/api/routes.py` is exercised.
- **Interview questions.**
  - Why inject the vector store rather than patch
    `app.services.retrieval.get_vector_store`?
  - Why does a refusal return HTTP 200 and not 404?
- **Commit.** `test: cover the query API refusal and accept paths`

### [ ] T0.6 — Emit real structured JSON logs

- **Goal.** Make `logger.py` do what its docstring already claims.
- **Concepts.** Structured vs. unstructured logging; one JSON object per
  line; why a `request_id` field must exist before you need it.
- **Files.** `app/logger.py`.
- **Subtasks.**
  1. Read the docstring (claims JSON) against `logger.py:14` (pipe-delimited
     text) — confirm the mismatch.
  2. Implement a JSON formatter: timestamp, level, logger, message, plus
     arbitrary extra fields.
  3. Reserve a `request_id` field, populated later in Phase 15.
- **Tests.** A test asserting a log line parses as JSON and carries the
  expected keys.
- **Acceptance.** `uv run uvicorn app.main:app` emits parseable JSON lines.
- **Interview questions.**
  - What does structured logging buy you that grep on text does not?
  - Why is a correlation ID worth adding before you have a use for it?
- **Commit.** `feat: emit structured JSON logs`

### [ ] T0.7 — Write the missing calibration script

- **Goal.** Close the repository's most damaging gap: the README promises the
  threshold "is verifiable with a script", and no such script exists.
- **Concepts.** Score distribution; separating two clusters; why a threshold
  is an operating point, not a constant of nature.
- **Files.** `scripts/calibrate_threshold.py` (new), `design/guardrail.md`.
- **Subtasks.**
  1. Define two labelled question lists: clearly in-topic, clearly off-topic.
  2. For each question, record the best distance returned by the store.
  3. Print both distributions (min / median / max) and the gap between them.
  4. Print a recommended threshold and the equivalent cosine similarity.
  5. Run it against the current 42-chunk corpus.
  6. Update `design/guardrail.md` with the **real** numbers. If they differ
     from the quoted 0.43–0.73 / ~1.31, say so explicitly — a documented
     correction is a stronger artifact than a number that happened to match.
- **Tests.** Not a pytest target (needs a live index); name it so pytest
  ignores it, as `inspect_metric.py` already does.
- **Metrics.** in-topic distribution; off-topic distribution; separation gap;
  recommended threshold.
- **Acceptance.** Every number in `design/guardrail.md` is reproducible by
  one command.
- **Interview questions.**
  - Your threshold is 0.9. Zero point nine of *what*, exactly?
  - What happens to this threshold if you change the embedding model, and why?
  - What does this guardrail *not* catch?
- **Commit.** `feat: add threshold calibration script and correct the design doc`

---

# Phase 1 — Define the Syntec corpus precisely

### [ ] T1.1 — Fix the corpus scope

- **Goal.** Decide exactly which texts are in, which are out, and why.
  Scope drift after the evaluation set exists invalidates every metric.
- **Concepts.** Corpus boundaries; how corpus density affects distance
  distributions and therefore the guardrail threshold.
- **Files.** `design/corpus.md` (new).
- **Subtasks.**
  1. Identify the Syntec agreement on Légifrance and record its **IDCC
     number** — commonly cited as 1486, but *verify it on Légifrance and
     record what you actually found*, do not copy it from here.
  2. Decide the scope: base text only, or base text + amendments
     (*avenants*) + salary grids. Recommended: base text + salary grids,
     amendments excluded for v1 and added in Phase 13.
  3. Decide the target size (aim 3,000–15,000 chunks) and check the real
     document volume against it.
  4. Write the inclusion/exclusion rules down.
- **Tests.** None.
- **Acceptance.** `design/corpus.md` states what is in, what is out, and why,
  in a way that a future you cannot reinterpret.
- **Interview questions.**
  - Why must the corpus be frozen before writing evaluation questions?
  - Why did you exclude amendments in the first version?
- **Commit.** `docs: define the Syntec corpus scope`

### [ ] T1.2 — Record the licensing position

- **Goal.** Be able to answer "are you allowed to use this?" without
  hesitating — a question a freelance client *will* ask.
- **Concepts.** Licence Ouverte 2.0; DILA open data; the difference between
  public legal texts and personal data under GDPR.
- **Files.** `design/corpus.md`, `README.md` (a short note).
- **Subtasks.**
  1. Record the licence covering Légifrance/DILA data and the decree that
     establishes free reuse.
  2. State explicitly that the corpus contains **no personal data**, so the
     project raises no GDPR processing question.
  3. Record the attribution the licence requires, and comply with it.
- **Tests.** None.
- **Acceptance.** A one-paragraph answer exists, with sources.
- **Interview questions.**
  - Under what licence is this data, and what does it require of you?
  - A client asks whether they can run this on their own HR files. What
    changes, legally and technically?
- **Commit.** `docs: record corpus licensing and GDPR position`

---

# Phase 2 — Acquire and version the sources

*(Requested as #4; moved before the evaluation set because questions require
documents that exist.)*

### [ ] T2.1 — Acquire the documents

- **Goal.** Get the corpus onto disk reproducibly, not by manual clicking.
- **Concepts.** Légifrance/DILA open data; the PISTE API (free, requires
  registration); rate limits and quotas; scripted acquisition as a
  reproducibility guarantee.
- **Files.** `scripts/fetch_corpus.py` (new), `data/raw/` (git-ignored).
- **Subtasks.**
  1. Register for PISTE API access, or choose documented manual download if
     the API proves disproportionate for a one-off corpus.
  2. Write the fetch script; make it idempotent and resumable.
  3. Store raw files under `data/raw/`; add it to `.gitignore`.
  4. Keep the existing `docs/` fixture as `docs_smoke/` for fast tests —
     **do not delete it**, the test suite depends on a small corpus.
- **Tests.** A test that the fetch script's parsing helpers work on a small
  saved fixture, without network.
- **Metrics.** number of documents; total bytes; acquisition date.
- **Acceptance.** `uv run python scripts/fetch_corpus.py` reconstructs the
  corpus from nothing on a clean machine.
- **Interview questions.**
  - Why script acquisition rather than commit the documents?
  - How would you handle the agreement being amended next month?
- **Commit.** `feat: add reproducible Syntec corpus acquisition script`

### [ ] T2.2 — Version and pin the corpus

- **Goal.** Make "which corpus produced this number?" answerable months later.
- **Concepts.** Content hashing; manifests; data versioning without a heavy
  tool; why corpus identity is part of a metric's meaning.
- **Files.** `data/manifest.json` (committed), `scripts/fetch_corpus.py`.
- **Subtasks.**
  1. Compute a SHA-256 per file and a single corpus-level hash.
  2. Record per document: source URL, retrieval date, legal version date,
     size, hash.
  3. Commit the manifest (not the documents).
  4. Add a verification mode that fails if local files drift from the
     manifest.
- **Tests.** A test that manifest verification detects a modified file.
- **Metrics.** corpus hash; document count; per-document version dates.
- **Acceptance.** Every future metric can name the corpus hash it was
  measured on.
- **Interview questions.**
  - Why is a corpus hash part of an evaluation result?
  - You are not using DVC or Git LFS. Defend that.
- **Commit.** `feat: version the corpus with a hashed manifest`

---

# Phase 3 — Build the evaluation set

*(Requested as #3.)*

### [ ] T3.1 — Design the evaluation schema

- **Goal.** Decide what a labelled example *is* before labelling a hundred of
  them. Relabelling later is the expensive mistake.
- **Concepts.** The four behaviour classes; why aggregate accuracy hides the
  interesting failures; ground truth at chunk level vs. answer level.
- **Files.** `eval/README.md` (new), `eval/schema.json` (new).
- **Subtasks.**
  1. Define the JSONL record: `id`, `question`, `class`, `relevant_chunk_ids`,
     `expected_answer_gist`, `notes`, `corpus_hash`.
  2. Define the four classes and the single correct behaviour for each:

     | Class | Correct behaviour |
     |---|---|
     | `in_topic_answerable` | retrieve, answer, cite |
     | `in_topic_unanswerable` | retrieve, then say "I don't know" |
     | `off_topic` | refuse before generation |
     | `adversarial` | refuse; never leak the system prompt |
  3. Write the labelling protocol: how you decide a question is
     unanswerable, how you pick relevant chunks, what to do on ties.
- **Tests.** A schema-validation test over the (still empty) JSONL file.
- **Acceptance.** Someone else could label examples consistently from your
  protocol alone.
- **Interview questions.**
  - Why are `in_topic_unanswerable` and `off_topic` different classes when
    both end in a refusal?
  - Why label relevant chunks rather than just correct answers?
- **Commit.** `feat: define the evaluation set schema and labelling protocol`

### [ ] T3.2 — Write the golden question set

- **Goal.** 60–100 labelled questions over the Syntec corpus. This is the
  most valuable and least automatable artifact in the project.
- **Concepts.** Why LLM-generated questions leak their answer into the
  question; class balance; held-out splits.
- **Files.** `eval/golden.jsonl` (new).
- **Subtasks.**
  1. Write ~30 `in_topic_answerable` questions of the kind a real user asks:
     notice periods, trial periods, salary coefficients, paid leave, overtime.
  2. Write ~15 `in_topic_unanswerable`: plausibly Syntec-shaped, genuinely
     absent from the text.
  3. Write ~15 `off_topic`: clearly outside the corpus domain.
  4. Write ~15 `adversarial`: prompt injections phrased with HR/legal
     vocabulary, so embedding distance alone cannot catch them.
  5. Label relevant chunk ids for the answerable class.
  6. Reserve a held-out split (~25 %) that you do not look at while tuning.
- **Tests.** Schema validation; class-balance assertion; no duplicate ids.
- **Metrics.** count per class; held-out split size.
- **Acceptance.** The file validates and the class counts are recorded.
- **Interview questions.**
  - How did you decide a question was genuinely unanswerable?
  - Why hold out a split when you are not training anything *yet*?
  - How would you detect that you accidentally wrote leading questions?
- **Commit.** `feat: add the Syntec golden evaluation set`

### [ ] T3.3 — Implement retrieval metrics

- **Goal.** Compute recall@k and MRR from the golden set.
- **Concepts.** recall@k, precision@k, MRR, nDCG; what recall@k hides; why
  recall is the primary retrieval metric for RAG.
- **Files.** `eval/metrics.py` (new), `tests/test_eval_metrics.py` (new).
- **Subtasks.**
  1. Implement recall@k, precision@k, MRR from scratch.
  2. Unit-test each against hand-computed toy examples.
  3. Keep them independent of the retrieval implementation.
- **Tests.** Toy-example unit tests where you can verify the number by hand.
- **Acceptance.** Metrics are correct on examples you computed manually.
- **Interview questions.**
  - Define MRR. When is it a better choice than recall@k?
  - Your recall@10 is 0.95 and answer quality is poor. What do you check?
- **Commit.** `feat: implement retrieval evaluation metrics`

### [ ] T3.4 — Implement guardrail metrics

- **Goal.** Measure the behaviour the whole project is about.
- **Concepts.** False-refusal rate and false-acceptance rate; the
  precision/recall trade-off framed as a safety/usefulness trade-off; why
  accuracy is meaningless under class imbalance.
- **Files.** `eval/metrics.py`, `tests/test_eval_metrics.py`.
- **Subtasks.**
  1. Implement false-refusal rate (in-topic wrongly refused).
  2. Implement false-acceptance rate (off-topic/adversarial wrongly accepted).
  3. Produce a confusion matrix over the four classes.
- **Tests.** Unit tests on synthetic predictions.
- **Acceptance.** Both rates computable from a prediction dump.
- **Interview questions.**
  - A guardrail that refuses everything has a false-acceptance rate of zero.
    Why is that not a good system, and which metric exposes it?
  - Which of the two errors is worse for a legal-document assistant, and why?
- **Commit.** `feat: implement guardrail false-refusal and false-acceptance metrics`

### [ ] T3.5 — Implement generation metrics

- **Goal.** Measure whether answers are supported by retrieved context.
- **Concepts.** RAGAS vocabulary — faithfulness, context precision, context
  recall, answer relevancy; LLM-as-judge as an instrument with its own error
  rate.
- **Files.** `eval/metrics.py`, `eval/judge.py` (new).
- **Subtasks.**
  1. Implement a cheap non-LLM groundedness proxy first: per-sentence
     lexical/semantic overlap with retrieved context.
  2. Add an optional LLM judge using `llama3.2:3b` (fits in 8 GB VRAM).
  3. Hand-label ~20 examples and **measure the judge's agreement with you**.
     Record that number.
  4. Use the RAGAS metric names even though you implemented them yourself.
- **Tests.** Proxy metric unit tests; judge tested against the hand-labelled
  set.
- **Metrics.** judge agreement rate with human labels; proxy vs. judge
  correlation.
- **Acceptance.** You can state your judge's error rate as a number.
- **Interview questions.**
  - You use an LLM to grade an LLM. Why is that not circular?
  - What is your judge's error rate, and how did you measure it?
  - Why implement RAGAS metrics yourself instead of importing the library?
- **Commit.** `feat: implement groundedness metrics and a measured LLM judge`

### [ ] T3.6 — Build the evaluation runner

- **Goal.** One command that produces the full metric table.
- **Concepts.** Reproducible experiment runs; recording configuration
  alongside results; latency percentiles (p50/p95) and why not the mean.
- **Files.** `eval/run.py` (new), `eval/results/`, `eval/RESULTS.md`.
- **Subtasks.**
  1. Run every golden question through the pipeline; dump raw predictions.
  2. Compute all metrics per class.
  3. Record per-stage latency p50/p95.
  4. Write `eval/results/<date>_<label>.json` including corpus hash, model
     names, config, and git SHA.
  5. Append a summary row to `eval/RESULTS.md`.
- **Tests.** A smoke test on `docs_smoke/` with a handful of questions.
- **Metrics.** the full table.
- **Acceptance.** `uv run python eval/run.py --label baseline` produces a
  committed result file.
- **Interview questions.**
  - Why record the git SHA and corpus hash inside the result file?
  - Why p95 latency rather than mean?
- **Commit.** `feat: add the evaluation runner`

### [ ] T3.7 — Build the failure taxonomy

- **Goal.** The single most portfolio-differentiating artifact: classify
  *why* each failure happened, not just how many there were.
- **Concepts.** Retrieval failure vs. generation failure vs. guardrail
  failure as three distinct problems with three distinct fixes. Industry
  analysis attributes ~73 % of RAG failures to retrieval — verify that
  proportion on *your* system rather than repeating it.
- **Files.** `eval/failures.py` (new), `eval/RESULTS.md`.
- **Subtasks.**
  1. For each failed question, classify: relevant chunk never retrieved
     (retrieval) · retrieved but answer wrong (generation) · wrongly
     refused or wrongly accepted (guardrail).
  2. Produce a breakdown table and the ten worst cases with their scores.
  3. Write `design/evaluation.md`: what you measured, what you found, what
     surprised you.
- **Tests.** Classification logic unit-tested on synthetic cases.
- **Metrics.** failure counts by category; your own retrieval-failure share.
- **Acceptance.** You can say "N % of my failures are retrieval failures" and
  point at the command that computed it.
- **Interview questions.**
  - What proportion of your failures are retrieval vs. generation, and how
    did that change what you built next?
  - Give a concrete example of a case retrieved correctly but answered wrongly.
- **Commit.** `feat: add failure taxonomy analysis`

---

# Phase 4 — Measure the current baseline

### [ ] T4.1 — Index the Syntec corpus with the current stack

- **Goal.** The unmodified pipeline (PyPDF + `RecursiveCharacterTextSplitter`
  + qwen3-embedding:8b + Chroma) running on the real corpus. This is the
  "before" every later comparison needs.
- **Concepts.** Why the baseline must use the *old* stack; ingestion cost.
- **Files.** none changed — configuration only.
- **Subtasks.**
  1. Point `DOCS_DIR` at the Syntec corpus.
  2. Run ingestion; record wall-clock time and final chunk count.
  3. Re-run `scripts/calibrate_threshold.py`; **expect the threshold to move**
     — a denser corpus compresses distances.
  4. Record the new recommended threshold and why it moved.
- **Tests.** Suite stays green (it runs on `docs_smoke/`).
- **Metrics.** chunk count; ingestion duration; peak VRAM; old vs. new
  threshold; both distributions.
- **Acceptance.** The corpus is indexed and the threshold is recalibrated,
  with both numbers recorded.
- **Interview questions.**
  - Your threshold moved when the corpus grew. Explain the mechanism.
  - Why not keep the old threshold and save the effort?
- **Commit.** `chore: index the Syntec corpus and recalibrate the threshold`

### [ ] T4.2 — Record the baseline evaluation

- **Goal.** The reference row every later change is judged against.
- **Concepts.** Controlled comparison; changing one variable at a time.
- **Files.** `eval/results/`, `eval/RESULTS.md`, `design/evaluation.md`.
- **Subtasks.**
  1. Run `eval/run.py --label baseline`.
  2. Record every metric and the failure taxonomy.
  3. Write the baseline section of `design/evaluation.md`, including what
     you expected versus what you got.
- **Tests.** —
- **Metrics.** recall@k, MRR, false-refusal, false-acceptance, faithfulness,
  latency p50/p95, failure breakdown.
- **Acceptance.** A committed baseline row in `eval/RESULTS.md`.
- **Interview questions.**
  - What was your baseline recall@5, and what was the dominant failure mode?
  - Which single change did those numbers tell you to make first?
- **Commit.** `docs: record baseline evaluation results`

---

# Phase 5 — PyPDF versus Docling

### [ ] T5.1 — Quantify the parsing problem

- **Goal.** Prove the parser is losing content before replacing it.
- **Concepts.** PDF text extraction; reading order; why layout-unaware
  extraction destroys tables — which matters enormously for salary grids.
- **Files.** `scripts/compare_parsers.py` (new).
- **Subtasks.**
  1. Reproduce the known symptom: the existing 44 KB PDF yields only 2 chunks.
  2. On Syntec PDFs, measure extracted characters, table count, and reading
     order errors with PyPDF.
  3. Manually inspect one salary grid and record what is lost.
- **Tests.** —
- **Metrics.** characters extracted; chunks produced; tables preserved.
- **Acceptance.** The loss is a number, not an impression.
- **Interview questions.**
  - How did you know your parser was the problem and not your chunker?
- **Commit.** `feat: add a parser comparison script`

### [ ] T5.2 — Integrate Docling

- **Goal.** Replace PyPDF with layout-aware parsing.
- **Concepts.** Layout detection; structure-preserving extraction; Markdown
  as an intermediate representation that keeps tables.
- **Files.** `app/services/ingestion.py`, `pyproject.toml`.
- **Subtasks.**
  1. Add `docling`; note the first run downloads models.
  2. Add a Docling loader beside the existing ones; keep `.md`/`.txt` paths
     unchanged.
  3. Preserve existing metadata keys (`source`, `page`) so nothing
     downstream breaks.
  4. Keep the parser selectable by config, so the comparison stays runnable.
- **Tests.** Ingestion tests on a small PDF fixture; existing tests green.
- **Metrics.** same as T5.1, side by side.
- **Acceptance.** Docling extracts measurably more, with tables intact.
- **Interview questions.**
  - Why Docling rather than LlamaParse or Unstructured?
  - What does Docling cost you — in dependencies, latency, and first-run time?
- **Commit.** `feat: replace PyPDF with Docling for layout-aware parsing`

### [ ] T5.3 — Measure the parsing change end to end

- **Goal.** Show whether better parsing improves *answers*, not just
  character counts.
- **Concepts.** End-to-end versus component metrics; why improving a component need not move the end metric, and what that tells you about your bottleneck.
- **Files.** `eval/RESULTS.md`, `design/ingestion.md` (new).
- **Subtasks.**
  1. Re-index; re-run the evaluation with `--label docling`.
  2. Diff against baseline.
  3. Write `design/ingestion.md` in the decision-log format.
- **Metrics.** recall@k and faithfulness delta; ingestion time cost.
- **Acceptance.** A measured delta, positive or negative, is recorded.
- **Interview questions.**
  - Better extraction did not necessarily improve recall. Why might that be?
- **Commit.** `docs: record the Docling versus PyPDF evaluation`

---

# Phase 6 — Structured legal chunking

### [ ] T6.1 — Design the chunking strategy

- **Goal.** Chunk along the document's own structure — articles, sections,
  annexes — instead of every 500 characters.
- **Concepts.** Fixed-size vs. structure-aware splitting; parent-child /
  sentence-window chunking (ranked first for retrieval precision in the
  ARAGOG study); why retrieval wants small chunks and generation wants large
  ones; how chunk size shifts the distance distribution.
- **Files.** `design/chunking.md` (new).
- **Subtasks.**
  1. Inspect the real structure of the Syntec text; identify article
     boundaries.
  2. Decide the parent/child granularity: retrieve on child, generate on
     parent.
  3. Decide what happens to salary tables — a split table is a wrong answer.
- **Tests.** —
- **Acceptance.** A written strategy with a worked example on a real article.
- **Interview questions.**
  - Why is splitting a legal article mid-sentence worse than splitting prose?
  - What is the retrieval/generation granularity trade-off?
- **Commit.** `docs: design structure-aware legal chunking`

### [ ] T6.2 — Implement structured chunking

- **Goal.** Turn the chunking strategy into code without ever splitting a table or an article.
- **Concepts.** Structural parsing of legal text versus regex heuristics; parent-child metadata linkage; table atomicity as a correctness constraint.
- **Files.** `app/services/ingestion.py`, `tests/test_chunking.py` (new).
- **Subtasks.**
  1. Implement article-boundary detection for the Syntec format.
  2. Implement parent-child linkage in metadata (`parent_id`, `article`).
  3. Keep tables whole.
  4. Keep the old splitter available behind config for comparison.
- **Tests.** Chunk boundaries on a real article fixture; table integrity; no
  empty chunks.
- **Metrics.** chunk count; size distribution; tables split (target: zero).
- **Acceptance.** No table is split; article boundaries respected.
- **Interview questions.**
  - How do you keep a chunk's article reference through to the citation?
- **Commit.** `feat: implement structure-aware legal chunking`

### [ ] T6.3 — Measure the chunking change

- **Goal.** Determine whether structure-aware chunking actually improved retrieval, and at what cost.
- **Concepts.** Confounded comparisons; why recalibration is part of this change rather than a separate step.
- **Files.** `eval/RESULTS.md`, `design/chunking.md`.
- **Subtasks.**
  1. Re-index the corpus with structured chunking enabled.
  2. Run `eval/run.py --label chunking`.
  3. Recalibrate the threshold — chunk size changed, so the distance distribution changed.
  4. Diff every metric against the `docling` row and write the conclusion into `design/chunking.md`.
- **Metrics.** recall@k, MRR, faithfulness delta vs. `docling` run.
- **Acceptance.** Recorded delta; threshold recalibrated (chunk size changed,
  so the distance distribution changed).
- **Interview questions.**
  - You changed chunk size and recall improved but the false-acceptance rate
    rose. Explain the mechanism.
- **Commit.** `docs: record the structured chunking evaluation`

---

# Phase 7 — BGE-M3 embeddings

### [ ] T7.1 — Understand and validate the model choice

- **Goal.** Be able to defend the swap, including its costs.
- **Concepts.** Dense vs. sparse vs. multi-vector representations; why BGE-M3
  emits all three from one model; 8k context; multilingual coverage and why
  it matters for a French corpus; MIT licence.
- **Files.** `design/embeddings.md` (new).
- **Subtasks.**
  1. Record the arithmetic: qwen3-embedding:8b at 4.7 GB plus gemma4 at
     9.6 GB exceeds 8 GB of VRAM; BGE-M3 plus llama3.2:3b does not.
  2. Record the alternative you rejected (Qwen3-Embedding-0.6B, dense only,
     would need a separate BM25 index) and why.
  3. Note that the top French MTEB model is an 8B — rejected for the same
     VRAM reason.
- **Tests.** —
- **Acceptance.** A written, numbers-backed justification.
- **Interview questions.**
  - Why BGE-M3 rather than the highest-scoring French model on MTEB?
  - What is a sparse embedding, and how does it differ from BM25?
- **Commit.** `docs: justify the BGE-M3 embedding choice`

### [ ] T7.2 — Integrate BGE-M3 (dense first)

- **Goal.** Add BGE-M3 as an embedding provider without breaking the existing provider paths.
- **Concepts.** Provider abstraction; embedding dimension; vector normalisation and how to verify it rather than assume it.
- **Files.** `app/services/ingestion.py`, `app/config.py`, `pyproject.toml`,
  `.env.example`.
- **Subtasks.**
  1. Add the provider; keep `openai` and `ollama` paths working.
  2. Wire dense embeddings only for now; sparse arrives in Phase 9.
  3. Verify vector dimension and normalisation — `scripts/inspect_metric.py`
     already does exactly this; run it.
- **Tests.** Provider-selection unit tests; existing suite green.
- **Metrics.** dimension; norm; ingestion time; peak VRAM.
- **Acceptance.** `inspect_metric.py` reports the new geometry correctly.
- **Interview questions.**
  - Are BGE-M3 vectors normalised? How did you check, and why does it matter
    for your threshold?
- **Commit.** `feat: add BGE-M3 as an embedding provider`

### [ ] T7.3 — Recalibrate and re-measure

- **Goal.** A new embedding model means a new vector space and therefore a
  new threshold. This is mandatory, not optional.
- **Concepts.** Why vector spaces are not comparable across models; a threshold as a model-specific artifact, not a constant.
- **Files.** `design/guardrail.md`, `design/embeddings.md`, `eval/RESULTS.md`.
- **Subtasks.**
  1. Re-run `scripts/calibrate_threshold.py`.
  2. Re-run the evaluation with `--label bge-m3`.
  3. Record old and new thresholds side by side with both distributions.
- **Metrics.** threshold before/after; full metric table; ingestion time and
  VRAM before/after.
- **Acceptance.** Recorded delta and a recalibrated threshold.
- **Interview questions.**
  - Why can a threshold not transfer between embedding models? Be precise.
- **Commit.** `docs: recalibrate the threshold for BGE-M3 and record results`

---

# Phase 8 — Migrate to Qdrant

### [ ] T8.1 — Justify the migration honestly

- **Goal.** State the real reason. At a few thousand vectors, Chroma is
  performant enough; the reason is **native sparse-vector support**, which
  the hybrid retrieval in Phase 9 requires and Chroma does not offer.
- **Concepts.** Named vectors; HNSW parameters; payload filtering;
  pre-filter vs. post-filter correctness.
- **Files.** `design/vectorstore.md` (new).
- **Subtasks.**
  1. Write the honest argument, including "performance is not the reason at
     my scale".
  2. Record the rejected alternative: Chroma plus a separate `rank_bm25`
     index, and its synchronisation cost.
- **Acceptance.** A defence that survives a sceptical reviewer.
- **Interview questions.**
  - You migrated to Qdrant. Was it for performance? (Correct answer: no —
    explain what it *was* for.)
  - What is the difference between filtering before and after the ANN search?
- **Commit.** `docs: justify the migration from Chroma to Qdrant`

### [ ] T8.2 — Run Qdrant and define the collection

- **Goal.** Get Qdrant running beside the existing services, and define the collection deliberately rather than by library default.
- **Concepts.** HNSW parameters (`m`, `ef_construct`, `ef`); named vectors; payload indexing; explicit distance-metric selection.
- **Files.** `docker-compose.yml`, `app/services/vectorstore.py` (new),
  `.env.example`.
- **Subtasks.**
  1. Add the Qdrant service to Compose with a persistent volume and a
     healthcheck, matching the existing Ollama service's shape.
  2. Define the collection: named dense vector, distance metric, HNSW
     parameters, payload schema.
  3. **Choose the distance metric explicitly and write down why** — Chroma's
     squared-L2 default is the source of this project's most instructive bug.
- **Tests.** Integration test behind a marker, skipped when Qdrant is absent.
- **Acceptance.** Collection created with an explicitly chosen metric.
- **Interview questions.**
  - Which distance metric did you configure, and why that one?
  - Your threshold was calibrated on squared L2. What happens if Qdrant is
    configured for cosine?
- **Commit.** `feat: add Qdrant service and collection definition`

### [ ] T8.3 — Port ingestion and retrieval

- **Goal.** Move storage to Qdrant while keeping behaviour identical, and
  take retrieval out of LangChain — this is the "LangChain for ingestion
  only" decision made concrete.
- **Concepts.** Idempotent upsert; batching; preserving an interface contract through a refactor; why the fake-store tests are your safety net.
- **Files.** `app/services/ingestion.py`, `app/services/retrieval.py`,
  `app/services/vectorstore.py`.
- **Subtasks.**
  1. Write upsert with batching and idempotency — keep the "clear before
     write" guarantee that `ingestion.py` documents today.
  2. Write search directly against the Qdrant client.
  3. Preserve the `retrieve()` return contract so `routes.py` and the
     guardrail tests are unaffected.
  4. Keep the guardrail comparison direction correct for the chosen metric.
- **Tests.** Existing guardrail tests must pass **unchanged** — they inject a
  fake store, so they are the proof your refactor preserved the contract.
- **Metrics.** ingestion time; query latency p50/p95 versus Chroma.
- **Acceptance.** Same evaluation numbers as the `bge-m3` run, within noise.
- **Interview questions.**
  - Your guardrail tests passed untouched through a database migration. Why
    is that evidence of good design?
  - Which parts of LangChain did you keep, and why those?
- **Commit.** `refactor: port ingestion and retrieval to Qdrant`

### [ ] T8.4 — Verify parity

- **Goal.** Confirm the migration preserved behaviour. A migration that silently changes quality is a bug, not an upgrade.
- **Concepts.** Parity testing; run-to-run noise; telling a real regression apart from variance.
- **Files.** `eval/RESULTS.md`, `design/vectorstore.md`.
- **Subtasks.**
  1. Run `eval/run.py --label qdrant`.
  2. Diff every metric against the `bge-m3` row.
  3. Investigate any difference larger than run-to-run noise, and explain it rather than accepting it.
  4. Record query latency p50/p95 for Chroma and Qdrant side by side.
- **Metrics.** full table vs. `bge-m3`; latency comparison.
- **Acceptance.** Quality unchanged (this is a migration, not an improvement);
  any difference explained.
- **Interview questions.**
  - Your recall changed slightly after migrating. Is that acceptable? How
    would you find out why?
- **Commit.** `docs: record Qdrant migration parity results`

---

# Phase 9 — Hybrid dense + sparse retrieval

### [ ] T9.1 — Understand sparse retrieval and fusion

- **Goal.** Build the mental model for sparse retrieval and rank fusion before writing any of it.
- **Concepts.** BM25 (term frequency, IDF, length normalisation, `k1`/`b`);
  learned sparse vectors versus classical BM25; **Reciprocal Rank Fusion**
  and why it merges rankings without requiring comparable score scales —
  which matters because one list is a **distance** (lower is better) and the
  other a **score** (higher is better). Inverting that is the same class of
  bug as the original Chroma-distance mistake.
- **Files.** `design/retrieval.md` (new).
- **Subtasks.**
  1. Hand-compute BM25 on three toy documents.
  2. Hand-compute RRF on two toy rankings.
  3. Write down which of your lists is a distance and which is a score.
- **Acceptance.** You can do both calculations on paper.
- **Interview questions.**
  - Why does RRF not need the two scores to be on the same scale?
  - Which of your two rankings is "lower is better"? What breaks if you get
    it backwards?
- **Commit.** `docs: design hybrid retrieval with RRF fusion`

### [ ] T9.2 — Index sparse vectors

- **Goal.** Store BGE-M3's sparse output beside the dense vector so both can be queried in one request.
- **Concepts.** Sparse vector representation (indices plus values); multiple named vectors in one collection; index-size trade-off.
- **Files.** `app/services/ingestion.py`, `app/services/vectorstore.py`.
- **Subtasks.**
  1. Enable BGE-M3 sparse output.
  2. Add a named sparse vector to the Qdrant collection.
  3. Upsert dense and sparse together.
- **Tests.** A test that a chunk carries both vectors.
- **Metrics.** ingestion time delta; index size delta.
- **Acceptance.** Both vectors queryable.
- **Interview questions.**
  - What does a BGE-M3 sparse vector contain, and how does it differ from a
    BM25 term-frequency vector?
- **Commit.** `feat: index BGE-M3 sparse vectors alongside dense`

### [ ] T9.3 — Implement hybrid search with RRF

- **Goal.** Merge the dense and sparse rankings into one result list, with all three modes independently selectable for evaluation.
- **Concepts.** The RRF formula and its `k` constant; rank versus score; mode switching as an ablation enabler.
- **Files.** `app/services/retrieval.py`, `app/config.py`,
  `tests/test_retrieval.py`.
- **Subtasks.**
  1. Add `RETRIEVAL_MODE=dense|sparse|hybrid`.
  2. Implement the fused query.
  3. Unit-test RRF against your hand-computed example.
- **Tests.** RRF correctness; all three modes return results; ranking
  direction correct in each.
- **Metrics.** latency per mode.
- **Acceptance.** All three modes independently evaluable.
- **Interview questions.**
  - Walk me through your RRF implementation. Where would an off-by-one in
    the rank hurt you?
- **Commit.** `feat: implement hybrid dense+sparse retrieval with RRF`

### [ ] T9.4 — Resolve the guardrail/fusion conflict

- **Goal.** **Do not skip this.** The guardrail thresholds a raw distance.
  RRF outputs a fused rank score with no distance semantics, so the existing
  threshold becomes meaningless the moment fusion is enabled.
- **Concepts.** Score-scale semantics; where in a pipeline a decision
  boundary belongs.
- **Files.** `app/services/retrieval.py`, `design/guardrail.md`.
- **Subtasks.**
  1. Decide explicitly: threshold the dense distance *before* fusion, or
     recalibrate on the fused score.
  2. Implement it; document the choice and the rejected option.
  3. Recalibrate whichever signal you chose.
- **Tests.** Guardrail tests extended to hybrid mode.
- **Metrics.** false-refusal and false-acceptance in hybrid mode.
- **Acceptance.** The guardrail behaves correctly in all three modes.
- **Interview questions.**
  - This is the sharpest question your project generates: after adding RRF,
    what exactly is your threshold comparing, and is that number meaningful?
- **Commit.** `fix: make the guardrail well-defined under hybrid retrieval`

### [ ] T9.5 — Measure hybrid retrieval

- **Goal.** Establish whether hybrid retrieval beats dense on this corpus, and on which question types.
- **Concepts.** Per-class analysis; lexical versus semantic query types and which retriever wins each.
- **Files.** `eval/RESULTS.md`, `design/retrieval.md`.
- **Subtasks.**
  1. Run the harness three times: `--label dense`, `--label sparse`, `--label hybrid`.
  2. Build the three-row comparison table in `eval/RESULTS.md`.
  3. Inspect by hand which question types each mode wins, and quote a real example of each.
  4. Write `design/retrieval.md` in the decision-log format, including the fusion/guardrail decision from T9.4.
- **Metrics.** dense vs. sparse vs. hybrid: recall@k, MRR, latency p95,
  guardrail rates.
- **Acceptance.** Three comparable rows. If hybrid does not win, record that
  honestly — a measured negative result beats an assumed positive one.
- **Interview questions.**
  - On which question types did sparse retrieval beat dense? Give a real
    example from your corpus.
- **Commit.** `docs: record the hybrid retrieval evaluation`

---

# Phase 10 — Abstention mechanism (the ML component)

*This is the phase that makes the project read as ML engineering rather than
API orchestration. Today `app/` contains no modelling at all.*

### [ ] T10.1 — Design the abstention formulation

- **Goal.** Frame "should I answer?" as a supervised learning problem.
- **Concepts.** Binary classification under class imbalance; feature
  engineering from retrieval signals; why accuracy is the wrong metric;
  precision-recall curves; probability calibration; choosing an operating
  point deliberately. Context: this is an active research direction —
  systems that judge whether retrieved evidence *supports, refutes, or is
  insufficient*, enabling abstention.
- **Files.** `design/abstention.md` (new).
- **Subtasks.**
  1. Define the label: from the golden set classes, which examples should be
     answered and which refused.
  2. Define the feature set from signals you already compute: best distance,
     mean and standard deviation of top-k distances, gap between rank 1 and
     rank k, number of results above threshold, sparse/dense rank agreement,
     query length.
  3. Define the baseline to beat: **the current single-threshold guardrail**.
- **Acceptance.** A written problem statement with features and baseline.
- **Interview questions.**
  - Why is accuracy the wrong metric here?
  - Why is score *dispersion* across top-k informative, beyond the best score?
- **Commit.** `docs: formulate abstention as a supervised learning problem`

### [ ] T10.2 — Build the feature extraction pipeline

- **Goal.** Turn retrieval signals into a clean, leakage-free dataset the classifier can train on.
- **Concepts.** Feature engineering; data leakage; train/test split hygiene; class balancing on small data.
- **Files.** `app/services/features.py` (new), `eval/build_dataset.py` (new).
- **Subtasks.**
  1. Extract features for every golden question.
  2. Generate additional negatives if the class balance is too skewed.
  3. Write `eval/abstention_dataset.csv`; keep the held-out split untouched.
- **Tests.** Feature extraction unit tests; no NaNs; no leakage of the label
  into a feature.
- **Metrics.** dataset size; class balance; feature statistics.
- **Acceptance.** A clean dataset with a documented train/test split.
- **Interview questions.**
  - How did you make sure no feature leaks the answer?
  - Your dataset has ~100 rows. What does that limit?
- **Commit.** `feat: build the abstention feature dataset`

### [ ] T10.3 — Train and evaluate the classifier

- **Goal.** Beat — or fail to beat, honestly — the threshold baseline.
- **Concepts.** Logistic regression and gradient boosting; regularisation on
  small data; cross-validation; feature importance; why an interpretable
  model is worth more here than a neural network.
- **Files.** `app/services/abstention.py` (new), `scripts/train_abstention.py`
  (new), `models/` (git-ignored except a manifest).
- **Subtasks.**
  1. Train logistic regression first; it is the explainable option.
  2. Try gradient boosting; compare with cross-validation.
  3. Report precision, recall, F1, ROC-AUC, PR-AUC **against the threshold
     baseline** on the held-out split.
  4. Plot the precision-recall curve; choose an operating point and justify
     it in false-refusal vs. false-acceptance terms.
  5. Record feature importances and what they mean.
- **Tests.** Deterministic training with a fixed seed; a test that a saved
  model loads and predicts.
- **Metrics.** precision, recall, F1, ROC-AUC, PR-AUC, chosen operating
  point, feature importances — all versus the baseline.
- **Acceptance.** A held-out comparison exists. **If the classifier does not
  beat the threshold, that is a valid, reportable result** — it means the
  single distance signal was already near-optimal on this corpus.
- **Interview questions.**
  - Which feature mattered most, and does that match your intuition?
  - How did you pick your operating point? Why not maximise F1?
  - With ~100 examples, how do you know you are not overfitting?
  - Your classifier beat the threshold by 4 points. Is that significant?
- **Commit.** `feat: train and evaluate the abstention classifier`

### [ ] T10.4 — Wire abstention into the pipeline

- **Goal.** Put the trained classifier on the request path without losing the ability to fall back to the threshold.
- **Concepts.** Model loading and versioning; graceful degradation; feature flags for ML components.
- **Files.** `app/services/retrieval.py`, `app/schemas.py`, `app/config.py`.
- **Subtasks.**
  1. Add `GUARDRAIL_MODE=threshold|classifier`, defaulting to the safer one.
  2. Add a `refusal_reason` field to `QueryAnswer` so the API states *why*.
  3. Log the decision, its reason, and the confidence.
- **Tests.** Both modes tested; the classifier path falls back gracefully if
  the model file is missing.
- **Metrics.** end-to-end false-refusal and false-acceptance in both modes.
- **Acceptance.** Both modes work; the API explains its refusals.
- **Interview questions.**
  - Why keep the threshold mode at all once the classifier works?
  - What happens if the model file is missing in production?
- **Commit.** `feat: integrate the abstention classifier behind a config switch`

---

# Phase 11 — Cross-encoder reranking

### [ ] T11.1 — Understand reranking

- **Goal.** Understand why reranking works and what it costs before adding a second model to the pipeline.
- **Concepts.** Bi-encoder versus cross-encoder: independent embedding
  (fast, cacheable, blind to interaction) versus joint pair scoring
  (accurate, one model call per candidate, uncacheable) — and why that
  arithmetic forces the retrieve-wide-then-rerank shape.
- **Files.** `design/reranking.md` (new).
- **Subtasks.**
  1. Record the model choice: Qwen3-Reranker-0.6B, Apache 2.0, 100+
     languages — reported 65.80 on MTEB-R against 57.03 for BGE-reranker-v2-m3.
  2. Record the counter-evidence: the ARAGOG study found a commercial
     reranker gave no advantage. You must measure on your own corpus.
  3. Decide the candidate width (k=20–50) and the final cut (3–5).
- **Acceptance.** A written justification including the counter-evidence.
- **Interview questions.**
  - Why not run the cross-encoder over the whole corpus?
  - Why does a cross-encoder score better than a bi-encoder, mechanically?
- **Commit.** `docs: design cross-encoder reranking`

### [ ] T11.2 — Implement reranking

- **Goal.** Add a toggleable reranking stage that reorders candidates without changing the pipeline's shape.
- **Concepts.** Model loading at startup versus per request; CPU inference; pair batching; latency budgets.
- **Files.** `app/services/reranking.py` (new), `app/services/retrieval.py`,
  `app/config.py`, `pyproject.toml`.
- **Subtasks.**
  1. Add `sentence-transformers`; load the reranker on CPU to avoid VRAM
     contention with Ollama.
  2. Implement `rerank(question, candidates, top_n)`.
  3. Add `RERANK_ENABLED` and `RERANK_CANDIDATES`.
  4. Load the model once at startup, not per request.
- **Tests.** Reranking reorders a known-bad ordering; disabled mode is a
  no-op; output shape unchanged.
- **Metrics.** added latency p50/p95; model load time; memory.
- **Acceptance.** Toggleable; adds latency you have measured.
- **Interview questions.**
  - Why load the model at startup? What breaks if you load it per request?
  - Why CPU and not GPU here?
- **Commit.** `feat: add cross-encoder reranking`

### [ ] T11.3 — Handle the guardrail score change, again

- **Goal.** The reranker emits its own score scale. Same trap as T9.4.
- **Concepts.** Composing multiple score scales in one pipeline; adding a feature to an existing model and retraining honestly.
- **Files.** `app/services/retrieval.py`, `app/services/abstention.py`,
  `design/guardrail.md`.
- **Subtasks.**
  1. Decide which signal the guardrail now uses.
  2. Add the reranker score as a **feature** to the abstention classifier and
     retrain — this is the natural place for it.
  3. Re-measure.
- **Tests.** Guardrail tests with reranking on and off.
- **Metrics.** classifier metrics with and without the reranker feature.
- **Acceptance.** The decision is explicit and documented.
- **Interview questions.**
  - You now have three score scales in one pipeline. How do you keep that
    coherent?
- **Commit.** `feat: add reranker score to the abstention classifier`

### [ ] T11.4 — Measure reranking

- **Goal.** Decide, on evidence, whether the reranker earns the latency it adds.
- **Concepts.** The quality/latency trade-off; deciding your rejection rule *before* seeing the results.
- **Files.** `eval/RESULTS.md`, `design/reranking.md`.
- **Subtasks.**
  1. Run the harness with reranking off, then on, changing nothing else.
  2. Record the quality delta and the p95 latency delta together.
  3. Decide keep or reject, and state in advance the rule you used to decide.
  4. Write the decision and its justification into `design/reranking.md`.
- **Metrics.** recall@5, MRR, faithfulness, **added p95 latency**.
- **Acceptance.** Quality gain and latency cost both reported. A reranker
  adding 400 ms for +0.01 MRR is a defensible **rejection**, and saying so is
  a stronger answer than shipping it.
- **Interview questions.**
  - Was the reranker worth it? Justify with your numbers, including latency.
- **Commit.** `docs: record the reranking evaluation`

---

# Phase 12 — Comparative evaluation (the ablation table)

### [ ] T12.1 — Produce the full ablation

- **Goal.** One table measuring every component's contribution on *your*
  corpus. This is the recruiter-facing centrepiece; the code was only how you
  got here.
- **Concepts.** Ablation study design; changing one variable at a time;
  cumulative versus isolated contribution.
- **Files.** `eval/ablation.py` (new), `eval/RESULTS.md`.
- **Subtasks.**
  1. Run every configuration through the harness:

     ```
     baseline (PyPDF + fixed chunks + qwen3-8b + Chroma + dense)
     + Docling
     + structured chunking
     + BGE-M3
     + Qdrant
     + hybrid (RRF)
     + reranking
     + abstention classifier
     ```
  2. Report recall@k, MRR, false-refusal, false-acceptance, faithfulness,
     p95 latency for each row.
  3. Report the failure taxonomy shift across rows.
- **Metrics.** the complete table.
- **Acceptance.** A single committed table a reader can scan in 30 seconds.
- **Interview questions.**
  - Which change gave the largest gain per unit of complexity added?
  - Did any change make something worse? What did you do about it?
- **Commit.** `docs: add the full retrieval ablation study`

### [ ] T12.2 — Add an evaluation regression test to CI

- **Goal.** Prevent a future change from silently degrading the guardrail.
- **Concepts.** Regression budgets; keeping slow tests off the default CI path; pytest markers.
- **Files.** `tests/test_eval_regression.py` (new), `.github/workflows/ci.yml`.
- **Subtasks.**
  1. Mark it `@pytest.mark.eval`, excluded from the default CI run.
  2. Fail if false-acceptance rate regresses past a fixed budget.
  3. Document how to run it locally.
- **Tests.** The regression test itself.
- **Acceptance.** CI stays fast; the guard exists and is runnable.
- **Interview questions.**
  - Why exclude the eval test from the default CI run?
  - Which metric did you choose to guard, and why that one?
- **Commit.** `test: add an evaluation regression guard`

---

# Phase 13 — Citations and legal-text versioning

### [ ] T13.1 — Implement chunk-level citations

- **Goal.** For legal answers, "according to the Syntec agreement" is
  useless. "Article 15, salary grid, version of <date>" is the product.
- **Concepts.** Attribution; span-level versus document-level citation;
  citation validity as a checkable property.
- **Files.** `app/services/generation.py`, `app/schemas.py`.
- **Subtasks.**
  1. Extend the response with structured citations: article, section, page,
     version date, chunk id.
  2. Note that ingestion already stores page metadata and the API currently
     drops it — stop dropping it.
  3. Reject any answer citing a chunk that was never retrieved.
- **Tests.** Citation presence; invalid-citation rejection; determinism.
- **Metrics.** citation validity rate.
- **Acceptance.** Every answer carries verifiable citations.
- **Interview questions.**
  - How do you prevent the model from inventing a citation?
  - Why is a document-level source insufficient here?
- **Commit.** `feat: add verifiable chunk-level citations`

### [ ] T13.2 — Handle text versions

- **Goal.** Collective agreements are amended. An answer from a superseded
  version is wrong even if it was right last year.
- **Concepts.** Temporal validity; effective dates; versioned corpora.
- **Files.** `app/services/ingestion.py`, `app/services/retrieval.py`,
  `design/versioning.md` (new).
- **Subtasks.**
  1. Store the effective date of each text in the payload.
  2. Return the version date with every citation.
  3. Decide the policy for superseded text: excluded at ingestion, or
     filtered at query time. Document the choice.
- **Tests.** Version metadata preserved; filtering works if implemented.
- **Metrics.** documents by version date.
- **Acceptance.** Every answer states which version it is based on.
- **Interview questions.**
  - The agreement is amended tomorrow. What is your update path?
  - A user asks about a rule as it stood in 2023. Can your system answer that?
- **Commit.** `feat: track and surface legal text versions`

### [ ] T13.3 — Output-side groundedness check

- **Goal.** Close the layered-guardrail gap `design/guardrail.md` already
  admits: one distance threshold cannot separate wrong-domain from
  adversarial from in-domain-but-unanswered.
- **Concepts.** Groundedness verification; input-side versus output-side
  defence; why the generation prompt's own instruction is structurally weak
  because it depends on model compliance.
- **Files.** `app/services/generation.py`, `app/services/guardrail.py` (new).
- **Subtasks.**
  1. Add input-side injection detection before retrieval.
  2. Add an output-side groundedness check; downgrade unsupported answers to
     an explicit "I don't know".
  3. Extend `refusal_reason` to cover every layer.
- **Tests.** Adversarial golden questions blocked; legitimate answers
  unaffected.
- **Metrics.** adversarial block rate; false-refusal cost of each layer.
- **Acceptance.** Each layer measured separately, including its cost.
- **Interview questions.**
  - Your embedding guardrail scored injections at 0.64–0.83, under a 0.9
    threshold. Why did distance fail to catch them?
  - Each layer adds false refusals. How did you decide it was worth it?
- **Commit.** `feat: add layered input and output guardrails`

---

# Phase 14 — API and demonstration interface

### [ ] T14.1 — Finalise the API contract

- **Goal.** Make the API self-explanatory, and honest about what it knows and why it refused.
- **Concepts.** API contract design; readiness versus liveness probes; expressing refusal inside a response schema rather than as an error.
- **Files.** `app/schemas.py`, `app/api/routes.py`.
- **Subtasks.**
  1. Finalise the response: answer, structured citations, `refusal_reason`,
     confidence, retrieval mode, latency breakdown.
  2. Ensure the OpenAPI docs at `/docs` are self-explanatory.
  3. Add `/health` readiness that reflects real dependencies (Qdrant,
     Ollama), not a hardcoded "ok".
- **Tests.** Schema tests; health reflects a downed dependency.
- **Acceptance.** `/docs` is readable by someone who has never seen the repo.
- **Interview questions.**
  - Why does your health check need to know about Qdrant?
  - Why is a refusal a 200 with a reason rather than an error?
- **Commit.** `feat: finalise the query API contract`

### [ ] T14.2 — Build the demo interface

- **Goal.** Something a client can watch work and a recruiter can see without
  cloning anything. Deliberately ~100 lines — this must not become a frontend
  project.
- **Concepts.** Demo-driven communication; exposing model internals to build trust; configuration over hardcoding.
- **Files.** `demo/app.py` (new), `pyproject.toml`.
- **Subtasks.**
  1. Gradio single page: question box, answer, **citations displayed
     prominently**.
  2. Show refusals as a **first-class outcome with the reason**, never as an
     error.
  3. Add an inspector panel: retrieved chunks with their scores, and the
     guardrail decision.
  4. Add example questions covering all four classes, so a visitor can
     trigger a refusal in one click.
  5. No hardcoded local paths — config through environment, so later
     deployment is an afternoon rather than a rewrite.
- **Tests.** A smoke test that the interface builds.
- **Acceptance.** A visitor can trigger a correct answer *and* a correct
  refusal without typing anything.
- **Interview questions.**
  - Why show the retrieved chunks and scores to the user?
  - Why is displaying a refusal well a product decision, not a UI detail?
- **Commit.** `feat: add a Gradio demonstration interface`

---

# Phase 15 — Tests, Docker and observability

### [ ] T15.1 — Consolidate the test suite

- **Goal.** Make the suite fast, hermetic, and honest about what it does not cover.
- **Concepts.** The test pyramid; hermetic tests; markers; coverage as a signal rather than a target.
- **Files.** `tests/`, `pyproject.toml`.
- **Subtasks.**
  1. Add markers: `unit`, `integration`, `eval`.
  2. Ensure unit tests still need no network, no API key, no `.env` — the
     property `tests/conftest.py` already guarantees today.
  3. Measure and record coverage.
- **Tests.** The whole suite.
- **Metrics.** test count; coverage; runtime.
- **Acceptance.** `uv run pytest -m unit` is fast and hermetic.
- **Interview questions.**
  - Why must unit tests run without a network?
  - What is your coverage, and which uncovered part worries you most?
- **Commit.** `test: consolidate and mark the test suite`

### [ ] T15.2 — Update Docker Compose

- **Goal.** One command must bring the whole system up from a clean clone.
- **Concepts.** Healthchecks and dependency ordering; volumes and persistence; Docker layer caching.
- **Files.** `docker-compose.yml`, `Dockerfile`, `.env.example`.
- **Subtasks.**
  1. Services: app, Qdrant, Ollama, demo — each with healthchecks and
     dependency ordering, following the existing Ollama pattern.
  2. Persistent volumes for Qdrant and models.
  3. Verify a cold `docker compose up --build` works from a clean clone.
- **Tests.** Manual cold-start verification; record the timing.
- **Metrics.** cold start time; image size.
- **Acceptance.** One command brings the whole system up.
- **Interview questions.**
  - Why does the app wait for Qdrant's healthcheck rather than just retrying?
- **Commit.** `chore: extend Docker Compose with Qdrant and the demo`

### [ ] T15.3 — Request-level observability

- **Goal.** Make a single request traceable end to end by someone who did not write the code.
- **Concepts.** Correlation IDs; per-stage timing; metric label cardinality traps.
- **Files.** `app/logger.py`, `app/api/routes.py`,
  `design/observability.md` (new).
- **Subtasks.**
  1. Generate a `request_id` per request; propagate it through retrieval →
     guardrail → generation (the field was reserved in T0.6).
  2. Log per stage: latency, chunk count, best score, guardrail decision and
     reason, reranker score, token counts.
  3. Add `/stats` or a Prometheus `/metrics` endpoint.
  4. Paste one real annotated request trace into `design/observability.md`.
- **Tests.** A test that `request_id` appears in every log line of a request.
- **Metrics.** per-stage latency breakdown; estimated cost per query.
- **Acceptance.** One `curl` produces a complete, readable trace.
- **Interview questions.**
  - Walk me through the trace of one query, stage by stage.
  - What would you add to debug a slow query in production?
- **Commit.** `feat: add request-level observability and a metrics endpoint`

---

# Phase 16 — Langfuse (optional, only if time allows)

*Explicitly last and explicitly optional. Structured JSON logs plus a metrics
endpoint already cover the real need. Langfuse is adopted here for
recognisability and for a README screenshot — that is a legitimate reason,
but a secondary one.*

### [ ] T16.1 — Integrate Langfuse

- **Goal.** Add recognisable tracing without letting it become load-bearing.
- **Concepts.** Tracing versus logging; nested spans; optional instrumentation that never becomes a hard dependency.
- **Files.** `docker-compose.yml`, `app/services/tracing.py` (new).
- **Subtasks.**
  1. Add the self-hosted Langfuse service (MIT licence).
  2. Trace the full pipeline as nested spans.
  3. Make it strictly optional: the app must run identically with tracing
     disabled.
  4. Capture a dashboard screenshot for the README.
- **Tests.** App works with Langfuse absent.
- **Acceptance.** Tracing adds nothing mandatory to the critical path.
- **Interview questions.**
  - Why is tracing optional rather than built in?
  - What does Langfuse give you that your JSON logs do not?
- **Commit.** `feat: add optional Langfuse tracing`

---

# Phase 17 — README, diagrams, demo video, CV and interview preparation

*Everything above is invisible until it is written down for someone who will
not read the code. This phase is the highest-leverage work in the file for
both audiences.*

### [ ] T17.1 — Draw the architecture diagram

- **Goal.** Let a reader understand the system without reading any code.
- **Concepts.** Mermaid; diagramming for a reader who will not read code; making decision points visible.
- **Files.** `docs/architecture.md`, `README.md`.
- **Subtasks.**
  1. Draw the ingestion pipeline and the query pipeline (Mermaid, so it
     renders on GitHub).
  2. Mark the three guardrail layers explicitly.
  3. Mark where each measured decision applies.
- **Acceptance.** A reader understands the system without reading code.
- **Interview questions.**
  - Draw your architecture on a whiteboard in two minutes.
- **Commit.** `docs: add architecture diagrams`

### [ ] T17.2 — Record the demo

- **Goal.** Make the thesis visible in under a minute — refusal included.
- **Concepts.** Demo narrative; what to show first; presenting failure as a feature.
- **Files.** `docs/demo.gif`, `docs/demo.mp4`, `README.md`.
- **Subtasks.**
  1. Record a successful answer with visible citations.
  2. Record a **correct refusal** with its reason — this is the money shot,
     it is the thesis made visible.
  3. Record an adversarial question being blocked.
  4. Keep it under 60 seconds; embed the GIF at the top of the README.
- **Acceptance.** A recruiter understands the project in 30 seconds without
  scrolling.
- **Interview questions.**
  - Why is the refusal the most important thing to show?
- **Commit.** `docs: add demo recording`

### [ ] T17.3 — Rewrite the README as a case study

- **Goal.** Not documentation — a case study. Numbers and pictures above the
  fold; `uv sync` below it.
- **Concepts.** Case study versus documentation; inverted-pyramid writing; claim-to-command traceability.
- **Files.** `README.md`.
- **Subtasks.**
  1. Structure: one-paragraph plain-language summary → demo GIF → the
     problem (confidently wrong answers on legal text are a liability) → the
     ablation table → the failure taxonomy → the classifier result versus its
     baseline → architecture diagram → *then* setup instructions.
  2. Keep the honest origin note that already exists.
  3. State corpus, licence, and cost per query.
  4. Link every claim to the script that reproduces it.
- **Acceptance.** Every number in the README is reproducible by one command.
- **Interview questions.**
  - Which number in your README are you least confident about, and why?
- **Commit.** `docs: rewrite the README as a case study`

### [ ] T17.4 — Write the article

- **Goal.** For freelance, a shareable write-up generates more inbound than
  the repository. For recruiters, it demonstrates communication — the skill
  most often missing in candidates who can build.
- **Concepts.** Technical writing for a reader with no context; why negative results build credibility rather than undermine it.
- **Files.** `docs/article.md`.
- **Subtasks.**
  1. Pick the single most interesting finding — most likely the ablation
     table, or "I trained a classifier to replace my threshold, here is what
     it learned."
  2. Write it for a technical reader who has not seen the repository.
  3. Include the negative results; they are what make it credible.
  4. Publish on LinkedIn or dev.to.
- **Acceptance.** A publishable article exists.
- **Interview questions.**
  - Explain your project to someone non-technical in two sentences.
- **Commit.** `docs: add the project write-up`

### [ ] T17.5 — CV framing

- **Goal.** Turn the measured results into CV lines a recruiter can scan in seconds.
- **Concepts.** Quantified achievement bullets; differentiating inside a saturated project category.
- **Files.** `docs/cv_notes.md`.
- **Subtasks.**
  1. Write two or three CV bullets, each with a number: the ablation gain,
     the false-acceptance reduction, the classifier's margin over baseline.
  2. Write a two-sentence oral pitch.
  3. Avoid "built a RAG system" — the category is saturated. Lead with what
     is measured and refused.
- **Acceptance.** Bullets contain numbers, not adjectives.
- **Interview questions.**
  - In one sentence, what makes this different from every other RAG project?
- **Commit.** `docs: add CV framing notes`

### [ ] T17.6 — Interview rehearsal

- **Goal.** Find the questions you cannot yet answer, while there is still time to fix that.
- **Concepts.** The Problem → Decision → Why → Result (measured) → Cost answer format; discovering gaps by writing answers down.
- **Files.** `docs/interview_prep.md`.
- **Subtasks.**
  1. Collect every "Interview questions" entry from this file.
  2. Answer each in writing, in the Problem → Decision → Why → Result
     (measured) → Cost format.
  3. Mark the ones you cannot answer without looking; those are your real
     gaps — go back and study those specific concepts before moving on.
  4. Prepare answers for the three hardest:
     - What does your guardrail *not* catch?
     - Which of your changes did not work, and why did you keep the finding?
     - If you had two more weeks, what would you do, and why that?
- **Acceptance.** Every question has a written answer you can defend.
- **Interview questions.** All of them.
- **Commit.** `docs: add interview preparation notes`

---

## Deliberately out of scope

Recorded so that "should I add…?" already has an answer, and so that the
refusals read as decisions rather than gaps:

- **GraphRAG, RAPTOR** — real gains, but each is a project-sized subsystem
  that would displace the refusal thesis from the centre.
- **Agentic RAG, Adaptive RAG, Self-RAG, CRAG** — the fashionable 2026
  answer, and the reason to decline: they turn a system you can fully explain
  into an orchestration you cannot. The self-critique idea is absorbed into
  Phase 10's abstention classifier instead.
- **Multi-turn conversation, query rewriting, agents, tool use.**
- **Fine-tuning the generator.**
- **Multimodal, video, federated variants.**
- **Kubernetes, Terraform, cloud deployment** — Compose proves the packaging
  point. Public deployment is a post-completion decision.
- **A React front end** — dilutes an AI-engineering pitch into a full-stack one.

If one becomes genuinely necessary, it goes in this file with a reason
**before** any code is written.

---

## Minimum viable path if time runs short

`Phase 0 → 1 → 2 → 3 → 4 → 5 → 7 → 10 → 14 → 17`

That is: audit, corpus, evaluation, baseline, better parsing, better
embeddings, the trained abstention classifier, a demo, and the write-up — a
system that refuses correctly, proves it with a model measured against a
baseline, shows it working, and explains itself.

Phases 6, 8, 9, 11, 12 add retrieval depth. Phases 13, 15 add production
credibility. Phase 16 is optional.

**The one genuinely bad outcome** is shipping retrieval sophistication
(Phases 8–11) without the evaluation (Phase 3–4). Unmeasured sophistication
is indistinguishable from every other RAG demo — which is the exact trap this
roadmap exists to avoid.
