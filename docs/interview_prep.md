# Interview preparation

Every interview question from `TODO.md`, with a short answer and the file that
backs it. These are notes to study from, not a script: for each one, check that
you can say it without reading, and mark the ones you cannot with **[gap]**.
Those are what to study next.

## The three hardest questions

### What does your guardrail not catch?

- **Problem.** A distance threshold measures how close a question is to the
  corpus, not whether the corpus answers it.
- **Decision.** Distance threshold (0.74) on the best chunk, plus an input check
  for injection phrasing and an output check on the answer's lexical support.
- **Why.** Each layer catches a different failure; none is enough alone.
- **Result.** With all layers on, 0 of 11 off-topic and 0 of 11 adversarial
  questions are answered; but the distance check alone lets 7 of 11 adversarial
  and 7 of 11 in-topic-unanswerable questions through, because they are written
  about the agreement's own subject (`design/abstention.md`).
- **Cost (what it does not catch).** An injection phrased differently from the
  patterns (0 of 4 held-out injections matched); an invented answer built from
  the context's own words (q036 passed the output check); a wrong value or a
  negation written with the right words; 4 of 33 in-topic questions are refused
  at 0.74.

### Which of your changes did not work, and why did you keep the finding?

- **Reranker** (`design/reranking.md`): rule fixed before the run (+0.15
  recall@3, no worse than top_k=10, at most +3 s at p95). Measured +0.136, more
  failures than the control, +4.4 s at p95, and it promoted a superseded article
  (an outdated answer). Kept because it pointed at the real problem: 40% of the
  index was superseded text.
- **Abstention classifier** (`design/abstention.md`): cross-validated ROC-AUC
  0.862 against 0.898 for the best distance alone. Kept as a measured negative:
  the distance already carries most of the signal.
- **Article chunking** (`design/chunking.md`): no hit@3 gain on top of
  versioning.
- **LLM judge** (`design/generation_eval.md`): 55% agreement with my labels, not
  used.
- Why keep them: a negative result with its numbers shows the positive ones were
  tested the same way.

### If you had two more weeks, what would you do, and why that?

1. An end-to-end run on the held-out split was made once at the end
   (`eval/RESULTS.md`); next, enlarge the question set: with 22 answerable
   questions, one question is 0.045 of a rate and most differences are noise.
2. Use the DILA/KALI structured export, which has start and end dates per
   article: it would replace the heading heuristic for "in force" and allow
   questions about a past date.
3. Read the remaining answer failures one by one (q001, q002, q022, q024, q026,
   q028): the automatic answer check is lexical and some may be correct
   paraphrases.

## All questions

### Phase 0: baseline and fixes

- **Why record a baseline before touching anything?** Every later claim is a
  difference from it; without it "better" has no reference (`design/audit.md`).
- **Your GPU has 8 GB. Which models fit?** `qwen3-embedding:8b` (4.68 GB) fits;
  `gemma4` (9.61 GB on disk) does not, Ollama keeps about 3.2 GB of it on the GPU
  and the rest on CPU, and swaps the models in and out (`design/audit.md`,
  `design/reranking.md`).
- **pydantic_settings works today; why is it still a bug?** It was only
  installed as a dependency of another package; an upgrade of that package could
  remove it. Declared explicitly in `pyproject.toml`.
- **Why does `list(set)` change order between runs but not within one?** String
  hashes are salted per process (`PYTHONHASHSEED`); within a process the salt is
  fixed. Sources are now sorted.
- **Why does that matter with an evaluation harness?** Two runs of the same code
  would differ, and a difference could be read as an effect.
- **Why not a module-level `settings = Settings()`?** It reads the environment
  at import time, before tests can set it; `get_settings()` with `lru_cache` reads
  it once, on first use.
- **How do tests override cached settings?** The fixture sets the environment and
  calls `get_settings.cache_clear()` before and after each test
  (`tests/conftest.py`).
- **Why inject the vector store rather than patch `get_vector_store`?**
  `set_vector_store` is a public seam; patching couples tests to where the
  function is imported.
- **Why does a refusal return HTTP 200 and not 404?** The request was valid and
  the system answered: "the documents do not support an answer" is an answer,
  with `refusal_reason`.
- **What does structured logging give you over grep?** Fields you can filter and
  aggregate (decision, best distance, latency per stage) without parsing text.
- **Why a correlation id before you need it?** Adding it after the first
  incident is too late to debug that incident; now every log line of a request
  carries it (`design/observability.md`).
- **Your threshold is 0.9 (now 0.74). Of what?** Squared L2 distance between
  normalised vectors: lower is closer; 0.74 is a cosine similarity of 0.63
  (`design/guardrail.md`, `scripts/inspect_metric.py`).
- **What happens if the embedding model changes?** The distances are in another
  vector space; the threshold must be recalibrated
  (`scripts/calibrate_threshold.py`).
- **What does this guardrail not catch?** See the first hard question.

### Phases 1 and 2: corpus

- **Why freeze the corpus before writing questions?** Labels point at chunks of a
  given corpus; if the corpus changes, the labels silently go stale (it happened:
  5 labels pointed at superseded text, `design/versioning.md`).
- **Why exclude amendments in the first version?** See `design/corpus.md`, scope
  decision. **[check you can explain it]**
- **Licence and its requirement?** Licence Ouverte / Open Licence 2.0 (Etalab):
  reuse allowed, including commercially; the condition is attribution with the
  source and date (`design/corpus.md`).
- **A client wants to run it on their HR files?** Personal data enters: GDPR
  applies (legal basis, retention, access), and technically the index must be per
  client, the data must stay local (Ollama makes that possible), and the golden set
  must be rebuilt for their documents.
- **Why script acquisition?** Reproducible from a list of ids, with a manifest of
  hashes to verify it (`scripts/fetch_corpus.py`, `data/manifest.json`).
- **The agreement is amended next month?** Re-fetch, re-convert, rebuild the
  manifest, rebuild the index (the app rebuilds it automatically when the
  documents change their settings), re-run the evaluation and check the labels.
- **Why is a corpus hash part of a result?** Two runs are comparable only on the
  same documents.
- **No DVC or Git LFS; defend it.** 172 Markdown files are small enough for git;
  a manifest of hashes gives verification without the extra tool
  (`design/corpus.md`).

### Phase 3: evaluation

- **Unanswerable vs off-topic, both refused; why two classes?** They fail at
  different points: off-topic should be stopped by the distance check,
  unanswerable passes it (it is about the agreement) and must be refused by the
  model or a later layer. Mixing them hides which layer fails.
- **Why label chunks and not just answers?** To tell a retrieval failure (the
  right chunk was not retrieved) from a generation failure (it was, and the
  answer is wrong).
- **How did you decide a question was unanswerable?** Searched the corpus for the
  topic and checked by reading that no text answers it. **[say how many you
  checked by hand]**
- **Why hold out a split with nothing to train?** Choosing thresholds and
  components is fitting to the questions too; the held-out split measures that
  bias.
- **How to detect leading questions?** Questions that copy the wording of their
  chunk get easy distances; compare their distance to paraphrased ones.
  **[not measured]**
- **MRR, and when it beats recall@k?** Mean of 1/rank of the first relevant
  chunk; better when only the first relevant result matters, as with a short
  prompt.
- **recall@10 = 0.95 and poor answers: what do you check?** The rank of the
  relevant chunk (is it in the part that reaches the prompt?), then generation
  failures with the right chunk present (the taxonomy).
- **A guardrail that refuses everything?** False acceptance 0, false refusal 1.0;
  only the two rates read together expose it (`tests/test_eval_metrics.py`).
- **Which error is worse for a legal assistant?** A false acceptance with a
  confident wrong answer: a refusal sends the user to a person, a wrong rule can
  be acted on.
- **An LLM grading an LLM: why not circular?** Because its agreement with human
  labels was measured: 55%, so it is not used (`design/generation_eval.md`).
- **Why implement RAGAS-style metrics yourself?** To know exactly what each
  number computes and to test it.
- **Why the git SHA and corpus hash in a result?** To trace a number to the code
  and documents that produced it.
- **Why p95 rather than mean?** Latency has a long tail (model swaps take
  seconds); the mean hides it.
- **Retrieval vs generation share, and what it changed?** Baseline: 5 retrieval
  failures, 6 generation, 11 guardrail-related out of 22. It pointed at ranking,
  and then at superseded text (`design/evaluation.md`).
- **A case retrieved correctly but answered wrongly?** q028: the listed chunk was
  retrieved and the model answered "Je ne sais pas".

### Phases 4 and 5: baseline and diagnosis

- **Your threshold moved when the corpus grew. Mechanism?** A question's score is
  the minimum distance over all chunks; more chunks can only lower it. Here it
  moved by 0.004 because the added text was the same domain
  (`design/ingestion.md`).
- **Why not keep the old threshold?** Its validity on the new corpus would be an
  assumption; recalibrating costs a minute.
- **Baseline recall and dominant failure mode?** recall@3 0.45 (current labels);
  failures dominated by ranking, not missing information.
- **Which change did the numbers point to first?** Ranking: the right chunk was
  within the top 10 for every question. That pointed to reranking; reranking then
  revealed the superseded-text problem, which was the real first fix.
- **Why can recall@k understate retrieval here?** About 20% of the index is
  repeated text; a copy of the right passage under another id counts as a miss.
- **What does duplication do to the top-k?** Copies take slots that could hold
  other relevant passages (7 of 32 top-3 lists had a repeated text).
- **recall@3 is 0.50: reranker or better index?** The rank of the relevant
  chunk: within the top 10 for all questions, so ranking.
- **Did the diagnosis point to the right first change?** Partly: ranking was the
  symptom; superseded text was the cause.
- **Why test top_k = 10 before a reranker?** It is the cheapest way to get a
  wider list into the prompt; a reranker must beat it.

### Phase 6: chunking

- **Why is splitting a legal article mid-sentence worse than prose?** A rule and
  its exception or condition can end up in different chunks; the retrieved piece
  then states the rule without its condition.
- **Retrieval vs generation granularity?** Small chunks match questions
  precisely; the model needs enough context. Parent-child retrieval was left out
  because longer prompts cost generation time.
- **How is the article reference kept to the citation?** Each chunk carries
  `article` in its metadata and the heading in its text; citations return it.
- **Chunk size changed, recall up, false acceptance up: mechanism?** Distances
  shift with chunk text, so the recalibrated threshold moved to 0.888, which lets
  more adversarial questions through (`design/chunking.md`).

### Phases 7, 8, 9: not run

- **BGE-M3, Qdrant, hybrid search?** Skipped: every answerable question had a
  relevant chunk in the top 10, so there was no coverage failure for them to fix
  (`TODO.md`). Know the concepts anyway: a sparse embedding is a learned weight per
  vocabulary token (BM25 uses term counts and document frequencies); RRF sums
  1/(k + rank), so it needs no common score scale; after RRF a distance threshold
  is meaningless unless it is applied to the dense distance before fusion.
  **[gap: study these if the job involves hybrid search]**

### Phase 10: abstention classifier

- **Why is accuracy the wrong metric?** 33 of 55 should be refused; "refuse all"
  is 60% accurate.
- **Why is dispersion informative?** An answerable question tends to have one
  clearly closest chunk; a question about the domain with no answer tends to be
  equally close to many.
- **How did you avoid leakage?** Features come only from distances and the
  question length, computed before generation; the held-out split was not used
  to fit or choose anything.
- **What do ~100 rows limit?** A 9-parameter linear model at most; standard
  deviations of the AUC are about 0.02 in cross-validation.
- **Which feature mattered most?** Mean top-5 distance and best distance; then
  question length, which reflects that adversarial questions are long in this
  set.
- **How was the operating point chosen?** The highest cut that refuses at most
  10% of answerable questions out of fold; F1 weighs both errors equally, which is
  not the trade-off wanted here.
- **How do you know you are not overfitting?** Repeated cross-validation, then
  one read of the held-out split.
- **"Beat the threshold by 4 points": significant?** It did not beat it. On 20
  held-out questions one question is 5 points, so a 4-point difference is noise.
- **Why keep the threshold mode?** It is the better-measured one here, and the
  fallback if the model file is missing.
- **If the model file is missing?** `load_model` logs a warning and the threshold
  is used (`tests/test_abstention.py`).

### Phase 11: reranking

- **Why not run the cross-encoder over the whole corpus?** 8 447 pairs per
  question at 16 ms per pair on the GPU is over 2 minutes.
- **Why does a cross-encoder score better, mechanically?** It reads question and
  passage together, so attention links their words; a bi-encoder compares two
  vectors computed separately.
- **Why load the model at startup?** Loading takes about 1 s and 1.2 GB; per
  request, every call would pay it.
- **Why CPU and not GPU?** Measured: CPU float32 took 7.6 s for 20 pairs; GPU
  0.33 s but kept resident it made gemma fail for lack of memory, so the model is
  lent to the GPU per request.
- **Three score scales: how to keep them coherent?** Never compare across them:
  the threshold reads only the dense distance, the reranker score is stored
  separately in metadata.
- **Was the reranker worth it?** No, by its own rules (see the hard questions).

### Phase 12: ablation and CI

- **Largest gain per unit of complexity?** Filtering superseded text: +0.27 hit@3,
  one metadata filter, no model.
- **Did any change make something worse?** Reranking (an outdated answer, +4.4 s)
  and article chunking (looser threshold, more adversarial passing the distance
  check). Neither was kept.
- **Why exclude the eval test from CI?** It needs the 8 447-chunk index and the
  embedding model; CI stays fast and hermetic.
- **Which metric does it guard?** Off-topic acceptance (budget 0) and in-topic
  refusal (budget 20%), the two errors a guardrail change moves.

### Phase 13: citations and versions

- **How do you prevent invented citations?** Passages are numbered; any cited
  number with no retrieved passage is removed and logged; the run measured 0.
- **Why is a document-level source insufficient?** The base text alone is 769
  chunks; a lawyer needs the article.
- **The agreement is amended tomorrow?** Re-fetch, rebuild the manifest and the
  index; amended articles are marked "(non en vigueur)" by Légifrance and filtered
  out; re-run the evaluation.
- **A rule as it stood in 2023?** No: only in force or not is stored, not dates.
  The DILA/KALI export has dates per article.
- **Injections scored 0.64 to 0.83; why did distance fail?** They are written
  about the corpus' subject, so their nearest chunk is close; distance measures
  topic, not intent.
- **Each layer adds false refusals; worth it?** Measured: the output check
  withdrew none of 17 correct answers; the input check flagged none of 63
  non-adversarial questions.

### Phases 14 to 17

- **Why does the health check need to know about the store (Qdrant in the
  roadmap)?** `/ready` must say whether a query can succeed; the index and Ollama
  are what it needs. `/health` only says the process runs.
- **Why a 200 with a reason rather than an error?** A refusal is a valid outcome
  the client should display, not retry.
- **Why show retrieved chunks and scores?** The user can check the answer against
  the text, and see why a refusal happened.
- **Why is displaying a refusal a product decision?** A refusal shown as an error
  reads as a broken system; shown with its reason, it reads as the system being
  careful.
- **Why must unit tests run without a network?** So that a failure means the code
  is wrong, not that a service was down.
- **Coverage, and what worries you most?** 88% of `app/`; `eval/run.py` is at 0%
  and every number comes from it (`design/testing.md`).
- **Why wait for a healthcheck rather than retry?** The app needs Ollama before
  it can build the index; `depends_on: service_healthy` orders startup.
- **Walk me through one query.** `design/observability.md`: 162 ms retrieval,
  33.5 s generation, support score 1.0, all lines under one request id.
- **Debugging a slow query?** Token counts and Ollama's model-load time per call.
- **Langfuse?** Not used: request ids and structured logs cover tracing here.
- **Draw the architecture.** README diagram: input check, filtered vector search,
  distance decision, generation with citations, output check.
- **Why show the refusal in the demo?** It is the thesis: the system says when the
  documents do not support an answer.
- **Least confident README number?** The answer-failure counts: they come from a
  lexical check, not from reading each answer.
- **Two sentences for a non-technical person?** See `docs/cv_notes.md`, oral
  pitch.
- **What makes it different?** Each component was kept or rejected on numbers
  measured before and after, and two were rejected.
