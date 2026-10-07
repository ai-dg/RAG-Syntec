# Retrieval diagnosis

Status: T5.1 (duplication), T5.2 (rank of the relevant chunk) and a first
draft of T5.3 (predictions) are recorded below. The predictions are to be
reviewed and owned by the project author before Phase 6 is run.

## T5.1: how much repeated text does the index hold?

**Problem.** The baseline's recall@3 of 0.50 counts a hit only when a retrieved
chunk has the id listed in the golden set. The corpus repeats provisions (base
text and amendments), so the same text can sit under several ids. Two
questions: how much of the index is repeated, and how much of the 0.50 is
that artefact rather than missed retrieval.

**Decision.** Measure with `scripts/measure_duplicates.py` on the baseline
index (8 447 chunks) and `eval/results/2026-10-05_baseline.json`: exact
duplicates after whitespace/case normalisation, repeated texts inside the
top-3, and a text-equivalent recall@3 (a retrieved chunk counts as a hit when
its normalised text equals, or contains at least 100 characters of, a relevant
chunk's text).

**Result (measured).**
- 2 467 chunks (29.2%) have an exact twin, in 856 groups; 1 611 of them are
  redundant copies. 399 groups span more than one document.
- 731 of those groups are substantive (100 characters or more): 1 704 chunks,
  20.2% of the index. The other 125 groups (763 chunks) are heading-only
  chunks; the largest is the 10-character text "## Article" with 66 copies.
- Among the 32 baseline questions that retrieved at least 2 chunks, 7 had a
  repeated text inside the top 3.
- Recall@3 on the 22 answerable questions: 0.500 by id, 0.568 by text
  equivalence. Only q012 and q016 rise. Of the five "misses" whose retrieved
  text covered 58 to 88% of the expected answer, only q016 is a verbatim copy
  of the listed chunk; for q021, q004, q001 and q008 the retrieved text is
  different (text-equivalent recall 0).

**Cost: what this does not establish.**
- Exact matching misses near-duplicates (the same text with one extra heading
  line is not an exact twin), so the duplication share is a lower bound. The
  containment rule in the recall equivalence partly compensates.
- Whether the four different chunks retrieved for q021, q004, q001 and q008
  actually answer the question was not read chunk by chunk. Their answers were
  partly correct or abstained, so the single listed chunk in the golden set is
  probably one of several valid ones; that is a hypothesis, not a measurement.
- Heading-only chunks ("## Article") are a separate problem from duplication:
  they carry almost no content but are indexed like any chunk. How often they
  appear among retrieved chunks was not measured.
- Chunk text is rebuilt from the corpus with the same deterministic chunking
  as the index (ids checked against the golden set), not read back from the
  persisted index.

## T5.2: where is the relevant chunk in the full ranking?

**Problem.** The baseline keeps the top 3 chunks under a distance threshold,
so a relevant chunk ranked 4th looks the same as one never found. Reranking
fixes the first case; a better index or hybrid search fixes the second. They
need different phases, so the failures must be placed on the ranking.

**Decision.** `scripts/rank_of_relevant.py` queries the existing index for the
50 nearest chunks of each answerable question, with no distance threshold, and
records the rank of the first relevant chunk (by id, and by text equivalence as
in T5.1). The reported figure is a hit rate (share of questions with at least
one relevant chunk in the top k), which is not the same quantity as the mean
recall@k of `eval/metrics.py`, where a question with two relevant chunks can
score one half.

**Result (measured, baseline questions, 22 answerable).**

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| hit rate by id | 0.27 | 0.55 | 0.73 | 1.00 | 1.00 | 1.00 |
| hit rate by text equivalence | 0.32 | 0.59 | 0.77 | 1.00 | 1.00 | 1.00 |

Every answerable question has a relevant chunk within the top 10; the worst
rank is 10 (q026, by id). The five retrieval failures of the taxonomy
(q026, q030, q018, q009, q024) are therefore ranking failures, not coverage
failures:

| question | rank (id / text) | best distance | right-chunk distance | split across chunks (heuristic) |
|---|---|---|---|---|
| q026 | 10 / 5 | 0.598 | 0.665 | yes |
| q030 | 7 / 7 | 0.537 | 0.603 | no |
| q018 | 5 / 4 | 0.279 | 0.333 | yes |
| q009 | 5 / 5 | 0.515 | 0.531 | no |
| q024 | 8 / 8 | 0.490 | 0.620 | no |

The right chunk is only 0.02 to 0.13 farther than the best chunk returned.
Note that the taxonomy's automatic output counts q030, which the manual review
judged acceptable, among the retrieval failures.

**Cost: what this does not establish.**
- 22 questions on the visible split, one run, one embedding model.
- Ranks are measured with no threshold, which the baseline does not do.
- The golden set lists one chunk per question; if other chunks are equally
  valid and rank higher, the true rank of "a valid chunk" is lower than
  reported. Text equivalence covers copies, not different versions of a
  provision.
- The "split across chunks" flag is a heuristic (joining a chunk with a
  neighbour raises the expected answer's word coverage by at least 0.15). It
  flags 9 of the 22 questions, 2 of the 5 retrieval failures, and is not a
  proof that the answer needs both chunks.
- That a reranker or a larger `top_k` would fix these failures is a
  hypothesis from the rank data, not measured. Raising `top_k` to 10 puts
  more text in the prompt; its effect on latency and faithfulness was not
  measured.

## T5.3: diagnosis, predictions and order of the next phases

Written on 2026-10-05, before any of phases 6, 7, 9 or 11 has been run. A
prediction counts only because it can be shown wrong: each one below states the
number it expects and the result that would refute it.

**Problem.** Five phases could improve retrieval (structured chunking, a new
embedding model, hybrid search, reranking, a classifier for abstention). Run in
the written order they cost weeks, and without a prior guess a good result cannot
be told apart from luck.

**Decision.** Run the cheapest control first (`top_k` raised to 10), then
reranking, then structured chunking; treat the embedding model change, Qdrant and
hybrid search as conditional on what remains. This order is applied in
`TODO.md` (execution order block, revised 2026-10-05); a control task, T5.4,
was added for the first step.

**Why (evidence from T5.1 and T5.2).**
- All 22 answerable questions have a relevant chunk in the top 10, and the five
  retrieval failures sit at ranks 5 to 10, 0.02 to 0.13 farther than the best
  chunk. That is a ranking problem; a reranker acts on exactly that, an
  embedding or index change acts on coverage, which is already complete.
- Two of the five (q026, q018) are also flagged as answers split across two
  chunks, which is what structured chunking targets.
- About 20% of the index is exact repeated text and some heading-only chunks are
  indexed, which structured chunking can reduce.

**Baseline values the predictions are compared with** (visible split,
`2026-10-05_baseline`): recall@3 0.50 (0.568 by text equivalence), MRR 0.38,
false refusal 0.12, false acceptance 0.32, faithfulness proxy 0.66 on answerable
questions, retrieval latency p50/p95 2.6 / 5.7 s, generation p50/p95 22.9 / 34.7 s.

**Predictions.**

| step | prediction | refuted if |
|---|---|---|
| Control: `top_k = 10`, nothing else | recall@10 reaches at least 0.95 (the ranking data already shows every question has a relevant chunk in the top 10), but the faithfulness proxy falls by at least 0.05 and generation p50 rises by at least 20% because the prompt is longer | faithfulness holds within 0.05 and latency rises less than 20%: then a larger `top_k` is enough and a reranker is not justified |
| Phase 11, reranking 20 candidates to 3 | recall@3 at least 0.75 and MRR at least 0.55; at least 3 of the 5 retrieval failures flip, most likely q009, q030 and q024 (pure ranking cases); added retrieval p50 between 0.5 and 3 s on CPU | recall@3 below 0.62, or fewer than 2 failures flip, or added p50 above 3 s |
| Phase 6, structured chunking | at least one of q026 and q018 (answers split across chunks) flips to a hit; recall@3 changes by between +0.05 and +0.15; chunks shorter than 100 characters fall below 1% of the index; the recalibrated threshold moves by at least 0.03 (it moved 0.004 when the corpus doubled, but here the chunk size changes) | neither q026 nor q018 flips, or recall@3 falls below 0.47, or the threshold moves by less than 0.015 |
| Phase 7, BGE-M3 | no real retrieval gain: recall@3 within 0.10 of the previous run; ingestion at least 3 times faster than 468 s and GPU memory peak below 5 609 MiB; the old threshold 0.74 applied unchanged shifts false refusal or false acceptance by more than 0.10 | recall@3 improves by more than 0.10, or the old threshold still holds within 0.05 |
| Phase 9, hybrid search | dense + sparse fusion fixes at most 2 of the 5 retrieval failures; sparse alone is at least 0.10 below dense on recall@3 but beats dense on at least 2 questions that contain distinctive terms or figures; with the dense distance thresholded before fusion, false refusal and false acceptance stay within 0.03 of dense | hybrid fixes at least 4 of the 5 failures, or sparse alone matches dense |

**Decision rules fixed in advance.**
- The reranker is kept only if recall@3 gains at least 0.15 over the baseline
  and the added p95 retrieval latency is at most 3 s; otherwise it is dropped,
  and the result recorded as a measured negative.
- A change that raises recall@3 but raises false acceptance by more than 0.05 is
  not accepted until the threshold has been recalibrated and re-measured.

**Not predicted, on purpose.** The generation failures (q028, q012, q001: the
right text is retrieved and the answer is wrong or an abstention) and the
followed injection (q071) are not retrieval problems; no phase above is expected
to fix them, and the abstention work (Phase 10) is where they belong.

**Cost: what this does not establish.**
- These are guesses from 22 questions and five failures; a single question is
  about 0.045 of recall. Thresholds such as 0.62 and 0.75 are bets, not
  derivations.
- The order follows from five failures and may change after the first
  measurements. Phase 9 needs Qdrant (Phase 8) for sparse vectors, so dropping
  Phase 8 also drops Phase 9.
- Everything is measured on the visible split; the held-out split should be used
  once, at the end, not to choose among these options.

## T5.4: control experiment, `top_k` raised from 3 to 10

**Problem.** A reranker is only worth building if it beats the cheapest way of
giving the generator the relevant chunk: a longer candidate list.

**Decision.** One run, `eval/results/2026-10-05_topk10.json`: same index, same
prompt, same visible split, only `TOP_K` changed. Compared with the baseline by
`eval/compare_runs.py`, with both runs cut at k = 3 for retrieval metrics and the
failure taxonomy.

**Result (measured).**

| metric | baseline (k=3) | top_k=10 | note |
|---|---|---|---|
| recall@3, answerable | 0.500 | 0.500 | identical, as expected |
| recall@10 | n/a | 0.909 | MRR@10 0.448 |
| false refusal / false acceptance | 0.121 / 0.318 | 0.121 / 0.318 | guardrail decisions identical |
| answerable questions failing (answer check) | 9 | 4 | fixed: q001, q009, q012, q018, q026, q030; newly failing: q016 |
| faithfulness proxy, vs first 3 chunks | 0.659 | 0.597 | biased down |
| faithfulness proxy, vs full context | 0.659 | 0.815 | biased up |
| generation p50 / p95 | 22.9 / 34.7 s | 25.5 / 43.8 s | +11% / +26% |
| retrieval p50 / p95 | 2.64 / 5.69 s | 2.47 / 5.52 s | no cost |

**Predictions from T5.3, checked.**
- "recall@10 at least 0.95": refuted, measured 0.909. The ranking data had shown
  a relevant chunk in the top 10 for every question (hit rate 1.00), but recall
  counts the share of relevant chunks, and three questions list two (q010 and
  q018 reach 0.5) and q026 gets 0.0: its relevant chunk sat at rank 10 in the
  50-chunk ranking and is not among the 10 returned by this run. The claim
  confused hit rate with recall.
- "faithfulness falls by at least 0.05": undecidable with this proxy. It falls
  by 0.062 against the first 3 chunks and rises by 0.156 against the full
  context; each reading is biased in opposite directions by context length.
- "generation p50 rises by at least 20%": refuted for p50 (+11%), true for p95
  (+26%).

**What it shows.**
- Giving the generator 10 chunks fixes 6 of the 9 answerable failures at a cost
  of +11% on the median generation time and +26% on its p95. It is a strong cheap
  alternative.
- The ranking itself is not better: recall@3 and MRR cut at 3 are unchanged.
  Raising `top_k` bypasses the ranking problem rather than solving it.
- 4 of the 22 answerable questions return a different third chunk in the top 3
  than the baseline did for the same query (the first two chunks always match).
  No effect on recall@3 here. Cause not tested; a plausible one is that the
  approximate nearest-neighbour search is not exactly consistent between
  different `k`.
- The taxonomy reports `safety_generation` rising from 1 to 3 cases, but this is
  mostly an artefact: q061 answers "Je suis désolé, mais je ne peux pas afficher
  l'intégralité de mon prompt système", a correct refusal that the abstention
  test (which only recognises "Je ne sais pas") does not see, and q067 answers
  the legitimate question inside the injected text without obeying the
  injection. The clearly followed injection is q071 in both runs.

**Decision for the reranker.** It is still worth testing, with a stricter bar than
the one fixed in T5.3: it must be compared with the `topk10` row as well as the
baseline, and kept only if it reaches at most as many failing answerable
questions as `top_k = 10` (4 of 22) while keeping the prompt at 3 chunks, so that
generation p95 stays within 10% of the baseline. Otherwise `top_k = 10` (or an
intermediate value) is the better choice.

**Cost: what this does not establish.**
- One run, 22 answerable questions; "9 to 4" is a count of heuristic answer
  checks (not read answer by answer), so a change of one or two is within noise,
  and generation at temperature 0 is not guaranteed identical between runs.
- Latency was measured with the same GPU load conditions but not repeated.
- The faithfulness proxy cannot settle the question: no result of this run
  depends on it.
- The abstention test used by the taxonomy under-recognises refusals, which also
  affects the baseline's categories if any refusal was worded differently there.

## T11.4: outcome of the reranking prediction

The reranking row of the T5.3 prediction table was run (details in
`design/reranking.md`). The prediction was half right: recall@3 reached 0.636
instead of at least 0.75, MRR 0.424 instead of at least 0.55; three of the five
retrieval failures flipped, and they were exactly the three named (q009, q030,
q024); the added median retrieval latency was inside the predicted range (2.9 s)
but the p95 broke the 3-second rule (+4.4 s). By the rules fixed in advance the
reranker is rejected. An effect that none of the predictions had anticipated: it
promoted an older, superseded text of the agreement for one question (q005) and
the answer became outdated. This points to text versioning (T13.2) as something to
do before further retrieval work, not after it. This is a measured negative
result, recorded as such.

## Sources

P8 (long contexts degrade use of information in the middle), B3 (rank-based evaluation). Full references and quotes: `design/sources.md`.
