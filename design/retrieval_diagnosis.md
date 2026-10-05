# Retrieval diagnosis

Status: in progress. T5.1 (duplication) and T5.2 (rank of the relevant chunk)
are recorded below; T5.3 (diagnosis and predictions) is not done yet.

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
