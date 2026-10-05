# Retrieval diagnosis

Status: in progress. T5.1 (duplication) is recorded below; T5.2 (rank of the
relevant chunk) and T5.3 (diagnosis and predictions) are not done yet.

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
