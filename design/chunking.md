# Chunking: along the agreement's articles

Status: designed and implemented (`CHUNKING=article`, T6.1 and T6.2); measured in
T6.3 (below, once run).

## Problem

The baseline splits every document into 500-character pieces, ignoring its
structure. Measured on the corpus (8 447 chunks):

- **11.7% of the chunks are under 100 characters**, and many are a heading alone:
  "## Article" appears 66 times as a whole 10-character chunk
  (`design/retrieval_diagnosis.md`). A heading-only chunk carries no rule but is
  indexed like any other, and can be retrieved.
- **A piece cut from the middle of an article does not say which article it
  belongs to.** Worked example, Article 6.2 (overtime) of the consolidated text
  `KALITEXT000044253019`:

  | fixed chunking | article chunking |
  |---|---|
  | `chunk_269`: `## Article 6.2` (14 characters, nothing else) | no heading-only chunk |
  | `chunk_271`: "Dans le cas d'un aménagement du temps de travail sur l'année, ..." | `chunk_234`: "## Article 6.2 / Dans le cas d'un aménagement ..." |
  | `chunk_272`: "Ingénieurs et cadres / Le contingent réglementaire s'applique." | `chunk_235`: "## Article 6.2 / Ingénieurs et cadres / Le contingent ..." |

- **A chunk can straddle two articles**, one of them superseded, so its legal
  status is mixed (`design/versioning.md` had to weigh it by characters).
- The heuristic flagged two of the five baseline retrieval failures (q026, q018)
  as answers split across two chunks.

## Decision

Split each document at its article headings first, then split each article into
pieces of at most 500 characters (same size and overlap as the baseline), and put
the article heading at the top of every piece. Drop article sections with fewer
than 20 characters of text (headings with no content).

**Why.**
- An article is the unit a legal answer cites; no piece mixes two articles, so a
  piece's status (in force or superseded) is exact, read from its own heading.
- Repeating the heading gives every piece the words that name its subject
  ("Article 6.2"), which the embedding of a mid-article fragment otherwise lacks.
- The piece size is kept at 500 so that the comparison with the baseline changes
  the boundaries and the heading, not the size.
- Parent-child retrieval (retrieve small pieces, send the whole article to the
  model) was considered and left out: it lengthens the prompt, which the
  `top_k = 10` control showed costs 11% to 26% of generation time, and it would
  change two variables at once.
- Tables: the salary grids in this corpus are not Markdown tables but one value
  per line (position, coefficient, amount), so there is no table structure to
  keep whole. A grid longer than 500 characters is still split inside its
  article, with the article heading on each piece.

**Measurement problem solved first.** The golden labels are fixed-chunk ids, which
do not exist under another chunking. `eval/labels.py` translates them by text
position: a new chunk is relevant when it shares at least half of the shorter span
with a labelled chunk of the same document. Every golden label translates to at
least one article chunk. Because a translated label can cover two chunks, recall
gets harsher; the comparison also reports the hit rate (at least one relevant
chunk in the top k).

**Effect on the index (measured).** 7 639 chunks instead of 8 447; chunks under
100 characters fall from 11.7% to 1.2%; median size 409 characters (376 before),
largest 537 (the heading is added to a 500-character piece).

## Prediction, written before the run

Run `article`: same settings as `versioning` (superseded text filtered, top 3,
no reranker), index rebuilt with `CHUNKING=article`, threshold recalibrated on the
visible split with `scripts/calibrate_threshold.py` because the chunk text
changed.

| measure | expected against `versioning` | refuted if |
|---|---|---|
| hit rate@3, answerable | rises by at least 0.05 | falls |
| q026 or q018 | at least one of them gets a relevant chunk in the top 3 | neither does |
| recalibrated threshold | moves by at least 0.02 (headings in every chunk change the distances) | moves by less than 0.01 |
| retrieval latency | unchanged within noise | median up by more than 1 s |

## Cost: what this does not establish

- The heading pattern (`## `) is specific to the Markdown produced by this
  project's converter; another source format needs another splitter.
- Translated labels are an approximation of what a person would label under the
  new chunking.

## Result (measured, run `2026-10-06_article`)

Compared with `versioning` (same filter, top 3), threshold recalibrated on the
article index to 0.888 by the script's rule (fewest errors on the visible
questions: 0 of 33 in-topic refused, 1 of 11 off-topic accepted).

| measure | versioning | article | prediction |
|---|---|---|---|
| hit rate@3, answerable | 0.773 | 0.773 | refuted (no gain; +0.05 expected) |
| recall@3 / MRR | 0.727 / 0.538 | 0.727 / 0.576 | |
| q026, q018 | q018 found, q026 missed | same | refuted (nothing new) |
| recalibrated threshold | 0.762 | 0.888 | held (moved by more than 0.02) |
| false refusal / false acceptance | 0.121 / 0.318 | 0.000 / 0.500 | |
| substantive answers: answerable, unanswerable, adversarial | 17, 1, 2 | 17, 2, 1 | |
| generation p50 / p95 | 24.8 / 53.0 s | 22.0 / 33.8 s | |

**Decision: keep fixed chunking as the default.** Splitting along articles did not
put more relevant passages in the top 3; it removed the heading-only chunks and
made the superseded flag exact, but those did not show up in the answers. Its
recalibrated threshold is looser (0.888), which trades all 4 false refusals for
10 of 11 adversarial questions and 1 off-topic question passing the distance
check; the model still refused most of them (1 adversarial question was
followed against 2 before). `CHUNKING=article` remains available.

**What surprised me.** The heading repeated in every piece was expected to help
retrieval; it moved MRR slightly (+0.038) but not the hit rate. With versioning
already removing the superseded duplicates, the remaining retrieval misses (q001,
q008, q022, q024, q026) are not boundary problems.

**Measurement fix found on the way.** The first comparison showed the
faithfulness proxy falling to 0.144: the comparison script rebuilt chunk texts
with the fixed chunking for both runs, so article chunk ids pointed at the wrong
text. Each run is now read with its own chunking (`eval/compare_runs.py`, test
added); the corrected value is 0.674.

**Latency.** This run had the machine to itself; `versioning` did not (see
`design/versioning.md`), so the latency columns are not a fair comparison.

## Sources

P10 (structure-aware chunking on financial reports, a result that did not transfer here). Full references and quotes: `design/sources.md`.
