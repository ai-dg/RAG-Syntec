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
