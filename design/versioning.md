# Legal text versions: superseded provisions in the index

Status: implemented (T13.2). Superseded text is tagged at ingestion and filtered at query time.

## Problem

The corpus keeps provisions that have been replaced. In the converted Markdown,
each article appears under a heading, and a superseded version carries
"(non en vigueur)" in that heading (the current one does not). The index treats
both alike, so a superseded passage can be retrieved and quoted as current law.
The reranking evaluation showed it happening: for the severance indemnity
(q005) the reranker promoted an older text ("2 années d'ancienneté") and the
system answered with a rule that is no longer in force (the current text says 8
months of seniority).

## Result (measured, baseline index, 8 447 chunks)

Chunks were labelled "non en vigueur" when the closest preceding article heading
says so. This is a heuristic on the heading structure, not a legal status from
the source.

- **40.4% of the chunks (3 410) belong to superseded articles.** The base text of
  1987 (`KALITEXT000005679895`) has 768 of its 769 chunks flagged; the 2021
  consolidated text (`KALITEXT000044253019`) has 267 of its 680.
- **45% of the chunks the baseline put in the top 3 for answerable questions are
  superseded** (28 of 62).
- **Leaving superseded chunks out of the ranking raises the share of questions
  with a relevant chunk in the top 3 from 0.59 to 0.73** (text-equivalent match;
  0.55 to 0.68 by id), without any model: rank improves for 12 questions. The
  share at the top 1 goes from 0.32 to 0.41, at the top 5 from 0.77 to 0.82.
  Two questions lose their relevant chunk entirely (q016, q022), which is
  explained below. This was measured on the existing ranking of 50 chunks per
  question; the pipeline itself was not changed.

## What the labelling exposed in the golden set

Four chunks listed as relevant in the golden set are themselves under
"(non en vigueur)" headings (q016, q017, q022, q026). Checking the two that
lose their chunk:
- **q022 (how often the point values are renegotiated): the expected answer is
  out of date.** "2 fois par an" comes from Article 7.1 in its superseded
  version. The version in force (Article 7.1 and an amendment, `KALITEXT000050228699`)
  says the minimum hierarchical salaries are examined "une fois par an". The
  question's premise ("valeurs du point") no longer appears in the current text.
- **q016 (child's illness): the listed chunk is the superseded wording**; the
  expected answer (3 days, 5 days if the child is under one year or the employee
  has three children or more) matches the chunk in force (`KALITEXT000044253019`
  chunk 254, and `KALITEXT000053434107` chunk 25), which the baseline did
  retrieve at rank 4.

So some baseline figures are affected: recall@k counts the listed chunk, and the
answer check counts the expected answer, both of which are wrong for these
questions. The size of the effect on the published numbers was not recomputed.

## Decision: tag at ingestion, filter at query time

**Decision.** Each chunk gets an `in_force` flag when it is created
(`app/services/ingestion.py`). The retriever passes `{"in_force": True}` as a
metadata filter to the vector search unless `INCLUDE_SUPERSEDED=true`.

**Why.**
- Deleting superseded text at ingestion would answer "what is the rule today"
  but make "what was it in 2023" impossible to even start, and could only be
  reverted by re-ingesting. A filter keeps the text and makes the choice a
  setting.
- The flag is computed from the character offsets of each chunk
  (`add_start_index` on the splitter) against the article headings of its
  document: a chunk is in force when at least half of its characters fall under
  a heading without "(non en vigueur)". This replaces the first measurement's
  shortcut (look at a heading near the start of the chunk, otherwise carry the
  last status forward); both give 40.4% superseded chunks (3 411 against 3 410).
- Chunk boundaries and chunk ids are unchanged, so every earlier measurement
  stays comparable.

**Golden set corrected.** Five labels pointed at superseded text and were
replaced by the version in force: q016, q017, q026 (same rule, current chunk),
q022 (the expected answer itself was outdated: "twice a year" became "once a
year"), and q020 in the held-out split (same rule, current chunk). This corrects
ground truth; it is not tuning, and the held-out split had not been evaluated.

**Effective dates: not stored.** The acquired pages carry the container title
("... mise à jour par avenant n° 46 du 16 juillet 2021") for 171 of 172 texts,
not each text's own effective date, so there is no reliable date to store. The
DILA/KALI structured export carries start and end dates per article; using it is
the path to answering questions about a past date.

## Cost: what this does not establish

- The status comes from heading text and carry-forward over chunks, not from the
  legal source's own status field; a chunk can be mislabelled when an article is
  long or a heading is missing. The share of errors was not measured.
- The effective date of each text is not captured; only whether an article is in
  force. Answering for a past date, or stating a version date in an answer,
  needs data this check does not use.
- The gain in the top 3 (0.59 to 0.73) is measured on 22 questions with a golden
  set that has just been shown to contain superseded labels; it must be
  re-measured after the labels are corrected.
- 20 of the 22 answerable questions have a relevant text that exists in force.

## Prediction, written before the run (2026-10-06)

Run `versioning`: same settings as the baseline (top 3, threshold 0.74, no
reranker), index rebuilt with the `in_force` tag, superseded text filtered. Both
runs scored with the corrected golden labels.

| measure | expected | refuted if |
|---|---|---|
| recall@3, answerable | rises by at least 0.10 over the baseline rescored with the new labels | rises by less than 0.05 |
| answerable questions failing the answer check | at least 2 fewer than the baseline | not fewer |
| false-refusal rate | rises by at most 0.06 (two questions): the nearest in-force chunk can be farther than the nearest chunk overall | rises by more than 0.06 |
| false-acceptance rate | unchanged or lower | higher |
| retrieval latency | unchanged within noise (a metadata filter) | p50 up by more than 1 s |

## Result (measured, run `2026-10-06_versioning`)

Both runs scored with the corrected golden labels and cut at 3 chunks
(`eval/compare_runs.py`).

| measure | baseline | versioning | prediction |
|---|---|---|---|
| recall@3, answerable | 0.455 | 0.727 | held (+0.273, at least +0.10 expected) |
| hit rate@3 | 0.500 | 0.773 | |
| MRR cut at 3 | 0.356 | 0.538 | |
| answerable questions failing the answer check | 10 | 8 | held (2 fewer) |
| false refusal / false acceptance | 0.121 / 0.318 | 0.121 / 0.318 | held (unchanged) |
| retrieval p50 | 2.6 s | 3.4 s | held (+0.7 s, at most +1 s) |
| generation p50 / p95 | 22.9 / 34.7 s | 24.8 / 53.0 s | not predicted |

Filtering superseded text is the largest retrieval gain measured in this project,
larger than reranking (+0.136 recall@3 with 2.9 s added), with no model and no
added latency by design. Questions that stopped failing: q009, q012, q018; one new
failure, q004.

**Latency is not reliable for this run.** Tests and the code formatter ran on the
same machine during it; the generation p95 of 53 s comes from that contention, not
from the filter, which only adds a metadata condition to the search. A clean
latency comparison needs a run on an otherwise idle machine.

**Threshold.** On this index, `scripts/calibrate_threshold.py` recommends 0.762
(3 of 33 in-topic questions refused, 0 of 11 off-topic accepted) against 4 of 33
at 0.74. The threshold was kept at 0.74 for this run so that only the filter
changed.
