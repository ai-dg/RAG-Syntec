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
