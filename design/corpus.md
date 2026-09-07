# Corpus design

## What this corpus is

The French Syntec collective bargaining agreement — the national agreement
for technical engineering firms, engineering-consulting firms, and
consulting companies ("bureaux d'études techniques, cabinets
d'ingénieurs-conseils et sociétés de conseils").

Verified directly against Légifrance (not from memory, not from a secondary
source alone — cross-checked against three independent aggregators first,
then confirmed on the primary source):

- **IDCC:** `1486`.
- **Official title:** *Convention collective nationale des bureaux d'études
  techniques, des cabinets d'ingénieurs-conseils et des sociétés de
  conseils.*
- **Base text date:** originally 15 December 1987, substantially rewritten
  by *avenant n° 46* of 16 July 2021.
- **Extension:** by *arrêté* of 5 April 2023, published in the *Journal
  officiel* on 28 April 2023.
- **Légifrance identifier:** `KALITEXT000044253019`.

## What's in `docs/` today

109 Markdown files, 4183 chunks at `CHUNK_SIZE=500` / `CHUNK_OVERLAP=100`
(measured in `design/audit.md`, T0.1) — inside the `T1.1` target range of
3,000–15,000 chunks.

Spot-checked (not assumed) against Légifrance directly: the base text file
carries the same Légifrance ID as the one confirmed above, the correct
extension order date, and per-article legal references (`KALIARTI...`),
plus a full old-to-new article concordance table. This is a faithful,
structured transcription of the official text — not an approximation, and
not a PDF-derived extraction (Légifrance's KALI database is natively
text/XML, article by article; PDF is one rendering of it among others, and
specifically the rendering this project's Phase 5 already knows destroys
tables — see `README.md`'s origin note and the planned PyPDF→Docling swap).

The 109 files split into three groups, by filename prefix:

| Group | Count | Content |
|---|---|---|
| `texte-de-base` | 1 | The base convention text (post-avenant-46 rewrite), articles 1.1–13.6 |
| `textes-attaches` | 88 | A mix: structural annexes (classification grids for ETAM and for engineers/executives, the minimum-pay grid, enquêteurs) **and** standalone amendments/accords on specific topics (partial unemployment, paritarism financing, complementary health insurance, etc.) |
| `textes-salaires` | 20 | Multiple **dated** salary-grid updates (e.g. "au 1er janvier 2025", "au 1er août 2022"), several without a visible date in the filename |

## Scope decision

**Problem.** `textes-attaches` mixes two structurally different kinds of
content under one label (load-bearing annexes vs. one-off amendments), and
`textes-salaires` holds multiple time-stamped versions of the same salary
grids rather than a single current one. Left as-is, the corpus can return an
outdated minimum-salary figure as if it were current, or surface a narrow
one-off accord with the same weight as the base convention.

**Decision.** Keep the full 109-file corpus as-is for v1, unpruned. Do not
yet split `textes-attaches` into annexes vs. amendments, and do not yet
reduce `textes-salaires` to the most recent version only.

**Why.** Not measured yet which way it should be split — pruning without
evidence would be a guess, not a decision.

**Cost, stated honestly rather than hidden:**
- A question about the current minimum salary can retrieve a superseded
  `textes-salaires` figure and present it without any signal that a newer
  one exists elsewhere in the corpus.
- One-off amendments in `textes-attaches` (e.g. a specific accord on
  partial unemployment) sit at the same retrieval weight as the load-bearing
  classification/salary annexes, despite covering a narrower, more
  situational scope.
- This will very likely need revisiting once the evaluation set (`TODO.md`
  Phase 3) exists and failure cases can be measured directly — e.g. a
  golden question about "the current minimum salary" retrieving the wrong
  dated grid would be direct evidence for pruning, rather than a guess about
  it.

## Not yet done

- **`T2.1`/`T2.2`** (Phase 2, not started): the corpus in `docs/` was not
  acquired through a reproducible fetch script, and is not yet versioned
  with a hashed manifest. It exists in the repo without a documented
  provenance beyond this file's verification pass. Fetching it properly
  (PISTE API or documented manual download) and hashing it remains on the
  roadmap, unblocked by this decision.
- **Amendments scope not yet acted on** — see Cost above; deferred to
  evidence from Phase 3's evaluation set, not decided here.
