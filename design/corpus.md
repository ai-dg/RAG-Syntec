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

## Licensing

Verified against the license's own text (`data.gouv.fr`'s hosted copy of
the Etalab 2.0 text), not assumed from the name alone:

- **License:** Licence Ouverte / Open Licence 2.0 ("Etalab 2.0"), the
  standard license for French public-sector open data, published by
  Etalab. Free, non-exclusive, worldwide, no time limit — reuse, adapt,
  redistribute, and commercial use are all explicitly permitted.
- **Legal basis:** the CRPA (*Code des relations entre le public et
  l'administration*) — the framework establishing that administrative
  information is freely reusable by default. Two different specific
  article citations turned up across sources for this project
  (`L.300-2`/`L.323-2` on one page, `L.321-1` on another) — noted rather
  than picked arbitrarily; the license and its legal basis are confirmed,
  the exact article number is not settled by this pass.
- **Attribution requirement — the one binding condition:** the reuser must
  mention the information's *paternité* — its source (at minimum, the name
  of the licensor) and the date it was last updated. For this project:
  attribute Légifrance/DILA as the source, and each Légifrance document
  already carries its own last-modification date (captured per-file in
  `docs/`, e.g. the base text's "Dernière modification : 2024-07-01").

## GDPR / personal data position

**No personal data is processed by this corpus.** This is a normative
legal text — a collective bargaining agreement, its annexes, and its
salary-grid updates — not a record of identifiable individuals.

Checked directly rather than assumed: `grep`-scanned every file in `docs/`
for anything email-shaped. One match, in
`039_textes-attaches_commission-paritaire-permanente-de-negociation-et-d-interpre.md`
— `secretariatcppni@CCN-BETIC.fr`, the institutional secretariat address of
the CPPNI (a joint negotiation commission), not a named individual. No
other personal identifiers found.

**This position does not extend to a different scenario.** If a client
asked to run this same pipeline over their own HR files (individual
contracts, payslips, performance reviews), that would process personal
data under GDPR, and the guardrail/retrieval/logging design would need
real changes: PII scrubbing before indexing, access control per employee,
a data-retention policy, and a lawful basis for processing — none of which
this project currently has, because none of it currently needs to.

## Not yet done

- **`T2.1`/`T2.2`** (Phase 2, not started): the corpus in `docs/` was not
  acquired through a reproducible fetch script, and is not yet versioned
  with a hashed manifest. It exists in the repo without a documented
  provenance beyond this file's verification pass. Fetching it properly
  (PISTE API or documented manual download) and hashing it remains on the
  roadmap, unblocked by this decision.
- **Amendments scope not yet acted on** — see Cost above; deferred to
  evidence from Phase 3's evaluation set, not decided here.
