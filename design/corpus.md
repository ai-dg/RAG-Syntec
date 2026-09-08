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

## Reproducible acquisition (T2.1)

`docs/` (109 files, indexed by the running app) still has no documented
provenance — that gap is unchanged by what follows. A separate, parallel
corpus now does: `scripts/fetch_kali_id.py` discovers all 172 KALITEXT
document identifiers for IDCC 1486 by scraping Légifrance's own index page
(`conv_coll/1486`) rather than depending on `docs/` already existing (which
would be circular — the whole point is reconstructing from nothing).
`scripts/fetch_corpus.py` downloads each one (idempotent, resumable, rate
-limited, atomic writes so a crash mid-download can't leave a corrupted
file mistaken for a completed one) into `data/raw/`. `scripts/convert_raw_data.py`
extracts clean article-level Markdown into `data/converted/` (172 files,
~3.1 MB) using BeautifulSoup against Légifrance's real DOM structure
(`article.list-article-consommation` → `.name-article` + `.content`),
verified to skip UI chrome (buttons, tabs, print controls) that a naive
full-page text dump would have captured.

`docs/` and `data/converted/` are two different corpora today — 109 files
of unknown provenance vs. 172 files fully reproducible from nothing. Not
yet reconciled; see `T4.1` in `TODO.md`.

## Corpus hash and manifest (T2.2)

**Problem.** "172 files, 3.1 MB" is not an identity. Two corpora with the
same file count can differ by one character in one file and produce
different evaluation numbers — without a hash, "which corpus was this
measured on?" has no answerable form months later, the same way an
un-hashed, un-committed pile of code has no way to say "which version was
this bug filed against?"

**Decision.** `scripts/build_manifest.py` computes a SHA-256 per file
(binary-mode read — text-mode reads can silently normalize line endings,
changing the hash of a file git considers unchanged) plus one corpus-level
hash (SHA-256 over the sorted, concatenated per-file hashes — sorted
explicitly, since directory listing order is not guaranteed stable across
filesystems or runs, the same nondeterminism class as `PYTHONHASHSEED` in
`T0.3`). `data/manifest.json` — the manifest itself, not the 172 documents
— is committed to the repo; `data/raw/` and `data/converted/` stay
gitignored, regenerable from the manifest's own `source_url` fields plus
the fetch pipeline. `verify()` recomputes every file's hash from disk and
reports exactly which filenames are `missing`, `added`, or `modified`
against the committed manifest — not a pass/fail boolean, since a boolean
can't be debugged.

**Why not DVC or Git LFS.** Both solve a problem this corpus doesn't have
yet: large binary files (gigabytes, not megabytes) and high churn (frequent
re-versioning of large intermediate states, with efficient binary diffing).
At 172 Markdown files / 3.1 MB, regenerated by a deterministic pipeline and
rarely changed (Légifrance does not republish this convention weekly), a
committed JSON manifest plus a verification command gives the same drift
-detection guarantee without the operational cost of remote storage
config, `.dvc` files, or a `dvc pull` step in CI. This reverses if the
corpus later grows to tens of thousands of frequently-updated documents —
a real trade-off, not a dismissal of those tools.

**Result (measured).**
`172 documents, corpus_hash=ec3aa954b89d8e85facb9ce8ebafd209fa94bb6fab7108993fbe79847cc0b4fa`
— reproducible with `PYTHONPATH=. uv run python scripts/build_manifest.py`.
Verified live: a single deliberately modified byte in one file is caught
by `verify()` and named specifically (`modified: [...]`), not just
flagged as "something changed."

**Cost.** No `retrieval_date` or `legal_version_date` recorded per document
yet — the manifest currently answers "is this exact byte content still
present," not "when was this legally last updated." `source_url` is
reconstructed from the filename pattern, not fetched from each page's own
metadata. Both are addable later without changing the hash mechanism.

## Not yet done

- **`docs/` still has no documented provenance** — unlike `data/converted/`,
  which now does. Reconciling the two (`T4.1`) means deciding whether
  `docs/` gets replaced by a converted, manifested corpus, or the
  acquisition pipeline gets pointed at producing `docs/`'s existing format
  instead.
- **Amendments scope not yet acted on** — see Scope decision's Cost above;
  deferred to evidence from Phase 3's evaluation set, not decided here.
