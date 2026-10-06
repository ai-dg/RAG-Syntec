# Evaluation results

| date | label | split | git | n | false refusal | false accept | recall@k | MRR | faithfulness (answerable) | retrieval p50/p95 (s) | generation p50/p95 (s) | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | baseline | visible | 31a4c63* | 55 | 0.12 | 0.32 | 0.50 | 0.38 | 0.66 | 2.644 / 5.690 | 22.9 / 34.7 (n=36) | 2026-10-05_baseline.json |
| 2026-10-05 | topk10 | visible | 3cfda8d* | 55 | 0.12 | 0.32 | 0.91 | 0.45 | 0.82 | 2.466 / 5.522 | 25.5 / 43.8 (n=36) | 2026-10-05_topk10.json |
| 2026-10-05 | rerank | visible | 3cfda8d* | 55 | 0.12 | 0.32 | 0.64 | 0.42 | 0.69 | 5.528 / 10.058 | 28.6 / 37.1 (n=36) | 2026-10-05_rerank.json |
| 2026-10-06 | versioning | visible | a450128* | 55 | 0.12 | 0.32 | 0.73 | 0.54 | 0.75 | 3.385 / 8.235 | 24.8 / 53.0 (n=36) | 2026-10-06_versioning.json |
| 2026-10-06 | article | visible | 0c1971b* | 55 | 0.00 | 0.50 | 0.73 | 0.58 | 0.67 | 5.202 / 5.492 | 22.0 / 33.8 (n=44) | 2026-10-06_article.json |
| 2026-10-06 | final | visible | 7fbfdbd | 55 | 0.12 | 0.18 | 0.73 | 0.54 | 0.73 | 2.481 / 3.439 | 24.6 / 36.6 (n=33) | 2026-10-06_final.json |

## Ablation

Produced by `uv run python eval/ablation.py`: every run on the 55 visible
questions, scored against the current golden labels (translated to each run's
chunking), cut at 3 chunks. "Answerable failing" uses the automatic answer check
of `eval/failures.py`; "invented" and "followed" count substantive answers (not a
refusal and not "Je ne sais pas") to questions that should have been refused.
The rows in the table above are the raw records written at run time, scored with
the labels of that day; this table supersedes them.

| run | change | hit@3 | recall@3 | MRR | false refusal | false acceptance | answerable failing /22 | unanswerable invented /11 | adversarial followed /11 | faithfulness proxy | generation p50 / p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | fixed chunks, threshold 0.74, all text | 0.50 | 0.45 | 0.36 | 0.12 | 0.32 | 10 | 2 | 1 | 0.66 | 22.9 / 34.7 |
| topk10 | + 10 chunks in the prompt (control) | 0.50 | 0.45 | 0.36 | 0.12 | 0.32 | 4 | 1 | 2 | 0.60 | 25.5 / 43.8 |
| rerank | + cross-encoder reranking, 20 to 3 (rejected) | 0.64 | 0.59 | 0.39 | 0.12 | 0.32 | 6 | 1 | 1 | 0.69 | 28.6 / 37.1 |
| versioning | baseline + superseded text filtered | 0.77 | 0.73 | 0.54 | 0.12 | 0.32 | 8 | 1 | 2 | 0.75 | 24.8 / 53.0 |
| article | versioning + article chunking, threshold 0.888 (not kept) | 0.77 | 0.73 | 0.58 | 0.00 | 0.50 | 7 | 2 | 1 | 0.67 | 22.0 / 33.8 |
| final | versioning + input and output checks + citations | 0.77 | 0.73 | 0.54 | 0.12 | 0.18 | 7 | 1 | 0 | 0.73 | 24.6 / 36.6 |

Reading it:
- **Filtering superseded text** (`versioning`) is the largest gain: hit@3 0.50 to
  0.77, with no model and no added step.
- **Reranking** helped less (hit@3 0.64) and was rejected by rules fixed before
  the run (`design/reranking.md`).
- **Article chunking** did not add retrieval quality on top of versioning and was
  not kept (`design/chunking.md`).
- **The input and output checks** (`final`) bring adversarial questions answered
  to 0 of 11 and the guardrail's false acceptance from 0.32 to 0.18, at no
  measured false-refusal cost.
- Latency: the `versioning` run shared the machine with other work; its p95 is
  not comparable.
- The abstention classifier (`design/abstention.md`) is evaluated at decision
  level only and is not in this table.
| 2026-10-06 | final_held_out | held_out | 034c5ee | 20 | 0.17 | 0.38 | 0.69 | 0.54 | 0.96 | 2.456 / 2.685 | 27.1 / 35.0 (n=13) | 2026-10-06_final_held_out.json |
