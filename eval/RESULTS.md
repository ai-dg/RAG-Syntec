# Evaluation results

| date | label | split | git | n | false refusal | false accept | recall@k | MRR | faithfulness (answerable) | retrieval p50/p95 (s) | generation p50/p95 (s) | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | baseline | visible | 31a4c63* | 55 | 0.12 | 0.32 | 0.50 | 0.38 | 0.66 | 2.644 / 5.690 | 22.9 / 34.7 (n=36) | 2026-10-05_baseline.json |
| 2026-10-05 | topk10 | visible | 3cfda8d* | 55 | 0.12 | 0.32 | 0.91 | 0.45 | 0.82 | 2.466 / 5.522 | 25.5 / 43.8 (n=36) | 2026-10-05_topk10.json |
| 2026-10-05 | rerank | visible | 3cfda8d* | 55 | 0.12 | 0.32 | 0.64 | 0.42 | 0.69 | 5.528 / 10.058 | 28.6 / 37.1 (n=36) | 2026-10-05_rerank.json |
| 2026-10-06 | versioning | visible | a450128* | 55 | 0.12 | 0.32 | 0.73 | 0.54 | 0.75 | 3.385 / 8.235 | 24.8 / 53.0 (n=36) | 2026-10-06_versioning.json |
| 2026-10-06 | article | visible | 0c1971b* | 55 | 0.00 | 0.50 | 0.73 | 0.58 | 0.67 | 5.202 / 5.492 | 22.0 / 33.8 (n=44) | 2026-10-06_article.json |
