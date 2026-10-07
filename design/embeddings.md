# Embeddings: qwen3-embedding:8b against BGE-M3

Status: measured below (T7.1 to T7.3). Phase 7 had been skipped because the
diagnosis found no coverage failure; it was run to measure the cost side
(speed, memory) that the diagnosis did not cover.

## Problem

The embedding model `qwen3-embedding:8b` (7.6 billion parameters, 4.68 GB) and
the chat model `gemma4` do not fit together in the 8 GB of GPU memory. Ollama
swaps them, and the embedding call for a question that follows a generation was
measured at 2.6 to 6.5 s, against 0.07 to 0.09 s when the model is already loaded
(`design/reranking.md`). Indexing the corpus takes 468 s.

## Decision

Measure BGE-M3 served by Ollama (`bge-m3`, 566.7 million parameters, F16, 1 024
dimensions, BERT family) as a drop-in replacement: same provider path, only
`EMBEDDING_MODEL_LOCAL` changes. Dense vectors only; BGE-M3's sparse output would
need hybrid search (Phase 9), which is not built.

**Why BGE-M3.** It is about 13 times smaller than the current model, multilingual,
and has a context of 8 192 tokens, larger than any chunk here. A smaller model
should index faster and leave room on the GPU next to the chat model.

**Geometry checked before anything else.** The vectors returned by Ollama have
norm 1.000, and the squared L2 distance equals 2 - 2 cos (0.627 both ways on a
question and its answer, 1.484 for an off-topic question). So the distance keeps
its meaning and the guardrail logic is unchanged; only the threshold value must
be recalibrated, since distances from two models are not comparable.

## Prediction, written before the measurement

From `design/retrieval_diagnosis.md` (T5.3), plus latency:

| measure | expected | refuted if |
|---|---|---|
| hit@3 / recall@3 | within 0.10 of the current model (no real gain) | improves by more than 0.10 |
| indexing time | at least 3 times faster than 468 s (under 156 s) | slower than 156 s |
| GPU memory peak while indexing | below 5 609 MiB | above |
| the 0.74 threshold applied unchanged | shifts false refusal or false acceptance by more than 0.10 | holds within 0.05 |
| retrieval latency p50 in an evaluation run | below 1 s (no more swapping of a 4.7 GB model) | above 2 s |
| generation latency | unchanged within 15% | |

## Result (measured)

**Indexing** (`scripts/` timing, GPU memory sampled every second): 230.6 s for
the 8 447 chunks (36.6 chunks per second) against 468 s; GPU memory 925 MiB
against 5 609 MiB. 4 chunks made Ollama return "unsupported value: NaN" (see
the fallback below); they are included.

**Evaluation** (run `2026-10-07_bge-m3`): the final configuration with only the
embedding model changed, threshold recalibrated on the visible split to 0.914
(0.74 would refuse 14 of 33 in-topic questions), compared with run `final`:

| measure | qwen3-embedding:8b | BGE-M3 | prediction |
|---|---|---|---|
| hit@3 / recall@3 / MRR | 0.77 / 0.73 / 0.54 | 0.59 / 0.55 / 0.42 | refuted: worse by 0.18, beyond the 0.10 margin |
| answerable questions failing (of 22) | 7 | 9 | |
| unanswerable invented / adversarial followed | 1 / 0 | 1 / 0 | |
| guardrail false refusal / false acceptance | 0.12 / 0.18 | 0.03 / 0.27 | |
| retrieval latency p50 / p95 | 2.48 / 3.44 s | 0.10 / 0.15 s | held (below 1 s) |
| generation latency p50 / p95 | 24.6 / 36.6 s | 17.8 / 33.9 s | refuted: 28% faster, not unchanged |
| indexing | 468 s | 231 s | refuted: 2 times faster, not 3 |
| GPU memory while indexing | 5 609 MiB | 925 MiB | held |
| the 0.74 threshold kept | | 14 of 33 in-topic refused | held: the threshold does not transfer |

**Why it is faster end to end.** The median answer time falls from about 27 s
to about 18 s. Retrieval becomes 25 times faster because the 1.2 GB model stays
on the GPU next to the chat model, so Ollama no longer swaps a 4.7 GB embedding
model in and out at each question; generation also gains 28%, which is
consistent with the chat model no longer being evicted (not isolated).

**Decision: keep `qwen3-embedding:8b` as the default.** For a legal assistant a
relevant passage matters more than 9 seconds: BGE-M3 loses the relevant passage
for about 4 of 22 answerable questions. BGE-M3 is available by setting
`EMBEDDING_MODEL_LOCAL=bge-m3` and recalibrating the threshold (0.914 measured
here); it is the better choice where answer time or GPU memory is the
constraint.

**Fallback found on the way.** BGE-M3 in F16 through Ollama fails on 4 of 8 447
short, ordinary chunks ("unsupported value: NaN"), and one failure fails the
whole request, since Chroma embeds every chunk in one call. The first fallback
retried text by text and made indexing take 734 s; it now splits the failed
batch in two recursively, and embeds a single failing text doubled
(`ResilientOllamaEmbeddings` in `app/services/ingestion.py`, tested).

## Cost: what this does not establish

- One run per model; latencies depend on what else Ollama holds in memory.
- Dense vectors only: BGE-M3's sparse and multi-vector outputs were not used,
  so hybrid retrieval with it (Phase 9) remains untested.
- The 4 doubled chunks have approximate vectors (cosine 0.875 to 0.963 to the
  original on comparable text).

## Sources

The model card claims are not relied on; everything above is measured.
`design/sources.md` covers the evaluation method (B3) and tail latency (B2).
