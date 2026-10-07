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
