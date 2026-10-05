# Baseline index: what indexing the Syntec corpus costs

Scope: the unmodified ingestion stack (Markdown loader, `RecursiveCharacterTextSplitter`
with `CHUNK_SIZE=500` / `CHUNK_OVERLAP=100`, `qwen3-embedding:8b` through Ollama, Chroma)
applied to the manifested corpus in `data/converted/`. This is the "before" every later
parser, chunker or embedding change is compared against.

## Decision: keep the old stack unchanged for the baseline

**Problem.** If the corpus, the parser, the chunker and the embedding model all change at
once, no later improvement can be attributed to any one of them.

**Decision.** The baseline runs the old stack on the new corpus; one variable changes at a
time from here on.

**Why.** A controlled comparison needs a fixed reference. The reference is
`eval/results/2026-10-05_baseline.json` (55 questions, visible split), produced on this
index (8 447 chunks, corpus hash recorded in the result file).

## Result (measured)

One full indexing run on an idle GPU (RTX 4060 Laptop, 8 188 MiB), into a throw-away
Chroma directory, with GPU memory sampled once per second:

| measure | value |
|---|---|
| documents | 172 |
| chunks | 8 447 |
| load + split | under 0.1 s |
| embedding + Chroma write | 468 s (7.8 min), 18.1 chunks/s |
| GPU memory before / peak | 13 MiB / 5 609 MiB |

The peak is the embedding model alone: the chat model was not loaded during this run.

**Relevance threshold.** Recalibrated after switching to this corpus (see
`design/guardrail.md`): 0.736 before, 0.7404 after, kept at 0.74. The corpus roughly
doubled and the threshold moved by 0.004. The distance gap between in-topic and off-topic
questions went from `0.701` to `0.780`.

## Cost: what this does not establish

- **One run.** No repetition, so no spread on the duration.
- **GPU memory is sampled at 1 s**, so the true peak may be slightly higher; it is a lower
  bound. With the chat model loaded as well (query time), memory use is higher: not
  measured here.
- **Why the threshold barely moved is a hypothesis, not a measurement.** The mechanism
  that predicted a larger move is real: a question's score is the minimum distance over all
  chunks, and the minimum over more chunks tends to be lower. The likely reason it stayed
  small is that the added content belongs to the same domain, so an off-topic question
  stays far from all of it whatever the chunk count. This was not tested by indexing
  subsets of different sizes.
- **The threshold was calibrated on a small question set** (3 in-topic and 3 off-topic
  questions), so it is only a calibration, not an estimate of false-refusal in
  production; the evaluation run measures that separately.
