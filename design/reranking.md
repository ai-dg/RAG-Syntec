# Reranking: what is decided before it is built

Scope: the cross-encoder stage planned in Phase 11 (T11.1 design, T11.2
implementation, T11.4 measurement). Everything below the "Result" heading was
measured on the baseline and the `top_k = 10` control; nothing about the
reranker itself has been measured yet.

## Decision: retrieve 20 candidates, rerank them, keep 3 for the prompt

**Problem.** The baseline's recall@3 is 0.50 and its MRR 0.38 (22 answerable
questions). The ranking data (`design/retrieval_diagnosis.md`) shows every
answerable question has a relevant chunk in the top 10, and the retrieval
failures sit at ranks 5 to 10, only 0.02 to 0.13 farther than the best chunk
returned. These are ranking failures, not coverage failures. The cheap
alternative, `top_k = 10`, fixes 6 of the 9 failing answerable questions but
leaves the ranking unchanged and costs +11% on the median generation time and
+26% on its p95.

**Decision.** Fetch 20 candidates by dense distance, score each (question,
chunk) pair with a cross-encoder, keep the best 3 for the prompt. The guardrail
keeps thresholding the dense distance of the best candidate, before reranking.

**Why.**
- A bi-encoder embeds the question and each chunk separately; chunk vectors are
  computed once at indexing and the score is a distance between two vectors. A
  cross-encoder reads the question and the chunk together, so words of the
  question can attend to words of the chunk (a negation, an exception, a
  figure). That is the mechanism by which it can separate near-identical
  articles that a vector distance puts a few hundredths apart.
- It costs one model call per candidate and nothing can be precomputed. Scoring
  all 8 447 chunks per question is therefore out of reach, which is why the
  design is "retrieve wide, then rerank". The first draft of this document
  assumed about 50 ms per pair on CPU. Measured, it is 8 to 25 times higher (see
  "Feasibility check" below): about 0.4 s per pair in float32, so 20 pairs cost
  about 8 s, 50 pairs about 20 s and 8 447 pairs about 53 minutes.
- Width 20: the deepest rank of a relevant chunk in the failures is 10, and
  the hit rate is already 1.00 at k = 10, 20 and 50. 20 leaves a margin of 2
  over rank 10, which matters because the `top_k = 10` run still missed q026 (a
  chunk at rank 10 in the 50-chunk ranking). 50 adds cost (2.5 times the pairs)
  for no measured benefit; keep it as a variant only if T11.4 shows a ceiling.
- Cut 3: that is the prompt size of the baseline. The point of reranking is to
  get the benefit of a wide list without paying for a long prompt.

## Model choice and counter-evidence

- **Model.** The roadmap names Qwen3-Reranker-0.6B, with the claims: Apache 2.0
  licence, more than 100 languages, a reported 65.80 on MTEB-R against 57.03 for
  BGE-reranker-v2-m3. These are claims taken from the roadmap and have not been
  checked against the model card here.
- **Counter-evidence.** The roadmap also records that the ARAGOG study found no
  advantage from a commercial reranker. That result does not transfer by
  itself: this corpus is French legal text, about 20% of the index is repeated
  text, and the starting retriever is already strong (every relevant chunk is
  in the top 10). Whether a reranker helps here is a question for the measured
  result, and a negative result would be a valid outcome.

## Rules fixed before any reranker result is seen

The reranker is kept only if all three hold on the visible split, compared with
both the `baseline` and the `topk10` rows:
1. recall@3 (by id, as in the evaluation runner) rises by at least 0.15 over the
   baseline, i.e. reaches at least 0.65; the text-equivalent recall is reported
   next to it because repeated text can make a correct retrieval look like a
   miss;
2. no more than 4 of the 22 answerable questions fail (the `top_k = 10` level)
   with only 3 chunks in the prompt, and generation p95 stays within 10% of the
   baseline's 34.7 s;
3. the added retrieval p95 latency is at most 3 s.

Otherwise it is rejected and the result recorded as a measured negative, as
the roadmap asks.

## Result (measured, without a reranker)

| measure | value |
|---|---|
| answerable questions with a relevant chunk in the top 1 / 3 / 5 / 10 / 20 / 50 | 0.27 / 0.55 / 0.73 / 1.00 / 1.00 / 1.00 |
| rank of the right chunk in the 5 retrieval failures (id) | 10, 7, 5, 5, 8 |
| `top_k = 10` control: answerable failures | 9 to 4 |
| `top_k = 10` control: generation p50 / p95 | 22.9 / 34.7 s to 25.5 / 43.8 s |

The reranker itself: not yet measured. It will be, in T11.4, by running the
evaluation harness with reranking off and then on, changing nothing else.

## Feasibility check (measured, throw-away environment)

Done in a separate Python 3.12 environment, outside the project's dependencies
(torch 2.14.1 CPU build, sentence-transformers 6.1.0), on this machine (32
logical cores, 15 GB of RAM, no model loaded in Ollama during the test).

- **Licence and loading.** The model card states `apache-2.0` and gives
  `CrossEncoder("Qwen/Qwen3-Reranker-0.6B")` as the loading route. The
  architecture is `Qwen3ForCausalLM`: the model scores a pair from the
  probability of "yes" against "no" (a `LogitScore` module), wrapped behind the
  cross-encoder interface. It is a 596-million-parameter language model, not a
  classification head. The weights are 1 207 MB on disk.
- **It is used correctly.** On one obviously relevant and one obviously
  irrelevant pair, the relevant one scores higher (about 8 against about -9.5).
  This checks the call, not the quality on the project's questions.
- **Latency on CPU, 20 pairs of a question and a real chunk (mean 432
  characters), batches of 1, 4 or 20:**

| precision | threads | ms per pair | 20 pairs |
|---|---|---|---|
| bfloat16 (default load) | 24 | 1 077 to 1 308 | 21 to 26 s |
| float32 | 24 | 381 to 403 | 7.6 to 8.1 s |
| float32 | 12 | 502 to 599 | 10 to 12 s |
| float32 | 6 | 389 to 486 | 7.8 to 9.7 s |

  Peak resident memory: about 2.7 GB in bfloat16 and 4.8 GB in float32 (weights
  plus activations).

**What the CPU numbers do to the decision rules.** Rule 3 (added p95 retrieval
latency at most 3 s) cannot be met with 20 candidates on this CPU: about 8 s in
float32 and about 25 s in the default bfloat16. The rule is not changed,
because it was fixed before any result; the design has to change to meet it.

### GPU placement (measured, same throw-away setup with a CUDA build of torch)

The roadmap chose the CPU to avoid contention with Ollama for the 8 GB of GPU
memory. That contention was measured instead of assumed (RTX 4060 Laptop, 8 188
MiB; Ollama serving `qwen3-embedding:8b` for retrieval and `gemma4` for
generation, which alternate and are swapped in and out of memory).

| setting (20 pairs, float16) | result |
|---|---|
| reranker alone on the GPU | 0.33 s (16 ms per pair), 24 times faster than CPU float32 |
| GPU memory: weights / peak during scoring | 1 136 MiB / 1 815 MiB |
| GPU memory used with the embedding model loaded and the reranker resident | 7 718 MiB of 8 188 |
| generation with gemma4 while the reranker stays resident | failed with HTTP 500 (error body not captured) |
| same generation, reranker not in GPU memory (control) | succeeded, 7.7 s including the model swap |
| reranker kept in CPU memory, moved to the GPU only while scoring | 0.75 to 1.55 s per request (to GPU 0.14 to 0.45 s, scoring 0.34 to 0.70 s, back to CPU 0.23 to 0.40 s) |
| three full cycles with that pattern: embedding, rerank, generation | no error; generation 5.3 to 6.1 s each |

So keeping the reranker resident in GPU memory does contend with Ollama (the
generation model could not be loaded), and the control shows the failure comes
from the reranker, not from the generation call. Borrowing the memory for the
duration of the scoring avoids it, at the price of 0.75 to 1.55 s per request
(3 cycles, the slowest being the first; this is not a p95).

The same test also measured retrieval's embedding call when it alternates with
generation: 6.4 s, 6.5 s and 2.6 s, against 0.07 to 0.09 s when the embedding
model is already loaded. This matches the baseline's retrieval latency (p50 2.6
s, p95 5.7 s) and supports the earlier guess that it comes from Ollama swapping
the embedding model back in after each generation. The reranker does not change
it.

**Design consequence.** The placement to implement and measure in T11.2 is: model
held in CPU memory (float16, about 1.2 GB), moved to the GPU for each request and
released afterwards, with 20 candidates. Fewer candidates (10 cost about 4 s on
the CPU) are no longer needed to meet rule 3, and they would risk losing the
chunks at rank 10 that the reranker is meant to promote.

**Points still to check in T11.2.** Latency of one real request under load
(Ollama generating at the same time), which was not measured here; whether the
quality on the project's questions matches the sanity check; and the effect of
input length (chunks here are about 430 characters, the limit being the
500-character chunk size).

**Score semantics.** The reranker's score is not a distance and is on an
unrelated scale; the guardrail threshold of 0.74 must never be applied to it.
Letting the guardrail read the reranker score is deferred to T11.3, after the
abstention classifier exists.

## Cost: what this does not establish

- The 20-candidate width and the cut of 3 follow from 22 questions and five
  failures. They are choices, not derivations.
- The decision rules are bets about acceptable trade-offs, fixed now so that
  they cannot be adjusted after seeing the numbers.
- The latency was measured on 20 pairs, in a throw-away environment, with the
  machine otherwise idle except for Ollama; the GPU numbers come from three
  cycles, so they carry no percentile.
- Dependency weight is a real cost, measured: the environment with the CPU
  build of torch and sentence-transformers takes 1.2 GB, the one with the CUDA
  build takes 6.9 GB, plus 1.2 GB for the model download. The machine has about
  19 GB of free disk. A container image would grow by the same amount, and the
  first run downloads the model.
- The generator can still fail when the right chunk is in the prompt (q028,
  q024, q002): reranking does not address that, and no rule above counts it as
  a reranker failure.
