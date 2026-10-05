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
  design is "retrieve wide, then rerank". With a placeholder of about 50 ms per
  pair on CPU (a hypothesis, to be measured in T11.2), 20 pairs cost about 1 s,
  50 pairs about 2.5 s and 8 447 pairs about 7 minutes.
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

## Points to check before implementing (T11.2)

- **How the model loads.** The roadmap assumes `sentence-transformers`. That
  library is not installed here, and rerankers built on a language model often
  score a pair through a prompt template and the probability of "yes" or "no",
  not through a classification head. If so, `rerank()` has a different shape
  and the latency estimate changes. Check the model card and attempt a real
  load first.
- **CPU latency and memory.** The 50 ms per pair above is a placeholder. Measure
  it on the real machine (32 cores, 15 GB of RAM, with Ollama also running). At
  full precision a model of 0.6 billion parameters needs about 2.4 GB for its
  weights alone (parameter count times 4 bytes, an estimate).
- **Score semantics.** The reranker's score is not a distance and is on an
  unrelated scale; the guardrail threshold of 0.74 must never be applied to it.
  Letting the guardrail read the reranker score is deferred to T11.3, after the
  abstention classifier exists.

## Cost: what this does not establish

- The 20-candidate width and the cut of 3 follow from 22 questions and five
  failures. They are choices, not derivations.
- The decision rules are bets about acceptable trade-offs, fixed now so that
  they cannot be adjusted after seeing the numbers.
- The latency figures for the reranker are placeholders until T11.2.
- The generator can still fail when the right chunk is in the prompt (q028,
  q024, q002): reranking does not address that, and no rule above counts it as
  a reranker failure.
