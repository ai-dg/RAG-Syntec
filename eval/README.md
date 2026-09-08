# Evaluation set

## Record schema

Each labelled example is one JSON object, validated against
[`schema.json`](schema.json). Fields:

| Field | Meaning |
|---|---|
| `id` | A unique identifier for this example (e.g. `q001`). |
| `question` | The question text, as a real user would phrase it. |
| `class` | One of the four classes below — the single correct pipeline behaviour for this question. |
| `relevant_chunk_ids` | Ground truth for **retrieval quality**, independent of generation quality — which chunk(s) actually contain the answer. Empty for `off_topic` and `adversarial` (nothing in the corpus is "relevant" to them by definition). |
| `expected_answer_gist` | What a correct answer should say, in a sentence or two — not a word-for-word match target, a human-checkable gist. |
| `notes` | Anything a future reader needs to interpret this example: why it was judged unanswerable, an ambiguity, a tie-break decision. |
| `corpus_hash` | The `corpus_hash` from `data/manifest.json` at labelling time. If the current corpus hash doesn't match, this example's `relevant_chunk_ids` may no longer point at the same content — treat the example as stale until re-verified. |

## The four classes

| Class | Correct behaviour | Where it should be decided |
|---|---|---|
| `in_topic_answerable` | Retrieve, answer, cite. | — |
| `in_topic_unanswerable` | Retrieve (guardrail passes — the question *is* on-topic), then answer "I don't know". | `app/services/generation.py` |
| `off_topic` | Refuse before generation is ever called. | `app/services/retrieval.py`'s guardrail |
| `adversarial` | Refuse; never leak the system prompt. Phrased with in-domain vocabulary — topic is not the signal, intent is. | Guardrail and/or generation, depending on how close the phrasing lands to the corpus |

**Why `in_topic_unanswerable` and `off_topic` stay separate, even though both end in
a refusal:** each names a different correct checkpoint in the pipeline, and
therefore a different failure mode if the eval fails.

- If an example labelled `in_topic_unanswerable` gets rejected by the
  guardrail (`context_found: False`) instead of reaching generation, that's
  either a mislabelling (the question wasn't really on-topic) or a
  guardrail-calibration bug — a **retrieval-layer** problem.
- If an example labelled `in_topic_unanswerable` passes the guardrail but
  the LLM invents an answer instead of saying "I don't know", that's a
  **generation-layer** problem (hallucination).
- If an example labelled `off_topic` passes the guardrail at all, that's a
  **guardrail** problem (threshold too permissive).

Collapsing both classes into one "refused correctly" bucket would make it
impossible to tell which of these three, structurally different bugs
occurred — exactly the failure-taxonomy question `T3.7` depends on being
answerable.

**Why `relevant_chunk_ids` is labelled at all, instead of only the final
answer:** it isolates retrieval quality (recall@k, precision@k — can the
pipeline find the right chunk) from generation quality (did the LLM phrase
a correct answer from that chunk) as two independent, separately
measurable failure surfaces. An example where the wrong chunk was
retrieved but the LLM still produced a plausible-sounding wrong answer is a
retrieval bug; an example where the right chunk was retrieved but the
answer is still wrong is a generation bug. Grading only the final answer
text collapses these into one undebuggable "wrong" verdict.

## Labelling protocol

**Deciding a question is genuinely unanswerable, not just under-searched.**
Read the full relevant article(s) — not a keyword search that returned
nothing — before concluding the corpus doesn't address a question. Record
which article(s) were checked in `notes`, so the judgement is verifiable
by someone else later, not just asserted.

**Choosing `relevant_chunk_ids` on ties.** When two or more chunks state
the same or near-identical information (a real risk in this corpus — see
`design/corpus.md`'s note on multiple dated `textes-salaires` versions and
duplicated `(non en vigueur)` articles), include the chunk(s) that are
currently in force, and note in `notes` which alternate chunk(s) were
excluded and why. Do not include a superseded version alongside the active
one just to be safe — that would make recall@k trivially satisfiable
without the pipeline actually finding the *current* answer.

**Verifying `corpus_hash` before committing an example.** Before adding a
row to `golden.jsonl` (`T3.2`), run
`PYTHONPATH=. uv run python scripts/build_manifest.py` and copy the current
`corpus_hash` from `data/manifest.json` — never hand-type or reuse an old
value from a previous session.
