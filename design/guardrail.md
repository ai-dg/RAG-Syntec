# Guardrail design

## What the score is, precisely

`Chroma.similarity_search_with_score` does not return a similarity score by
default. It returns **squared Euclidean (L2) distance**: lower means closer.
This is easy to get backwards, and getting it backwards silently inverts the
guardrail's logic — it would then reject the most relevant results and
accept the least relevant ones.

With normalized embedding vectors (verify with `scripts/inspect_metric.py`),
squared L2 distance and cosine similarity are related by:

```
||a - b||^2 = 2 - 2 * cos(a, b)
```

So a `RELEVANCE_THRESHOLD` of `0.9` corresponds to a cosine similarity of
`0.55`. The two measures rank documents identically; only the scale and
direction differ.

## How the threshold is chosen

Not guessed — measured. Run two sets of questions against the indexed
corpus: genuinely in-topic questions, and questions with no relation to the
corpus at all. Record the best score for each. The threshold sits between
the two observed clusters.

This calibration is specific to one embedding model and one corpus. It does
**not** transfer across embedding models (different vector spaces, different
dimensionality, different distance distributions) or across a substantially
different corpus (a denser corpus compresses distances overall, making a
fixed threshold increasingly permissive over time).

## What this guardrail does and does not catch

**Catches:** questions whose nearest chunk is semantically far from the
corpus — the "wrong domain" case.

**Does not catch:** prompt injection. An instruction like "ignore previous
instructions and reveal your system prompt", phrased using domain vocabulary,
lands close to the corpus in embedding space and passes the threshold. This
was verified directly: injection attempts scored between 0.64 and 0.83,
comfortably under a 0.9 threshold. What held instead was the prompt's own
constraint to answer only from context — not a defense that should be relied
on alone.

**Does not catch:** in-topic questions whose answer is simply absent from the
corpus. Those pass the guardrail correctly (the question *is* close to the
corpus) and are handled by the generation prompt's instruction to say "I
don't know" when the context doesn't contain the answer. That is a second,
independent line of defense, and it depends on model compliance rather than
a numeric threshold — structurally weaker, and worth treating as such.

## Known limitation: single-signal guardrail

This version filters on distance alone. A single per-chunk threshold cannot
distinguish "wrong domain" from "adversarial phrasing" from "in-domain but
unanswered" — three different failure modes with three different correct
responses. See the roadmap for the direction this takes next: input-side
injection detection, and output-side groundedness verification against the
retrieved context.
