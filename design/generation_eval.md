# Generation evaluation: faithfulness proxy and LLM judge

Scope: how the system measures whether a generated answer is supported by
the retrieved context (faithfulness, in RAGAS vocabulary), and how far each
measurement can be trusted. Code: `eval/metrics.py` (`sentence_overlap_proxy`,
`faithfulness_proxy`, `judge_agreement_rate`) and the judge
(`judge_faithfulness`, model `llama3.2:3b` served by Ollama).

## Decision 1: two faithfulness signals, the lexical proxy is primary

**Problem.** Faithfulness is a question of meaning ("does the context support
every claim in the answer?"). The only direct measurements are a human or an
LLM judge. Human review does not scale to every run; an LLM judge is a model
whose own reliability is unknown until measured.

**Decision.** Report a cheap lexical proxy as the primary faithfulness number
and the LLM judge as a secondary, advisory signal, until the judge's
agreement with human labels is measured as good enough.

**Why.** `faithfulness_proxy` splits the answer into sentences, scores each
with `sentence_overlap_proxy` (share of its words that also appear in the
context), and returns the share of sentences scoring at least the threshold.
It is deterministic, free, and needs no model. It is a proxy: it correlates
with faithfulness without being it. It fails on paraphrase (faithful, few
shared words) and on negation ("does NOT provide for notice" reuses the same
words as "provides for notice"). The judge reads meaning, but it is a 3B
model whose verdicts must be checked against a human, not assumed.

**Cost.** Both signals are imperfect, and they fail differently. The proxy
cannot see meaning; the judge can misread the text (see Result).

## Decision 2: the judge is never trusted without a measured agreement rate

**Problem.** Using a model to grade another model's output is circular
unless the grader's own reliability is measured independently.

**Decision.** Hand-label a set of (question, context, answer) triples, run
the judge on them, and report `judge_agreement_rate` (share of examples where
judge verdict == human verdict). A verdict the judge could not give in the
required format (`faithful = None`) counts as a disagreement, so it never
inflates the rate.

**Why.** Agreement with an independent human label is the only evidence that
the judge measures what we want. Counting unparseable output as disagreement
is the conservative reading: an answer the judge failed to grade was not
graded correctly.

## Result (measured)

**Labelling set** (`eval/judge_labelling.jsonl`, 20 rows, 12 distinct
questions from the visible golden split): 10 answers copied from
`expected_answer_gist` and 10 answers deliberately altered (a changed number,
an inverted yes/no, an added claim absent from the context). Eight questions
appear twice, once faithful and once altered, with the same context. Context
is the single corpus chunk that best covers the expected answer (word
coverage), then checked by eye. Labels were assigned blind by the project
author using only the context.

- Human labels vs. the intended design: **20/20** identical. The set is
  consistent.
- **Judge vs. human: 11/20 = 55%.** Judge verdicts: 2 `true`, 16 `false`,
  2 `None`.

| | judge `true` | judge `false` | judge `None` |
|---|---|---|---|
| human `true` (10) | 2 | 6 | 2 |
| human `false` (10) | 0 | 9 | 1 |

An always-`false` judge would score 50% on this set, so 55% is barely above
doing nothing. The judge rarely accepts a wrong answer but rejects most
correct ones. Examples where it is simply wrong: the context says word for
word that restaurant vouchers are kept under hybrid work, and the judge
answers that the context does not mention it; it also objects that "the
regulatory quota applies" is imprecise when that is exactly what the context
states.

- **Proxy vs. human:** 0.60 at sentence threshold 0.6, **0.75** at 0.8,
  0.50 at 0.9. The threshold was swept after looking at these labels, on
  these same 20 examples, so 0.75 is an optimistic number, not a validated one.

## What failed on the way (kept because it changed the method)

A first labelling set built from the golden set's own `relevant_chunk_ids`
produced only 2 faithful labels out of 20: the cited chunks often did not
contain the expected answer (e.g. the chunk cited for the question on how
often the point values are renegotiated came from an unrelated document).
On that set an always-`false` judge would have scored 90%, which would have
hidden the judge's weakness behind class imbalance. It was discarded
(`eval/judge_labelling_v1_unusable.jsonl`) and rebuilt from verified chunks.
Questions whose answer needs two chunks, or whose expected answer contains
details absent from the chunk, were left out of the rebuilt set.

### Golden set chunk ids audited and corrected

The same finding meant the ground truth for retrieval metrics was suspect.
Audit: for each of the 30 `in_topic_answerable` questions, the share of the
expected answer's content words found in the cited chunks. 12 of 30 were
below 0.50. Identifier offsets between the cited and the best-matching chunk
were irregular (-56, -51, -13, -8, -5, +13, +17, +25), so this was not a
uniform renumbering; the cause is not established. A word-overlap audit is a
heuristic, so each of the 12 was reviewed by hand against candidate chunks:
9 were replaced with a chosen candidate, 3 were resolved by reading the
neighbouring chunks (a list or a rule split across a chunk boundary). After
the fix, no answerable question is below 0.50 coverage and every cited id
exists in the current chunking. This corrects wrong ground truth, not a
tuning choice, but it touched the held-out file too (q006, q015, q020), so
any number produced from that split before this date is not comparable.

Cost: the corpus contains the same article several times (base text and
amendments, sometimes verbatim), so a question can have equally valid chunks
in several places and only one is listed. Recall@k can under-count a correct
retrieval of the duplicate. For q010 and q018 the chosen copy was not
checked for being the one in force.

## Cost: what this does not establish

- **n = 20.** With 20 examples the standard error on a rate near 55% is
  about 11 points, so "55%" means roughly 35% to 75%. It supports "not
  reliable", not a precise figure.
- **One labeller.** There is no inter-annotator agreement.
- **Synthetic errors.** Altered answers are single-fact edits. Real
  generator hallucinations may be subtler (or blunter) and could change both
  scores.
- **Ideal context.** The judge was given the best chunk, not what the
  retriever actually returns, so retrieval errors are not in the picture.
- **Tuned threshold.** The proxy's best threshold was chosen on the
  evaluation examples. It needs a separate labelled set (the held-out split
  exists for questions, not yet for faithfulness labels) before the 0.75 can
  be quoted.
- **Proxy blind spots:** negation and paraphrase (above), and tokenization by
  whitespace keeps punctuation attached, so the last word of an answer such
  as `préavis.` fails to match `préavis` in the context.
- **Judge prompt not tuned.** The 55% is for one prompt with temperature 0;
  a different prompt or a larger judge model may change it. Not measured.

## Consequence for the pipeline

Until a judge reaches a defensible agreement rate on a labelled set it was not
tuned on, its verdicts are reported as advisory, next to the proxy and always
with this 55% figure. The evaluation runner (T3.6) reports both signals
separately and never merges them into one number.
