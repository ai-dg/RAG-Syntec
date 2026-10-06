# The biggest gain in my RAG system did not come from a model

*Draft for LinkedIn or dev.to. Every number links to a file in the repository.*

I built a question-answering system on the Syntec collective agreement, the
French agreement that sets notice periods, leave and minimum salaries for about
a million engineering and consulting employees [TODO: check the employee count
before publishing, or remove it]. A wrong answer on a legal text is worse than no
answer, so the system has to refuse when the text does not support an answer,
and say why.

I wanted every change to be justified by a measurement, so the first thing I
built was not the system but the test: 75 questions, each labelled with its class
(answerable from the agreement, about the agreement but not answerable, off-topic,
or a prompt injection), the passage that answers it, and the expected answer. 55
are used to make decisions, 20 are kept aside.

## The baseline, and where it failed

The starting point was ordinary: 500-character chunks, an embedding model, the 3
closest chunks, a distance threshold to refuse off-topic questions. Half of the
answerable questions did not get a relevant passage among the 3 retrieved.

Before changing anything, I looked at *why*. For every answerable question, the
right passage was among the 10 closest, at ranks 5 to 10 for the failures, only
slightly farther than what was returned. Information was not missing; it was
ranked too low.

## Two things I tried that did not work

**A reranker.** A cross-encoder reads the question and each candidate passage
together, which should separate near-identical articles. Before running it I
wrote down the rule that would reject it: +0.15 recall@3, no worse than simply
sending 10 passages, at most 3 s of added latency. It reached +0.136, failed more
questions than the 10-passage control, and added 4.4 s. Worse, for one question
it promoted an older version of an article and the system answered with a rule
that is no longer in force: 2 years of seniority for a severance indemnity, where
the text in force says 8 months. Rejected.

**A trained classifier to decide when to refuse.** Eight features from the search
results, a logistic regression, cross-validated. It ranked questions *worse* than
the best distance alone (ROC-AUC 0.862 against 0.898). It is in the code behind a
switch, not on by default.

## What worked: reading the corpus

The reranker's mistake pointed at something I had not measured: the corpus keeps
superseded versions of articles, marked "(non en vigueur)" in their headings. I
counted: **40% of the index** was superseded text, and it made up nearly half of
the passages the baseline put in its top 3.

Excluding it from the search, a metadata filter and no model, raised the share
of questions with a relevant passage in the top 3 **from 50% to 77%**. It also
exposed five of my own labels that pointed at outdated text; one expected answer
was itself out of date ("twice a year" where the text in force says once). The
test set had a bug, and the fix to the system found it.

## The guardrails around it

Two cheap layers complete the threshold: a pattern check for injection phrasing
before the search, and an output check that withdraws an answer whose sentences
do not appear in the retrieved passages. Together they took the injections the
model obeyed from 1 of 11 to 0 of 11 with no added wrong refusal. The pattern
check is weak, and I say so: it catches 4 of 11 injections in the questions I
wrote it from, and 0 of 4 in the questions kept aside.

## What I take from it

- Write the rule that would reject a change before measuring it. Two of my three
  components failed their own rule; without it I would have kept them.
- Look at the data before adding a model. The largest gain was a filter.
- Keep the negative results in the write-up. They are why the positive one is
  believable.

The code, the 75 questions, every result file and the decisions behind them:
[TODO: repository link].
