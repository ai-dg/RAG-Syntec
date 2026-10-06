# Abstention: deciding whether to answer, as a supervised problem

Status: formulated (T10.1), pipeline built (T10.2), trained and evaluated (T10.3,
results below once run), wired behind `GUARDRAIL_MODE=classifier` (T10.4).

## Problem

The guardrail answers when the best chunk is closer than a fixed distance (0.74).
One number cannot tell apart the cases that matter: on the baseline it let 7 of 11
adversarial questions and 7 of 11 in-topic-but-unanswerable questions through,
because both are written about the agreement's own subject, so their nearest
chunk is close. A question can be close to the corpus and still have no answer
in it.

## Formulation

- **Label.** 1 (answer) for `in_topic_answerable`; 0 (refuse) for
  `in_topic_unanswerable`, `off_topic` and `adversarial`. This is stricter than the
  distance guardrail, which only tries to refuse off-topic questions: the
  classifier is asked to refuse whatever the system cannot answer correctly.
- **Features**, all computed before generation from the 10 nearest in-force
  chunks and the question (`app/services/features.py`): best distance, mean and
  standard deviation of the 5 best, gaps between rank 1 and ranks 2, 5 and 10,
  number of chunks within 0.05 of the best, question length in words. None reads
  the golden label or the answer. Sparse/dense rank agreement, listed in the
  roadmap, is not available because hybrid search was not built.
- **Why dispersion can help.** A question with a real answer tends to have one
  or a few chunks clearly closer than the rest (a large gap); a question about
  the domain with no answer tends to sit at a similar distance from many chunks
  (a flat profile). The best distance alone does not see the shape.
- **Baseline to beat:** the distance threshold, scored on the same label.
- **Why not accuracy.** With 22 positives and 33 negatives, "always refuse" is
  60% accurate and useless. The comparison uses ROC-AUC and PR-AUC (ranking
  quality, threshold-free) and, at the chosen operating point, false refusals
  and false acceptances separately.
- **Model.** Logistic regression on standardised features, class-weighted: with
  55 rows, a model with 9 parameters is the most that can be estimated, and its
  coefficients can be read. Gradient boosting is compared under the same
  cross-validation, as a check on whether a non-linear model finds more.
- **Protocol.** Repeated stratified 5-fold cross-validation (10 repeats) on the
  visible split only. The operating point is fixed from out-of-fold
  probabilities: the highest cut that refuses at most 10% of answerable
  questions. Only then is the model fitted on the whole visible split and scored
  once on the held-out split (20 questions, 8 answerable).
- **Extra negatives.** The roadmap allows generating negatives if the classes
  are too skewed. 33 refuse against 22 answer is not skewed enough to justify
  synthetic questions, which would also be easier than real ones; none were added.
- **Storage.** The model is a JSON file of coefficients and scaling (no pickle):
  inference needs only numpy, and loading it cannot execute code.

## Prediction, written before training

- Cross-validated ROC-AUC of the logistic regression within 0.05 of the
  threshold's, either side: the features are all derived from the same distances,
  so most of their information is already in the best distance.
- Refuted if the classifier's ROC-AUC exceeds the threshold's by more than 0.10.
- On 20 held-out questions no difference will be distinguishable from noise (one
  question is 0.05 of a rate).

## Result (measured, `scripts/train_abstention.py`, features from the versioning index)

Label: answer only `in_topic_answerable`. Visible split: 55 questions (22 to
answer). Held-out split: 20 (8 to answer), read once after the operating point
was fixed.

| | ROC-AUC | PR-AUC | false refusal | false acceptance |
|---|---|---|---|---|
| distance threshold 0.74, visible | 0.898 | 0.789 | 0.00 (0/22) | 0.42 (14/33) |
| logistic regression, 5-fold CV x10, out-of-fold | 0.862 (sd 0.017) | 0.784 | 0.09 (2/22) | 0.18 (6/33) |
| gradient boosting, 5-fold CV x10 | 0.864 (sd 0.026) | 0.705 | | |
| distance threshold, held-out | 0.969 | 0.966 | 0.00 (0/8) | 0.42 (5/12) |
| logistic regression, held-out | 0.979 | 0.975 | 0.00 (0/8) | 0.33 (4/12) |

Operating point: probability 0.415, the highest cut that refuses at most 10% of
answerable questions out of fold.

**Prediction checked:** "ROC-AUC within 0.05 of the threshold's" held
(-0.036). The classifier does not rank questions better than the best distance
alone. Gradient boosting does no better than the linear model.

**What the classifier does buy:** at its operating point it refuses 8 more of the
33 questions that should be refused, at the cost of 2 of 22 answerable ones
(cross-validated). On the held-out split the difference is one question, which
is within noise for 20 examples.

**Coefficients (standardised; negative = more likely to refuse).** The mean
distance of the top 5 (-1.22) and the best distance (-1.12) dominate, as
expected. Question length is next (-0.68): longer questions are refused more
often; in this golden set, adversarial questions are the long ones, so this is
a property of the set, not a reliable signal. Dispersion features carry little
weight.

**Decision:** `GUARDRAIL_MODE=threshold` stays the default. The classifier is
available behind the switch, with a measured trade-off: fewer wrong
acceptances, a few more wrong refusals. Its end-to-end effect after generation
was not measured (each run takes about 25 minutes of GPU), so these figures are
decision-level, before the model writes anything.

**Cost.** 55 training examples; the question-length feature may learn the style
of this golden set; the held-out split was used once here, so it can no longer
give an unbiased number for any decision that follows from this table.
