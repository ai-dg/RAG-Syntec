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
