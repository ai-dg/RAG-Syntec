# CV framing notes

Every figure below comes from `eval/RESULTS.md` (ablation table) or `design/`,
measured on 55 labelled questions about the Syntec collective agreement. Keep the
denominators when space allows: they are what makes the numbers credible.

## Bullets (English)

- Built a question-answering system over the Syntec collective agreement that
  cites the passage behind every answer (0 invented citations) and refuses
  unsupported questions with a stated reason; the guardrail's false-acceptance
  rate fell from 0.32 to 0.18 with no added false refusals.
- Diagnosed retrieval failures with a labelled evaluation set and found 40% of
  the index was superseded legal text; filtering it raised the share of
  questions with a relevant passage in the top 3 from 50% to 77%, without
  adding a model.
- Tested and rejected a cross-encoder reranker and a trained abstention
  classifier against decision rules fixed before each run (reranker: +0.136
  recall@3 for +4.4 s at p95; classifier: ROC-AUC 0.862 against 0.898 for the
  distance threshold).

- Benchmarked BGE-M3 against qwen3-embedding:8b on the same pipeline: 34% faster
  median answer (27 s to 18 s), 2 times faster indexing and 6 times less GPU
  memory, but 18 points lower retrieval (top-3 hit rate 0.59 against 0.77); kept
  the larger model for accuracy on legal text.

Stack line: Python, FastAPI, LangChain (ingestion), Chroma, Ollama
(qwen3-embedding:8b, BGE-M3, gemma4), scikit-learn, pytest, Docker.

## Lignes de CV (français)

- Conception d'un système de questions-réponses sur la convention collective
  Syntec qui cite le passage de chaque réponse (0 citation inventée) et refuse
  les questions sans réponse dans le texte, en donnant la raison : taux de fausse
  acceptation du filtre de 0,32 à 0,18, sans refus à tort supplémentaire.
- Diagnostic des échecs de recherche sur un jeu de questions étiqueté : 40 % de
  l'index était du texte juridique abrogé ; l'exclure a porté la part de
  questions avec un passage pertinent parmi les 3 premiers de 50 % à 77 %, sans
  modèle supplémentaire.
- Comparaison de BGE-M3 et de qwen3-embedding:8b sur le même pipeline : réponse
  médiane 34 % plus rapide (27 s à 18 s), indexation 2 fois plus rapide, 6 fois
  moins de mémoire GPU, mais 18 points de recherche en moins (0,59 contre 0,77) ;
  le modèle le plus précis est conservé pour un texte juridique.
- Test puis rejet d'un reranker et d'un classifieur d'abstention selon des
  règles fixées avant chaque mesure (reranker : +0,136 de rappel@3 pour +4,4 s
  au p95 ; classifieur : ROC-AUC 0,862 contre 0,898 pour le seuil de distance).

## Oral pitch (two sentences)

EN: I built a question-answering system on a French collective agreement that
refuses when the text does not support an answer, and I measured every change on
a labelled question set. The biggest gain did not come from a model: 40% of the
index was repealed text, and filtering it took retrieval from 50% to 77%.

FR : J'ai construit un système qui répond aux questions sur une convention
collective et qui refuse quand le texte ne permet pas de répondre, en mesurant
chaque changement sur un jeu de questions étiqueté. Le plus gros gain n'est pas
venu d'un modèle : 40 % de l'index était du texte abrogé, et l'exclure a fait
passer la recherche de 50 % à 77 %.

## One sentence: what is different

Each component was kept or rejected on numbers measured before and after, and
two of them (a reranker, a trained classifier) were rejected.

## To check before using

- [TODO: the target job's vocabulary] Reuse the job offer's terms (RAG,
  retrieval, evaluation, MLOps...) in the bullets; do not copy its slogans.
- The figures are on 55 questions (22 answerable); say "on a labelled set of 55
  questions" if asked, never present them as production traffic.
- Do not claim injections are blocked: 0 of 11 on the visible questions, but 2 of
  4 held-out injections still got an answer (`design/evaluation.md`).
- On the 20 held-out questions retrieval held (hit@3 0.75 against 0.77), which
  you can quote as evidence the 50% to 77% gain is not tuned to the test set.
