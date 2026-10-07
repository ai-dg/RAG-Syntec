# Sources behind each design choice

Every source below was opened and read on 2026-10-07 (publisher page, table of
contents, free online text or abstract); the quote is from that page. Where no
outside source supports a choice, the choice rests on this project's own
measurement, and that is said explicitly. A source supports the *method*; the
*result* on this corpus always comes from the measurement in `design/`.

## Books

- **[B1]** C. Huyen, *AI Engineering*, O'Reilly, 2025. Table of contents read
  at https://github.com/chiphuyen/aie-book (ToC.md). Chapters used: 3
  "Evaluation Methodology" (section "AI as a Judge"); 4 "Evaluate AI Systems"
  ("Design Your Evaluation Pipeline": evaluate all components, create an
  evaluation guideline, define evaluation methods and data); 5 "Prompt
  Engineering" ("Defensive Prompt Engineering": jailbreaking and prompt
  injection, defenses against prompt attacks); 6 "RAG and Agents" ("RAG
  Architecture", "Retrieval Algorithms", "Retrieval Optimization"); 10 "AI
  Engineering Architecture and User Feedback" ("Put in Guardrails").
- **[B2]** B. Beyer, C. Jones, J. Petoff, N. R. Murphy (eds.), *Site Reliability
  Engineering*, O'Reilly, 2016, chapter "Monitoring Distributed Systems",
  https://sre.google/sre-book/monitoring-distributed-systems/ — the four golden
  signals and "Worrying About Your Tail": "If you run a web service with an
  average latency of 100 ms at 1,000 requests per second, 1% of requests might
  easily take 5 seconds."
- **[B3]** C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information
  Retrieval*, Cambridge University Press, 2008, chapter "Evaluation in
  information retrieval",
  https://nlp.stanford.edu/IR-book/html/htmledition/information-retrieval-system-evaluation-1.html
  (a test collection is documents, information needs and relevance judgments;
  "a document is relevant if it addresses the stated information need"; tune on a
  development collection and test on a fresh one for an unbiased estimate; about
  50 information needs as a minimum) and
  https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
  (precision at k is "the least stable of the commonly used evaluation measures").

## Papers and official documentation

- **[P1]** P. Rajpurkar, R. Jia, P. Liang, "Know What You Don't Know: Unanswerable
  Questions for SQuAD", ACL 2018, https://arxiv.org/abs/1806.03822 — systems must
  "determine when no answer is supported by the paragraph and abstain from
  answering."
- **[P2]** Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural
  Networks", 2017, https://arxiv.org/abs/1705.08500 — "the classifier rejects
  instances as needed, to grant the desired risk".
- **[P3]** T. Saito, M. Rehmsmeier, "The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
  Datasets", PLOS ONE 10(3), 2015,
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
- **[P4]** L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena", NeurIPS 2023 Datasets and Benchmarks, https://arxiv.org/abs/2306.05685
  — judges show "position, verbosity, and self-enhancement biases, as well as
  limited reasoning ability"; GPT-4 reaches "over 80% agreement" with humans.
- **[P5]** OWASP, "LLM01:2025 Prompt Injection", OWASP Top 10 for LLM
  Applications 2025, https://genai.owasp.org/llmrisk/llm01-prompt-injection/ —
  "it is unclear if there are fool-proof methods of prevention for prompt
  injection."
- **[P6]** K. Greshake et al., "Not what you've signed up for: Compromising
  Real-World LLM-Integrated Applications with Indirect Prompt Injection", 2023,
  https://arxiv.org/abs/2302.12173 — "LLM-Integrated Applications blur the line
  between data and instructions."
- **[P7]** T. Gao, H. Yen, J. Yu, D. Chen, "Enabling Large Language Models to
  Generate Text with Citations", EMNLP 2023, https://arxiv.org/abs/2305.14627 —
  "even the best models lack complete citation support 50% of the time".
- **[P8]** N. F. Liu et al., "Lost in the Middle: How Language Models Use Long
  Contexts", TACL 2023, https://arxiv.org/abs/2307.03172 — performance "significantly
  degrades when models must access relevant information in the middle of long
  contexts."
- **[P9]** P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks", NeurIPS 2020, https://arxiv.org/abs/2005.11401 — "providing
  provenance for their decisions and updating their world knowledge remain open
  research problems."
- **[P10]** A. Jimeno Yepes et al., "Financial Report Chunking for Effective
  Retrieval Augmented Generation", 2024, https://arxiv.org/abs/2402.05131 —
  "element type based chunking largely improve RAG results on financial
  reporting".
- **[P11]** Kubernetes documentation, "Liveness, Readiness, and Startup Probes",
  https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/
  — "Readiness probes determine when a container is ready to accept traffic."
- Reranking sources (Nogueira & Cho 2019, Sentence-Transformers "Retrieve &
  Re-Rank", ARAGOG 2024, Qwen3 Embedding 2025) and Ragas (Es et al. 2023) are
  listed in the README and `design/reranking.md`.

## Choice by choice

| choice | design doc | sources | what the source does and does not settle |
|---|---|---|---|
| Labelled golden set with relevant chunks, held-out split | `evaluation.md`, `eval/README.md` | B3, B1 ch. 4 | Method: relevance judgments per information need, tune on one set and test on another. B3 suggests about 50 information needs; this set has 55 visible, 22 of them answerable, which is why small differences are treated as noise. |
| recall@k, precision@k, MRR | `evaluation.md` | B3 | Precision at k is unstable on small sets (B3), consistent with treating one question (0.045) as noise. MRR's definition was not checked in B3. |
| False refusal and false acceptance read together, PR-AUC, no accuracy | `evaluation.md`, `abstention.md` | P3 | PR curves are more informative on imbalanced data; here 22 answer against 33 refuse. |
| Refusing unanswerable questions as a separate class | `eval/README.md`, `abstention.md` | P1 | Abstaining when no answer is supported is part of the task, not an error. |
| Abstention classifier with an operating point fixed from an accepted error | `abstention.md` | P2, B1 ch. 10 | Rejecting instances to reach a target risk is the selective-classification setting; whether it beats the distance here is this project's measurement (it did not). |
| LLM judge measured before use | `generation_eval.md` | P4, B1 ch. 3 | Judges have known biases; even strong judges reach about 80% agreement. The 3B judge here reached 55%, so it is not used. |
| Superseded text filtered at query time | `versioning.md` | P9 | Updating knowledge is an open problem of RAG (P9). The "(non en vigueur)" marking comes from Légifrance's own headings. No outside source measures this filter: the gain (hit@3 0.50 to 0.77) is this project's measurement. |
| Article-based chunking (tested, not kept) | `chunking.md` | P10 | Structure-aware chunking helped on financial reports (P10); on this agreement it brought no gain. A positive result elsewhere does not transfer. |
| Keeping 3 passages rather than 10 | `retrieval_diagnosis.md` | P8 | Longer contexts degrade use of information in the middle, beyond the latency cost measured here. |
| Reranking (tested, rejected) | `reranking.md` | README sources | Unchanged. |
| Citations with invented numbers rejected | `evaluation.md` | P7, P9 | Even strong models often lack citation support (P7), so citations are checked against retrieved passages, and support of each sentence is not assumed. |
| Input and output guardrail layers | `guardrail.md` | P5, P6, B1 ch. 5 and 10 | No fool-proof prevention exists (P5); injections blur data and instructions (P6). This is why the input check is documented as a filter, not a defence, which the held-out result confirmed. |
| p50/p95 latency, not the mean | `evaluation.md`, `observability.md` | B2 | Tail latency is what users notice; the mean hides it. |
| `/health` versus `/ready` | `observability.md` | P11 | Liveness decides restarts, readiness decides traffic. |
| Writing predictions before measuring | `retrieval_diagnosis.md` | none verified | The preregistration literature (Nosek et al. 2018) could not be opened; the practice is used here without a cited source. |

## Not verified, not cited

Berryman and Ziegler, *Prompt Engineering for LLMs* (O'Reilly, 2024), and the
section names of Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022):
the publisher pages could not be read, so neither is cited.
