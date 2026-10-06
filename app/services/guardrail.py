"""Input-side and output-side guardrails around the retrieval guardrail.

Input side: a pattern check for prompt-injection phrasing, before any retrieval.
The distance threshold cannot catch injections written about the corpus' own
subject (design/guardrail.md), so this layer looks at the wording instead.

Output side: a lexical support check on the generated answer. An answer whose
sentences mostly do not appear in the retrieved context is downgraded to a
refusal. The prompt already asks the model to answer only from the context, but
that relies on the model complying; this check does not.
"""

import re

INJECTION_PATTERNS = [
    r"\bignore[rz]?\b.{0,40}\b(instruction|consigne|r[eè]gle)s?\b",
    r"\b(oublie|oubliez)\b.{0,40}\b(instruction|consigne|r[eè]gle)s?\b",
    r"\bprompt\s+syst[eè]me\b",
    r"\bsystem\s+prompt\b",
    r"\b(nouvelle|nouvelles)\s+(consigne|instruction)s?\b",
    r"#{2,}\s*(fin|end)\b",
    r"\bd[ée]sactive[rz]?\b.{0,40}\b(filtr|garde|s[ée]curit|contr[oô]l)",
    r"\b(tu es|vous [eê]tes|you are)\s+(d[ée]sormais|maintenant|now)\b",
    r"\b(r[eé]ponds|r[eé]pondez|answer)\b.{0,40}\bsans\s+(citer|source|v[ée]rif)",
    r"\b(applique|appliquez|int[eè]gre|int[eé]grez)\b.{0,40}\b(note|consigne|instruction)\b",
    r"\bignore (all|previous|the above)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

WORD = re.compile(r"[a-zàâçéèêëîïôûùüÿœ0-9%]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
ABSTENTION_OPENINGS = ("je ne sais pas", "je ne peux pas", "je suis désolé", "désolé")


def detect_injection(question: str) -> str | None:
    """The first injection pattern the question matches, or None."""
    for pattern in _COMPILED:
        if pattern.search(question):
            return pattern.pattern
    return None


def _content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 3 or w.isdigit()}


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().replace("’", "'").startswith(ABSTENTION_OPENINGS)


def support_score(answer: str, context: str, sentence_threshold: float = 0.5) -> float:
    """Share of the answer's sentences whose content words mostly occur in the context."""
    context_words = _content_words(context)
    sentences = [
        s
        for s in SENTENCE_END.split(re.sub(r"\s*\[\d+\]", "", answer))
        if _content_words(s)
    ]
    if not sentences:
        return 1.0
    supported = sum(
        1
        for s in sentences
        if len(_content_words(s) & context_words) / len(_content_words(s))
        >= sentence_threshold
    )
    return supported / len(sentences)
