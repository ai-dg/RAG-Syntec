"""One question through every layer: input check, retrieval and decision,
generation, output check. Used by the API and by the evaluation runner, so the
evaluation measures exactly what the API serves."""

import time

from app.config import get_settings
from app.services import retrieval
from app.services.generation import generate
from app.services.guardrail import detect_injection, is_abstention, support_score


def answer_question(question: str) -> dict:
    settings = get_settings()
    timings = {}

    if settings.input_guard:
        pattern = detect_injection(question)
        if pattern:
            return {
                "answered": False,
                "retrieval": None,
                "refusal_reason": "prompt_injection_detected",
                "matched_pattern": pattern,
                "latency_ms": timings,
            }

    start = time.perf_counter()
    retrieved = retrieval.retrieve(question)
    timings["retrieval"] = (time.perf_counter() - start) * 1000
    if not retrieved["context_found"]:
        return {
            "answered": False,
            "retrieval": retrieved,
            "refusal_reason": retrieved.get("refusal_reason"),
            "latency_ms": timings,
        }

    start = time.perf_counter()
    generated = generate(question, retrieved)
    timings["generation"] = (time.perf_counter() - start) * 1000

    if settings.output_guard and not is_abstention(generated["answer"]):
        context = "\n\n".join(doc.page_content for doc, _ in retrieved["chunks"])
        score = support_score(generated["answer"], context)
        generated["support_score"] = score
        if score < settings.groundedness_threshold:
            return {
                "answered": False,
                "retrieval": retrieved,
                "generation": generated,
                "refusal_reason": "ungrounded_answer",
                "latency_ms": timings,
            }

    return {
        "answered": True,
        "retrieval": retrieved,
        "generation": generated,
        "refusal_reason": None,
        "latency_ms": timings,
    }
