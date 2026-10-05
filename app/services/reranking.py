
import logging
import threading
import time
from typing import Callable

from langchain_core.documents import Document


logger = logging.getLogger(__name__)
_model = None
_lock = threading.Lock()
RERANK_BATCH_SIZE = 4



Scorer = Callable[[str, list[str]], list[float]]   # higher = more relevant

def load_reranker(model_name: str) -> None:
    global _model
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranking needs the 'rerank' extra: uv sync --extra rerank"
        ) from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    start = time.perf_counter()
    _model = CrossEncoder(
        model_name, 
        device="cpu", 
        model_kwargs={"torch_dtype": dtype}
    )
    
    logger.info(
        "reranker %s loaded in %.1f s (%s)", model_name, time.perf_counter() - start, dtype
    )

def _score_with_borrowed_gpu(question: str, texts: list[str]) -> list[float]:
    import torch

    if _model is None:
        raise RuntimeError("Reranker not loaded: call load_reranker() first")

    pairs = [(question, text) for text in texts]
    with _lock:
        if not torch.cuda.is_available():
            return [float(s) for s in _model.predict(pairs)]
        _model.model.to("cuda")
        try:
            scores = _model.predict(pairs, batch_size=RERANK_BATCH_SIZE)
        finally:
            _model.model.to("cpu")
            torch.cuda.empty_cache()
    return [float(s) for s in scores]

    
def rerank(
    question: str,
    candidates: list[tuple[Document, float]],
    top_n: int,
    scorer: Scorer | None = None,
) -> list[tuple[Document, float]]:
    if not candidates:
        return []

    scorer = scorer or _score_with_borrowed_gpu
    start = time.perf_counter()
    scores = scorer(question, [doc.page_content for doc, _ in candidates])
    if len(scores) != len(candidates):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} candidates")

    scored = []
    for (doc, distance), score in zip(candidates, scores):
        copy = Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score})
        scored.append((copy, distance))
    scored.sort(key=lambda pair: pair[0].metadata["rerank_score"], reverse=True)

    logger.info("reranked %d candidates in %.2f s", len(candidates), time.perf_counter() - start)
    return scored[:top_n]