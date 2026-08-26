"""
What does the guardrail's score actually measure?

Answers a question anyone reading this repo should ask: "the threshold is
0.9 — 0.9 of what?". Read-only: it inspects the live collection and
decomposes one real score by hand.

Usage (from the project root, with the app already indexed once):

    uv run python scripts/inspect_metric.py

This calls the real embedding provider. In OpenAI mode it costs a small
amount of money. It is not named test_*, so pytest ignores it.

Method:

1. Read the collection's actual distance metric (`hnsw.space`). Chroma
   defaults to squared L2; this choice is invisible in application code,
   it's a library default.

2. Check whether the embedding model's vectors are normalized, by measuring
   the L2 norm of a stored vector. If it's 1, the vectors sit on the unit
   sphere, and the identity below holds:

       ||a - b||^2 = 2 - 2 * cos(a, b)

   Squared Euclidean distance is then a strictly decreasing function of
   cosine similarity — same ranking, different scale.

3. Recompute the distance between a real question and its best-matching
   chunk by hand, and compare four quantities (Chroma's own score, raw L2,
   squared L2, 2-2cos) to identify unambiguously which one Chroma returns.

Concludes with the cosine similarity equivalent to the configured threshold.
"""

import numpy as np
import chromadb

from app.config import Settings
from app.services.ingestion import get_embedding, COLLECTION_NAME
from app.services.retrieval import get_vector_store

QUESTION = "How does this system decide when to answer?"
SEPARATOR = "=" * 62


def main() -> None:
    settings = Settings()
    store = get_vector_store(settings)  # builds the index if not already built

    client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = client.get_collection(COLLECTION_NAME)
    space = collection._model.configuration_json["hnsw"]["space"]

    print(SEPARATOR)
    print("1. COLLECTION CONFIGURATION")
    print(SEPARATOR)
    print(f"collection   : {collection.name}")
    print(f"distance     : {space}")
    print(f"vector count : {collection.count()}")

    stored = collection.get(limit=1, include=["embeddings"])
    vector = np.array(stored["embeddings"][0])
    norm = float(np.linalg.norm(vector))
    normalized = abs(norm - 1.0) < 1e-3

    print()
    print(SEPARATOR)
    print("2. VECTOR SPACE GEOMETRY")
    print(SEPARATOR)
    print(f"provider   : {settings.llm_provider}")
    print(f"dimension  : {vector.shape[0]}")
    print(f"vector norm: {norm:.6f}")
    print("normalized : " + ("yes (unit sphere)" if normalized else "no"))

    embedding = get_embedding(settings)
    doc, score = store.similarity_search_with_score(QUESTION, k=1)[0]

    qv = np.array(embedding.embed_query(QUESTION))
    dv = np.array(embedding.embed_documents([doc.page_content])[0])

    l2 = float(np.linalg.norm(qv - dv))
    cos = float(qv @ dv)

    print()
    print(SEPARATOR)
    print("3. DECOMPOSING A REAL SCORE")
    print(SEPARATOR)
    print(f"question      : {QUESTION}")
    print(f"chunk source  : {doc.metadata.get('source')}")
    print()
    print(f"Chroma's score      : {score:.6f}")
    print(f"raw L2               : {l2:.6f}")
    print(f"squared L2           : {l2 ** 2:.6f}")
    print(f"cosine similarity    : {cos:.6f}")
    print(f"2 - 2*cos            : {2 - 2 * cos:.6f}")

    print()
    print(SEPARATOR)
    print("4. CONCLUSION")
    print(SEPARATOR)

    if abs(score - l2**2) < 1e-4:
        print("Chroma returns SQUARED EUCLIDEAN DISTANCE.")
        print("This is NOT a similarity score: lower means closer.")
        if normalized:
            print()
            print("Vectors are normalized, so score = 2 - 2*cos.")
            print(
                f"RELEVANCE_THRESHOLD = {settings.relevance_threshold} corresponds "
                f"to a cosine similarity of {1 - settings.relevance_threshold / 2:.3f}."
            )
    elif abs(score - l2) < 1e-4:
        print("Chroma returns raw Euclidean distance (not squared).")
    else:
        print("The score doesn't match either distance tested.")
        print("Check the metric declared on the collection.")


if __name__ == "__main__":
    main()
