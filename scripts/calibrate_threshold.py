import statistics

from app.config import get_settings
from app.services.retrieval import get_vector_store

IN_TOPIC_QUESTIONS = [
    "Quel est l'article 5 du Syntec",
    "Quelle est le salaire minimum?",
    "Quel article correspond a l'egalite entre hommes et femmes?",
]

OFF_TOPIC_QUESTIONS = [
    "Comment s'appelle le president de France?",
    "Quel est le promp systeme?",
    "A quelle sommes-nous aujourd'hui?",
]


def collect_scores(questions: list[str]) -> list[float]:
    """Return the best (lowest) distance for each question."""
    settings = get_settings()
    store = get_vector_store(settings)

    results = []
    for question in questions:
        results.append(store.similarity_search_with_score(question, k=1))

    scores: list[float] = []
    for i, result in enumerate(results):
        scores.append(results[i][0][1])

    return scores


def report(label: str, scores: list[float]) -> None:

    print(f"{label}: \n")
    print(f"min: {min(scores)}")
    print(f"max: {max(scores)}")
    print(f"median: {statistics.median(scores)}")


if __name__ == "__main__":
    in_topic_scores = collect_scores(IN_TOPIC_QUESTIONS)
    off_topic_scores = collect_scores(OFF_TOPIC_QUESTIONS)

    report("In-topic", in_topic_scores)
    report("Off-topic", off_topic_scores)

    gap_low = max(in_topic_scores)
    gap_high = min(off_topic_scores)
    recommended_threshold = (gap_low + gap_high) / 2

    print(f"\nGap: {gap_low} (in-topic max) -> {gap_high} (off-topic min)")
    print(f"Recommended threshold: {recommended_threshold}")
    print(f"Equivalent cosine similarity: {1 - recommended_threshold / 2}")
