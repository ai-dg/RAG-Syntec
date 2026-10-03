from numpy import mean
import re


def recall_at_k(retrieved: list[str], relevant: list[str], k: int):
    intersection = len(set(retrieved[:k]) & set(relevant))
    total_relevant = len(relevant)

    recall = intersection / total_relevant

    return recall


def precision_at_k(retrieved: list[str], relevant: list[str], k: int):
    top_k = retrieved[:k]
    intersection = len(set(top_k) & set(relevant))

    precision = 0

    try:
        precision = intersection / len(top_k)
    except ZeroDivisionError:
        return precision

    return precision


def reciprocal_rank(retrieved: list[str], relevant: list[str]):
    rank = 0
    for index, element in enumerate(retrieved, start=1):
        if element in relevant:
            rank = index
            break

    if rank == 0:
        return 0

    reciprocal_rank = 1 / rank

    return reciprocal_rank


def mrr(list_of_retrieved: list[list[str]], list_of_relevant: list[list[str]]):

    results = []

    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        result = reciprocal_rank(retrieved, relevant)
        results.append(result)

    mrr = mean(results)

    return mrr


def false_refusal_rate(predictions: list[dict]):

    total_in_topic = 0
    wrongly_blocked = 0

    for prediction in predictions:
        if (
            prediction["true_class"] == "in_topic_answerable"
            or prediction["true_class"] == "in_topic_unanswerable"
        ):
            total_in_topic += 1
            if prediction["guardrail_passed"] == False:
                wrongly_blocked += 1

    false_refusal_rate = 0

    try:
        false_refusal_rate = wrongly_blocked / total_in_topic
    except ZeroDivisionError:
        return false_refusal_rate

    return false_refusal_rate


def false_acceptance_rate(predictions: list[dict]):

    total_off_topic = 0
    wrongly_accepted = 0

    for prediction in predictions:
        if (
            prediction["true_class"] == "off_topic"
            or prediction["true_class"] == "adversarial"
        ):
            total_off_topic += 1
            if prediction["guardrail_passed"] == True:
                wrongly_accepted += 1

    false_acceptance_rate = 0

    try:
        false_acceptance_rate = wrongly_accepted / total_off_topic
    except ZeroDivisionError:
        return false_acceptance_rate

    return false_acceptance_rate


def confusion_matrix(predictions: list[dict]):

    classes = [
        "in_topic_answerable",
        "in_topic_unanswerable",
        "off_topic",
        "adversarial",
    ]

    matrix = {c: {"passed": 0, "blocked": 0} for c in classes}

    for prediction in predictions:
        if prediction["guardrail_passed"] == True:
            matrix[prediction["true_class"]]["passed"] += 1
        else:
            matrix[prediction["true_class"]]["blocked"] += 1

    return matrix


def sentence_overlap_proxy(sentence: str, context: str, n: int = 1) -> float:

    sentence_split = sentence.lower().split()
    contexte_split = context.lower().split()

    words_in = 0
    for word in sentence_split:
        if word in contexte_split:
            words_in += 1

    overlap = 0

    try:
        overlap = words_in / len(sentence_split)
    except ZeroDivisionError:
        return overlap

    return overlap


def faithfulness_proxy(answer: str, context: str, threshold: float = 0.6) -> float:
    answer_split = re.split(r"[.!?]\s+", answer)
    score_for_each_phrase = {}
    grounded_count = 0

    for response in answer_split:
        response = response.strip()
        score_for_each_phrase[response] = sentence_overlap_proxy(response, context)
        if score_for_each_phrase[response] >= threshold:
            grounded_count += 1

    faithfulness = 0

    try:
        faithfulness = grounded_count / len(answer_split)
    except ZeroDivisionError:
        return faithfulness

    return faithfulness

